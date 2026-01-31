import torch
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt模板
SYS_PROMPT = "You are a privacy investigator. Extract the specific attribute from the text."
USER_PROMPT = "Text: {text}\nQuestion: What is the author's {k}?"

def disable_thinking_mode(tokenizer):
    """关闭Qwen3的Reasoning模式"""
    if not hasattr(tokenizer, "apply_chat_template"):
        return
    original_apply = tokenizer.apply_chat_template
    def patched_apply(*args, **kwargs):
        kwargs.setdefault("enable_thinking", False)
        return original_apply(*args, **kwargs)
    tokenizer.apply_chat_template = patched_apply

class UniversalAttacker:
    """通用隐私攻击者模型，用于计算隐私奖励""" 
    def __init__(self, model_path, device="cuda"):
        print(f"[Attacker] Loading Universal Attacker from {model_path}...")
        self.device = device
        
        # 1. 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        disable_thinking_mode(self.tokenizer)

        # 2. 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
        ).eval()

    def compute_privacy_reward(self, anonymized_texts, profiles):
        """
        计算隐私奖励: Attacker越难猜出隐私(Loss越高), Reward越高
        
        Args:
            anonymized_texts: Defender生成的脱敏文本列表
            profiles: 每个样本对应的真实Profile列表
        Returns:
            torch.Tensor: 每个样本的隐私奖励 (batch_size,)
        """
        batch_prompts = []
        batch_targets = []
        valid_indices = []

        # 1. 构造Batch数据
        for i, (text, profile) in enumerate(zip(anonymized_texts, profiles)):
            valid_attrs = [k for k, v in profile.items() 
                          if v and str(v).lower() not in ["none", "null", ""]]
            if not valid_attrs:
                continue
            
            # 随机抽取一个属性进行攻击
            target_key = random.choice(valid_attrs)
            target_val = str(profile[target_key])
            
            # 构造输入消息
            messages = [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(text=text, k=target_key)},
            ]
            prompt_str = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            batch_prompts.append(prompt_str)
            batch_targets.append(target_val)
            valid_indices.append(i)

        # 无有效样本时返回零奖励
        if not batch_prompts:
            return torch.zeros(len(anonymized_texts), device=self.device)

        # 2. Tokenize
        inputs_prompt = self.tokenizer(
            batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False
        )
        inputs_target = self.tokenizer(
            batch_targets, return_tensors="pt", padding=True, add_special_tokens=False
        )
        
        # 3. 拼接input_ids和labels
        input_ids_list = []
        labels_list = []
        max_len = 0
        eos_id = self.tokenizer.eos_token_id
        
        for p_ids, t_ids in zip(inputs_prompt.input_ids, inputs_target.input_ids):
            # [Prompt] + [Target] + [EOS]
            full_ids = torch.cat([p_ids, t_ids, torch.tensor([eos_id])])
            # Label: Prompt部分为-100(忽略), Target部分保留
            label_ids = torch.cat([
                torch.full_like(p_ids, -100), 
                t_ids, 
                torch.tensor([eos_id])
            ])
            input_ids_list.append(full_ids)
            labels_list.append(label_ids)
            max_len = max(max_len, len(full_ids))

        # 4. Padding到同一长度
        pad_id = self.tokenizer.pad_token_id
        final_input_ids = []
        final_labels = []
        final_attention_masks = []
        
        for ids, lab in zip(input_ids_list, labels_list):
            pad_len = max_len - len(ids)
            final_input_ids.append(
                torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
            )
            final_labels.append(
                torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            )
            final_attention_masks.append(
                torch.cat([torch.ones(len(ids), dtype=torch.long), 
                          torch.zeros(pad_len, dtype=torch.long)])
            )

        # 转为Tensor并移到设备
        input_tensor = torch.stack(final_input_ids).to(self.device)
        label_tensor = torch.stack(final_labels).to(self.device)
        mask_tensor = torch.stack(final_attention_masks).to(self.device)

        # 5. Forward计算Per-Sample Loss
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_tensor,
                attention_mask=mask_tensor,
            )
            logits = outputs.logits
            
            # Shift logits and labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = label_tensor[..., 1:].contiguous()
            
            # 计算每个token的loss
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1)
            )
            
            # 计算每个样本的平均loss
            token_losses = token_losses.view(len(valid_indices), -1)
            valid_tokens = (shift_labels != -100).sum(dim=1).float()
            sample_losses = token_losses.sum(dim=1) / valid_tokens
            
        # 6. 填充结果
        rewards = torch.zeros(len(anonymized_texts), device=self.device)
        for idx, loss_val in zip(valid_indices, sample_losses):
            rewards[idx] = loss_val

        return rewards