import torch
import random

# Prompt
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

class SharedModelAttacker:
    """
    共享基座模型的Attacker
    不加载新模型，而是将Attacker LoRA挂载到现有的Policy模型上
    """
    def __init__(self, base_model, tokenizer, attacker_adapter_path, adapter_name="attacker"):
        self.model = base_model # 引用主模型
        self.tokenizer = tokenizer
        self.adapter_name = adapter_name
        self.device = base_model.device
        
        # 确保Tokenizer配置正确
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        disable_thinking_mode(self.tokenizer)

        print(f"[Attacker] Loading Adapter from {attacker_adapter_path} as '{adapter_name}'...")
        
        # 加载Attacker的LoRA权重
        # 此时模型会有两个Adapter: "default" (Anonymizer) 和 "attacker" (Attacker)
        self.model.load_adapter(attacker_adapter_path, adapter_name=adapter_name)

    def compute_privacy_reward(self, anonymized_texts, profiles):
        """
        计算隐私奖励: Attacker越难猜出隐私(Loss越高), Reward越高
        此函数自动切换Adapter，计算完后切回原状态
        """
        # 1. 保存当前正在训练的Adapter (通常是 "default")
        previous_adapter = self.model.active_adapter
        
        # 2. 切换到Attacker Adapter并设为eval
        self.model.set_adapter(self.adapter_name)
        self.model.eval() 
        
        rewards = torch.zeros(len(anonymized_texts), device=self.device)

        try:
            batch_prompts = []
            batch_targets = []
            valid_indices = []

            # 数据构造逻辑
            for i, (text, profile) in enumerate(zip(anonymized_texts, profiles)):
                valid_attrs = [k for k, v in profile.items() 
                              if v and str(v).lower() not in ["none", "null", ""]]
                if not valid_attrs:
                    continue
                
                target_key = random.choice(valid_attrs)
                target_val = str(profile[target_key])
                
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

            if not batch_prompts:
                return rewards

            # Tokenize
            inputs_prompt = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, add_special_tokens=False)
            inputs_target = self.tokenizer(batch_targets, return_tensors="pt", padding=True, add_special_tokens=False)
            
            input_ids_list = []
            labels_list = []
            max_len = 0
            eos_id = self.tokenizer.eos_token_id
            
            for p_ids, t_ids in zip(inputs_prompt.input_ids, inputs_target.input_ids):
                full_ids = torch.cat([p_ids, t_ids, torch.tensor([eos_id])])
                label_ids = torch.cat([torch.full_like(p_ids, -100), t_ids, torch.tensor([eos_id])])
                input_ids_list.append(full_ids)
                labels_list.append(label_ids)
                max_len = max(max_len, len(full_ids))

            pad_id = self.tokenizer.pad_token_id
            final_input_ids = []
            final_labels = []
            final_attention_masks = []
            
            for ids, lab in zip(input_ids_list, labels_list):
                pad_len = max_len - len(ids)
                final_input_ids.append(torch.cat([ids, torch.full((pad_len,), pad_id, dtype=torch.long)]))
                final_labels.append(torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)]))
                final_attention_masks.append(torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)]))

            input_tensor = torch.stack(final_input_ids).to(self.device)
            label_tensor = torch.stack(final_labels).to(self.device)
            mask_tensor = torch.stack(final_attention_masks).to(self.device)

            # Forward计算
            with torch.no_grad():
                outputs = self.model(input_ids=input_tensor, attention_mask=mask_tensor)
                logits = outputs.logits
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = label_tensor[..., 1:].contiguous()
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=-100)
                token_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                
                token_losses = token_losses.view(len(valid_indices), -1)
                valid_tokens = (shift_labels != -100).sum(dim=1).float()
                sample_losses = token_losses.sum(dim=1) / valid_tokens
            
            for idx, loss_val in zip(valid_indices, sample_losses):
                rewards[idx] = loss_val

        finally:
            # 3. 切回Anonymizer Adapter，并恢复训练模式
            self.model.set_adapter(previous_adapter)
            if self.model.training:
                self.model.train()

        return rewards