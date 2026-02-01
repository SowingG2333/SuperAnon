import torch
import os
import wandb
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser, 
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

from privacy_model import SharedModelAttacker
from utility_model import UtilityRewardModel

# Prompt
SYSTEM_PROMPT = """You are an expert text anonymizer.
Your goal is to rewrite the text to protect PII (Personal Identifiable Information) mentioned in the user's profile, while preserving the original meaning and style as much as possible. Just output the anonymized text directly.
"""

# WandB采样Callback
class WandbSampleCallback(TrainerCallback):
    def __init__(self, tokenizer, dataset, num_samples=2, log_steps=50):
        self.tokenizer = tokenizer
        self.sample_dataset = dataset.select(range(num_samples))
        self.log_steps = log_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.log_steps == 0 and state.global_step > 0:
            if not wandb.run: return
            
            # 临时切换到eval并推理
            model.eval()
            records = []
            device = next(model.parameters()).device
            
            for sample in self.sample_dataset:
                input_text = sample['prompt']
                inputs = self.tokenizer(input_text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.6)
                
                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                records.append({
                    "step": state.global_step,
                    "original_text": sample.get('original_text', '')[:100] + "...",
                    "profile": str(sample.get('profile', '')),
                    "full_output": generated_full
                })

            table = wandb.Table(
                columns=["step", "original_text", "profile", "full_output"], 
                data=[list(r.values()) for r in records]
            )
            wandb.log({"sample_generations": table})
            model.train()

def parse_think_content(text):
    """
    更鲁棒的解析函数：
    1. 优先提取 </think> 之后的内容
    2. 如果没有 </think> 但有 <think>，则通过正则去掉 <think>... 块
    3. 如果都没有，返回原文本
    """
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    
    # 如果生成被截断，导致只有<think>没有</think>
    if "<think>" in text:
        # 尝试去掉 <think> 开头的所有内容（这是最坏情况，说明生成失败了）
        # 但为了RL不报错，我们可以返回空字符串或者剩下的部分
        # 更好的策略是：如果只有思考没有正文，这甚至不如原样输出
        return text.replace("<think>", "").strip() 
        
    return text.strip()

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(default="Qwen/Qwen3-1.7B")
    attacker_model_path: str = field(default="model/attacker/checkpoint-402")
    utility_model_path: str = field(default="BAAI/bge-m3")
    train_file: str = field(default="data/train.jsonl")
    privacy_weight: float = field(default=1.0)
    utility_weight: float = field(default=1.0)
    use_4bit: bool = field(default=True)

def main():
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # 显存优化：启用Expandable Segments
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    print(f"Loading Base Model: {script_args.model_name_or_path}")
    
    # 1. 加载基座模型 (Anonymizer和Attacker共用)
    bnb_config = None
    if script_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )

    if script_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 2. 配置并加载Anonymizer LoRA (Adapter 'default')
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    print("Anonymizer Adapter (default) loaded.")

    # 3. 初始化Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 4. 加载Attacker (作为第二个 Adapter 挂载)
    # 使用 SharedModelAttacker
    attacker = SharedModelAttacker(
        base_model=model,
        tokenizer=tokenizer,
        attacker_adapter_path=script_args.attacker_model_path,
        adapter_name="attacker"
    )
    
    # 5. 加载Utility Model (独立加载)
    utility_model = UtilityRewardModel(script_args.utility_model_path, device="cuda")

    # 6. 定义独立的奖励函数 (以便WandB分别记录)
    def privacy_reward_func(prompts, completions, profile, **kwargs):
        """计算隐私奖励 (Normalized)"""
        clean_completions = [parse_think_content(c) for c in completions]
        
        # 计算Loss (0 ~ inf)
        raw_privacy_loss = attacker.compute_privacy_reward(clean_completions, profile)
        
        # 归一化: 1 - exp(-loss/scale) -> 映射到 [0, 1]
        privacy_scale = 3.0
        norm_privacy_scores = 1 - torch.exp(-raw_privacy_loss / privacy_scale)
        
        return script_args.privacy_weight * norm_privacy_scores

    def utility_reward_func(prompts, completions, original_text, **kwargs):
        """计算效用奖励"""
        clean_completions = [parse_think_content(c) for c in completions]
        
        # 计算相似度^3 (0 ~ 1)
        utility_scores = utility_model.compute_score(original_text, clean_completions)
        
        return script_args.utility_weight * utility_scores

    # 7. 数据处理
    def process_data(sample):
        text = sample['text']
        profile = sample['profile']
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Anonymize this text:\n{text}"}
        ]
        # prompt_str = tokenizer.apply_chat_template(
        #     messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        # )

        # 关闭think模式
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        return {
            "prompt": prompt_str,
            "profile": profile, # 确保键名是 profile
            "original_text": text
        }

    dataset = load_dataset("json", data_files=script_args.train_file, split="train")
    dataset = dataset.map(process_data, remove_columns=dataset.column_names)

    # 8. 初始化Callback
    sample_callback = WandbSampleCallback(tokenizer, dataset, log_steps=50)

    # 9. Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[privacy_reward_func, utility_reward_func], # 传入列表
        args=training_args,
        train_dataset=dataset,
        peft_config=None,
        processing_class=tokenizer,
        callbacks=[sample_callback]
    )

    print("Starting GRPO Training...")
    trainer.train()
    
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()