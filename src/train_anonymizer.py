import torch
import os
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    HfArgumentParser, 
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

from privacy_model import UniversalAttacker
from utility_model import UtilityRewardModel

# System Prompt
SYSTEM_PROMPT = """You are an expert text anonymizer.
Your goal is to rewrite the text to protect PII (Personal Identifiable Information) mentioned in the user's profile, while preserving the original meaning and style as much as possible. Just output the anonymized text directly.
"""

def parse_think_content(text):
    """从completion中提取实际内容（去除<think>块）"""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "基座模型的HuggingFace ID或本地路径"}
    )
    attacker_model_path: str = field(
        default="model/checkpoint-268",
        metadata={"help": "Attacker模型路径"}
    )
    utility_model_path: str = field(
        default="BAAI/bge-m3",
        metadata={"help": "Utility模型路径"}
    )
    train_file: str = field(
        default="data/train.jsonl",
        metadata={"help": "训练数据路径"}
    )
    privacy_weight: float = field(
        default=1.0,
        metadata={"help": "隐私奖励权重"}
    )
    utility_weight: float = field(
        default=1.0,
        metadata={"help": "效用奖励权重"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4-bit量化 (QLoRA)"}
    )

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ScriptArguments, GRPOConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading model: {script_args.model_name_or_path}")
    print(f"Training data: {script_args.train_file}")

    # 2. 加载奖励模型
    print("Loading Reward Models...")
    attacker = UniversalAttacker(script_args.attacker_model_path, device="cuda")
    utility_model = UtilityRewardModel(script_args.utility_model_path, device="cuda")

    # 3. 定义复合奖励函数
    def composite_reward_func(prompts, completions, original_text, profiles, **kwargs):
        clean_completions = [parse_think_content(c) for c in completions]
        privacy_scores = attacker.compute_privacy_reward(clean_completions, profiles)
        utility_scores = utility_model.compute_score(original_text, clean_completions)
        
        final_rewards = (script_args.privacy_weight * privacy_scores) + \
                       (script_args.utility_weight * utility_scores)
        return final_rewards

    # 4. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, 
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. 数据处理函数
    def process_data(sample):
        text = sample['text']
        profile = sample['profile']
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Anonymize this text:\n{text}"}
        ]
        prompt_str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return {
            "prompt": prompt_str,
            "profile": profile,
            "original_text": text
        }

    # 6. 加载并处理数据
    dataset = load_dataset("json", data_files=script_args.train_file, split="train")
    dataset = dataset.map(process_data, remove_columns=dataset.column_names)

    # 7. 配置 LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 8. 配置4-bit量化并加载模型
    bnb_config = None
    if script_args.use_4bit:
        print("Applying 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",            
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager",    )

    # 准备量化训练所需的特殊层
    if script_args.use_4bit:
        model = prepare_model_for_kbit_training(model) # gradient checkpointing handled by Trainer or manually if needed

    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 9. 配置Trainer
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=composite_reward_func,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 10. 开始训练
    print("Starting GRPO Training...")
    trainer.train()
    
    # 11. 保存模型
    print("Training Done!")
    if trainer.is_world_process_zero():
        os.makedirs(training_args.output_dir, exist_ok=True)
        trainer.save_model(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"Model saved to {training_args.output_dir}")

if __name__ == "__main__":
    main()