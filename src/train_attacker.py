import torch
from dataclasses import dataclass, field 

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "基座模型的HuggingFace ID或本地路径"}
    )
    train_file: str = field(
        default="data/attacker_train.jsonl",
        metadata={"help": "训练数据路径"}
    )
    max_seq_length: int = field(
        default=1024,
        metadata={"help": "最大序列长度"}
    )
    use_4bit: bool = field(
        default=True,
        metadata={"help": "是否使用4-bit量化加载 (QLoRA)"}
    )

def main():
    # 1. 解析参数
    parser = HfArgumentParser((ScriptArguments, SFTConfig))
    script_args, training_args = parser.parse_args_into_dataclasses()

    print(f"Loading model: {script_args.model_name_or_path}")
    print(f"Training data: {script_args.train_file}")

    # 2. 配置4bit量化
    bnb_config = None
    if script_args.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

    # 3. 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map='cuda:0',
        trust_remote_code=True,
        attn_implementation="flash_attention_2" if torch.cuda.is_bf16_supported() else "eager"
    )
    
    # 4. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        trust_remote_code=True
    )
    # pad_token处理
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 5. 配置lora
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 准备对模型进行k-bit训练
    if script_args.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # 6. 加载数据集
    dataset = load_dataset("json", data_files=script_args.train_file, split="train")
    print("Sample data:", dataset[0])

    # 7. 配置 SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # 8. 开始训练
    print("Starting training...")
    trainer.train()

    # 9. 保存模型
    print("Training Done!")
    trainer.save_model(script_args.output_dir)
    tokenizer.save_pretrained(script_args.output_dir)
    
if __name__ == "__main__":
    main()