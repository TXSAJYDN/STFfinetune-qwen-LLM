"""
SFT (Supervised Fine-Tuning) 指令微调训练脚本
支持 LoRA 和 QLoRA 两种模式，通过配置文件切换

用法:
    python train_sft.py --config configs/lora_sft.yaml
    python train_sft.py --config configs/qlora_sft.yaml
"""

import os
import json
import argparse
import yaml
import torch
from typing import Dict, List, Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


# ======================== 数据处理 ========================

def load_sft_data(file_path: str) -> List[Dict]:
    """加载 SFT 数据集 (JSON 格式)"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓ 加载数据: {file_path}, 共 {len(data)} 条")
    return data


def format_example_qwen(example: Dict) -> str:
    """Qwen ChatML 格式化"""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_example_chatml(example: Dict) -> str:
    """通用 ChatML 格式化"""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        user_content = f"{instruction}\n{input_text}"
    else:
        user_content = instruction

    return (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


def format_example_alpaca(example: Dict) -> str:
    """Alpaca 格式化"""
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]

    if input_text:
        return (
            f"Below is an instruction that describes a task, paired with an input that provides further context. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{output}"
        )
    else:
        return (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )


FORMAT_FUNCTIONS = {
    "qwen": format_example_qwen,
    "chatml": format_example_chatml,
    "alpaca": format_example_alpaca,
}


def build_dataset(data: List[Dict], template: str = "qwen") -> Dataset:
    """构建 HuggingFace Dataset"""
    format_fn = FORMAT_FUNCTIONS.get(template, format_example_qwen)
    formatted = [{"text": format_fn(item)} for item in data]
    dataset = Dataset.from_list(formatted)
    print(f"✓ 数据集构建完成, 使用模板: {template}")
    return dataset


# ======================== 模型加载 ========================

def load_model_and_tokenizer(config: dict):
    """根据配置加载模型和分词器"""
    model_path = config["model_name_or_path"]
    trust_remote = config.get("trust_remote_code", True)

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 量化配置 (QLoRA)
    quant_config = config.get("quantization", {})
    bnb_config = None
    if quant_config.get("enabled", False):
        compute_dtype = getattr(torch, quant_config.get("bnb_4bit_compute_dtype", "bfloat16"))
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=quant_config.get("load_in_4bit", True),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_config.get("bnb_4bit_use_double_quant", True),
            bnb_4bit_quant_type=quant_config.get("bnb_4bit_quant_type", "nf4"),
        )
        print("✓ QLoRA 模式: 4-bit 量化已启用")

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
        trust_remote_code=trust_remote,
    )

    # QLoRA 需要额外准备
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f"✓ 模型加载完成: {model_path}")
    return model, tokenizer


def build_lora_config(config: dict):
    """构建 LoRA 配置 (由 SFTTrainer 负责应用)"""
    lora_cfg = config["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=lora_cfg["task_type"],
        bias=lora_cfg.get("bias", "none"),
    )
    print(f"✓ LoRA 配置就绪: r={lora_cfg['r']}, alpha={lora_cfg['lora_alpha']}")
    return peft_config


# ======================== 训练 ========================

def train(config: dict):
    """主训练流程"""
    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)
    peft_config = build_lora_config(config)

    # 加载数据
    data_cfg = config["data"]
    train_data = load_sft_data(data_cfg["train_file"])
    train_dataset = build_dataset(train_data, data_cfg.get("prompt_template", "qwen"))

    eval_dataset = None
    if data_cfg.get("val_file") and os.path.exists(data_cfg["val_file"]):
        val_data = load_sft_data(data_cfg["val_file"])
        eval_dataset = build_dataset(val_data, data_cfg.get("prompt_template", "qwen"))

    # 训练参数
    train_cfg = config["training"]
    training_args = SFTConfig(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.01),
        warmup_steps=int(train_cfg.get("warmup_ratio", 0.03) * 49248 / train_cfg["gradient_accumulation_steps"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_steps=train_cfg.get("save_steps", 100),
        eval_steps=train_cfg.get("eval_steps", 100),
        eval_strategy=train_cfg.get("eval_strategy", "steps") if eval_dataset else "no",
        save_total_limit=train_cfg.get("save_total_limit", 3),
        fp16=train_cfg.get("fp16", False),
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        report_to=train_cfg.get("report_to", "none"),
        seed=train_cfg.get("seed", 42),
        max_length=data_cfg.get("max_seq_length", 2048),
        dataset_text_field="text",
        packing=False,
    )

    # 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    # 开始训练
    print("\n" + "=" * 60)
    print("开始训练...")
    print(f"  输出目录: {train_cfg['output_dir']}")
    print(f"  训练轮数: {train_cfg['num_train_epochs']}")
    print(f"  有效批大小: {train_cfg['per_device_train_batch_size'] * train_cfg['gradient_accumulation_steps']}")
    print("=" * 60 + "\n")

    trainer.train()

    # 保存最终模型
    final_dir = os.path.join(train_cfg["output_dir"], "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n✓ 训练完成! 模型已保存至: {final_dir}")


# ======================== 入口 ========================

def main():
    parser = argparse.ArgumentParser(description="SFT 指令微调训练")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print("SFT 指令微调训练")
    print(f"配置文件: {args.config}")
    quant = config.get("quantization", {})
    mode = "QLoRA (4-bit)" if quant.get("enabled") else "LoRA (全精度)"
    print(f"训练模式: {mode}")
    print("=" * 60)

    train(config)


if __name__ == "__main__":
    main()
