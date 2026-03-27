"""
LoRA 权重合并脚本
将训练好的 LoRA 适配器权重合并到基座模型中，导出完整模型

用法:
    python merge_lora.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --lora_path outputs/lora_sft/final \
        --output_dir outputs/merged_model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_weights(base_model_path: str, lora_path: str, output_dir: str):
    """合并 LoRA 权重到基座模型"""

    print("=" * 60)
    print("LoRA 权重合并")
    print(f"  基座模型: {base_model_path}")
    print(f"  LoRA 路径: {lora_path}")
    print(f"  输出目录: {output_dir}")
    print("=" * 60)

    # 加载分词器
    print("\n[1/4] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path, trust_remote_code=True
    )

    # 加载基座模型 (全精度)
    print("[2/4] 加载基座模型 (BFloat16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 加载 LoRA 适配器
    print("[3/4] 加载 LoRA 适配器并合并...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model = model.merge_and_unload()

    # 保存合并后的模型
    print("[4/4] 保存合并后的完整模型...")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    print(f"\n✓ 合并完成! 模型已保存至: {output_dir}")
    print("  现在可以像使用普通模型一样加载和使用它。")


def main():
    parser = argparse.ArgumentParser(description="合并 LoRA 权重到基座模型")
    parser.add_argument("--base_model", type=str, required=True, help="基座模型路径或 HuggingFace ID")
    parser.add_argument("--lora_path", type=str, required=True, help="LoRA 适配器路径")
    parser.add_argument("--output_dir", type=str, required=True, help="合并后模型的输出路径")
    args = parser.parse_args()

    merge_lora_weights(args.base_model, args.lora_path, args.output_dir)


if __name__ == "__main__":
    main()
