"""
推理脚本
支持三种加载方式:
  1. 加载 LoRA 适配器 (未合并)
  2. 加载合并后的完整模型
  3. 直接加载原始模型 (用于对比)

用法:
    # 使用 LoRA 适配器推理
    python inference.py --base_model Qwen/Qwen2.5-7B-Instruct --lora_path outputs/lora_sft/final

    # 使用合并后的模型推理
    python inference.py --model_path outputs/merged_model

    # 交互式对话模式
    python inference.py --model_path outputs/merged_model --interactive
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(
    model_path: str = None,
    base_model: str = None,
    lora_path: str = None,
    load_in_4bit: bool = False,
):
    """加载模型和分词器"""

    if lora_path and base_model:
        # 模式1: 加载基座模型 + LoRA 适配器
        print(f"加载基座模型: {base_model}")
        print(f"加载 LoRA 适配器: {lora_path}")

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, lora_path)
        model.eval()

    elif model_path:
        # 模式2: 加载合并后的完整模型
        print(f"加载模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

    else:
        raise ValueError("请提供 --model_path 或 --base_model + --lora_path")

    print("✓ 模型加载完成")
    return model, tokenizer


def chat(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """单轮对话推理"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    # 只取新生成的部分
    response_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response


def interactive_mode(model, tokenizer, max_new_tokens: int = 512):
    """交互式对话模式"""
    print("\n" + "=" * 60)
    print("交互式对话模式 (输入 'quit' 或 'exit' 退出)")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n用户> ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("quit", "exit", "q"):
                print("再见!")
                break

            response = chat(model, tokenizer, prompt, max_new_tokens)
            print(f"\n助手> {response}")

        except KeyboardInterrupt:
            print("\n再见!")
            break


def main():
    parser = argparse.ArgumentParser(description="LLM 推理")
    parser.add_argument("--model_path", type=str, default=None, help="合并后的模型路径")
    parser.add_argument("--base_model", type=str, default=None, help="基座模型路径")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA 适配器路径")
    parser.add_argument("--load_in_4bit", action="store_true", help="使用 4-bit 量化加载")
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    parser.add_argument("--prompt", type=str, default=None, help="单次推理的提示语")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="最大生成 token 数")
    args = parser.parse_args()

    model, tokenizer = load_model(
        model_path=args.model_path,
        base_model=args.base_model,
        lora_path=args.lora_path,
        load_in_4bit=args.load_in_4bit,
    )

    if args.interactive:
        interactive_mode(model, tokenizer, args.max_new_tokens)
    elif args.prompt:
        response = chat(model, tokenizer, args.prompt, args.max_new_tokens)
        print(f"\n回复:\n{response}")
    else:
        # 默认测试
        test_prompts = [
            "请介绍一下什么是大语言模型？",
            "用 Python 写一个冒泡排序算法。",
        ]
        for prompt in test_prompts:
            print(f"\n{'='*60}")
            print(f"提问: {prompt}")
            response = chat(model, tokenizer, prompt, args.max_new_tokens)
            print(f"回复: {response}")


if __name__ == "__main__":
    main()
