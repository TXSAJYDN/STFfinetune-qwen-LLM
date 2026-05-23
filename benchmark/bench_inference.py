"""
推理性能基准测试脚本

测试不同配置下的推理延迟、吞吐量和显存占用。
结果保存至 outputs/bench_results.json

用法:
  python -m benchmark.bench_inference --base_model ./models/Qwen2.5-3B-Instruct --load_in_4bit
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

TEST_PROMPTS = [
    "你好，请介绍一下你自己。",
    "请用韦小宝的语气讲一个笑话。",
    "什么是量子计算？请简单解释。",
    "请写一首关于春天的诗。",
    "给我讲讲三国演义中诸葛亮的故事。",
]


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
    return 0


def run_benchmark(model, tokenizer, prompts, max_new_tokens=128, num_rounds=2):
    results = []
    total_tokens = 0
    total_time = 0

    for _ in range(num_rounds):
        for prompt in prompts:
            messages = [
                {"role": "system", "content": "你是一个角色扮演助手。"},
                {"role": "user", "content": prompt},
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            input_len = inputs["input_ids"].shape[1]

            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    do_sample=False, temperature=1.0,
                )
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.perf_counter() - t0

            gen_tokens = out.shape[1] - input_len
            total_tokens += gen_tokens
            total_time += elapsed

            results.append({
                "prompt": prompt[:50],
                "input_tokens": input_len,
                "gen_tokens": gen_tokens,
                "latency_s": round(elapsed, 3),
                "tokens_per_s": round(gen_tokens / elapsed, 1) if elapsed > 0 else 0,
            })

    return {
        "samples": results,
        "summary": {
            "total_prompts": len(results),
            "avg_latency_s": round(total_time / len(results), 3),
            "avg_tokens_per_s": round(total_tokens / total_time, 1) if total_time > 0 else 0,
            "total_gen_tokens": total_tokens,
            "peak_gpu_memory_mb": get_gpu_memory_mb(),
            "max_new_tokens": max_new_tokens,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="推理性能基准测试")
    parser.add_argument("--base_model", type=str, default="./models/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--output", type=str, default="outputs/bench_results.json")
    args = parser.parse_args()

    print("=" * 60)
    print("推理性能基准测试")
    print("=" * 60)

    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )

    config_label = "4-bit" if args.load_in_4bit else "bf16"
    if args.lora_path and Path(args.lora_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.lora_path)
        config_label += "+LoRA"

    model.eval()
    print(f"模型: {args.base_model} ({config_label})")
    print(f"设备: {next(model.parameters()).device}")
    print(f"测试轮次: {args.num_rounds}, max_new_tokens: {args.max_new_tokens}")

    # warmup
    print("\nWarmup...")
    _ = run_benchmark(model, tokenizer, TEST_PROMPTS[:1], args.max_new_tokens, 1)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print("Running benchmark...")
    result = run_benchmark(model, tokenizer, TEST_PROMPTS, args.max_new_tokens, args.num_rounds)
    result["config"] = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "quantization": config_label,
        "device": str(next(model.parameters()).device),
    }

    s = result["summary"]
    print(f"\n{'='*60}")
    print(f"  平均延迟:   {s['avg_latency_s']}s")
    print(f"  吞吐量:     {s['avg_tokens_per_s']} tokens/s")
    print(f"  峰值显存:   {s['peak_gpu_memory_mb']} MB")
    print(f"{'='*60}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 结果已保存: {args.output}")


if __name__ == "__main__":
    main()
