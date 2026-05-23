"""
评测脚本：比较 Base 与 LoRA 微调模型在角色扮演任务上的表现

指标：
  - BLEU-4     字符级 n-gram 精度
  - ROUGE-L    字符级最长公共子序列 F1
  - 角色一致性  是否保持角色（未出现"我是AI"类跳出语句）
  - 平均生成长度
  - 平均推理延迟

用法：
  # 仅评测 Base 模型
  python eval.py --base_only

  # 评测并对比 Base vs LoRA
  python eval.py --base_model ./models/Qwen2.5-3B-Instruct \\
                 --lora_path outputs/qlora_sft/final \\
                 --num_samples 100
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


# ============ 评测指标 ============

def compute_bleu4(reference: str, hypothesis: str) -> float:
    """字符级 BLEU-4，fallback 到 unigram overlap"""
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
        sf = SmoothingFunction().method1
        return sentence_bleu(
            [list(reference)], list(hypothesis),
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=sf,
        )
    except ImportError:
        if not hypothesis:
            return 0.0
        inter = len(set(reference) & set(hypothesis))
        return inter / len(set(hypothesis))


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    """字符级 ROUGE-L（LCS F1）"""
    if not reference or not hypothesis:
        return 0.0
    m, n = len(reference), len(hypothesis)
    if m * n > 100_000:
        inter = len(set(reference) & set(hypothesis))
        p = inter / n
        r = inter / m
        return 2 * p * r / (p + r) if (p + r) else 0.0
    prev = [0] * (n + 1)
    for c in reference:
        curr = [0] * (n + 1)
        for j, d in enumerate(hypothesis, 1):
            curr[j] = prev[j - 1] + 1 if c == d else max(curr[j - 1], prev[j])
        prev = curr
    lcs = prev[n]
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_role_consistency(response: str) -> float:
    """角色一致性：检测是否跳出角色"""
    break_phrases = ["我是ai", "我是人工智能", "我是语言模型", "我无法扮演", "作为ai助手", "作为一个ai"]
    resp_lower = response.lower().replace(" ", "")
    for phrase in break_phrases:
        if phrase in resp_lower:
            return 0.0
    return 1.0 if response.strip() else 0.0


def aggregate(results: list) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "bleu4":            round(sum(r["bleu4"] for r in results) / n, 4),
        "rouge_l":          round(sum(r["rouge_l"] for r in results) / n, 4),
        "role_consistency": round(sum(r["role_consistency"] for r in results) / n, 4),
        "avg_pred_len":     round(sum(r["pred_len"] for r in results) / n, 1),
        "avg_ref_len":      round(sum(r["ref_len"] for r in results) / n, 1),
        "avg_latency_s":    round(sum(r["latency_s"] for r in results) / n, 3),
        "sample_count":     n,
    }


# ============ 模型加载 ============

def load_model(base_model: str, lora_path: str = None, load_in_4bit: bool = False):
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
    if lora_path and Path(lora_path).exists():
        print(f"  加载 LoRA 适配器: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, instruction: str, user_input: str, max_new_tokens: int) -> tuple:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    t0 = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.85,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0
    resp_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    return response, round(latency, 3)


# ============ 评测主循环 ============

def run_eval(model, tokenizer, samples: list, label: str, max_new_tokens: int) -> list:
    results = []
    for i, sample in enumerate(samples):
        print(f"\r  [{label}] {i + 1}/{len(samples)}", end="", flush=True)
        response, latency = generate(model, tokenizer, sample["instruction"], sample["input"], max_new_tokens)
        results.append({
            "input":            sample["input"],
            "reference":        sample["output"],
            "prediction":       response,
            "bleu4":            round(compute_bleu4(sample["output"], response), 4),
            "rouge_l":          round(compute_rouge_l(sample["output"], response), 4),
            "role_consistency": compute_role_consistency(response),
            "pred_len":         len(response),
            "ref_len":          len(sample["output"]),
            "latency_s":        latency,
        })
    print()
    return results


# ============ 输出格式 ============

def print_metrics_table(base_m: dict, lora_m: dict = None):
    keys = ["bleu4", "rouge_l", "role_consistency", "avg_pred_len", "avg_latency_s"]
    print("\n" + "=" * 70)
    if lora_m:
        print(f"  {'指标':<22} {'Base 模型':>14} {'LoRA 微调':>14} {'提升':>12}")
        print("=" * 70)
        for k in keys:
            b = base_m.get(k, 0)
            l = lora_m.get(k, 0)
            delta = l - b
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            print(f"  {k:<22} {b:>14.4f} {l:>14.4f} {delta_str:>12}")
    else:
        print(f"  {'指标':<22} {'Base 模型':>14}")
        print("=" * 70)
        for k in keys:
            b = base_m.get(k, 0)
            print(f"  {k:<22} {b:>14.4f}")
    print("=" * 70)


def print_samples(base_res: list, lora_res: list = None, n: int = 3):
    print(f"\n{'─'*70}")
    print(f"样本输出对比（前 {n} 条）")
    for i in range(min(n, len(base_res))):
        b = base_res[i]
        print(f"\n[样本 {i + 1}]")
        print(f"  输入:       {b['input'][:80]}")
        print(f"  参考:       {b['reference'][:80]}")
        print(f"  Base 回复:  {b['prediction'][:80]}")
        if lora_res and i < len(lora_res):
            print(f"  LoRA 回复:  {lora_res[i]['prediction'][:80]}")


# ============ 入口 ============

def main():
    parser = argparse.ArgumentParser(description="角色扮演 LLM 评测")
    parser.add_argument("--base_model",    type=str, default="./models/Qwen2.5-3B-Instruct")
    parser.add_argument("--lora_path",     type=str, default="outputs/qlora_sft/final")
    parser.add_argument("--val_file",      type=str, default="data/sft_val.json")
    parser.add_argument("--num_samples",   type=int, default=100)
    parser.add_argument("--max_new_tokens",type=int, default=256)
    parser.add_argument("--load_in_4bit",  action="store_true")
    parser.add_argument("--output",        type=str, default="outputs/eval_report.json")
    parser.add_argument("--seed",          type=int, default=42)
    parser.add_argument("--base_only",     action="store_true", help="仅评测 Base 模型，不加载 LoRA")
    args = parser.parse_args()

    print("=" * 70)
    print("角色扮演 LLM 评测脚本")
    print("=" * 70)

    with open(args.val_file, "r", encoding="utf-8") as f:
        val_data = json.load(f)
    random.seed(args.seed)
    samples = random.sample(val_data, min(args.num_samples, len(val_data)))
    print(f"\n评测样本数: {len(samples)} / {len(val_data)}")

    report = {"args": vars(args), "sample_count": len(samples)}

    # ---- Base 模型评测 ----
    print(f"\n[1] 加载 Base 模型: {args.base_model}")
    base_model, tokenizer = load_model(args.base_model, load_in_4bit=args.load_in_4bit)
    print("  推理中...")
    base_results = run_eval(base_model, tokenizer, samples, "Base", args.max_new_tokens)
    base_metrics = aggregate(base_results)
    report["base"] = {"metrics": base_metrics, "results": base_results}

    # ---- LoRA 模型评测 ----
    lora_results, lora_metrics = None, None
    if not args.base_only:
        if Path(args.lora_path).exists():
            print(f"\n[2] 加载 LoRA 模型: {args.lora_path}")
            del base_model
            torch.cuda.empty_cache()
            lora_model, tokenizer = load_model(
                args.base_model, lora_path=args.lora_path, load_in_4bit=args.load_in_4bit
            )
            print("  推理中...")
            lora_results = run_eval(lora_model, tokenizer, samples, "LoRA", args.max_new_tokens)
            lora_metrics = aggregate(lora_results)
            report["lora"] = {"metrics": lora_metrics, "results": lora_results}
        else:
            print(f"\n[!] LoRA 路径不存在: {args.lora_path}，跳过 LoRA 评测")

    # ---- 输出结果 ----
    print_metrics_table(base_metrics, lora_metrics)
    print_samples(base_results, lora_results)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ 评测报告已保存: {args.output}")


if __name__ == "__main__":
    main()
