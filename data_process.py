"""
数据处理流水线：
- 读取 ChatHaruhi-54K 原始数据
- 清洗、过滤（长度/空值/角色缺失）、去重
- 支持多轮对话拼接（more_dialogues 字段）
- 生成数据统计报告
- 划分训练/验证集，输出 SFT 格式
"""

import json
import random
import re
from collections import Counter
from pathlib import Path

INPUT_FILE = "data/ChatHaruhi-54K/Haruhi_54K_v1.jsonl"
TRAIN_FILE = "data/sft_train.json"
VAL_FILE = "data/sft_val.json"
STATS_FILE = "data/data_stats.json"

VAL_RATIO = 0.1
SEED = 42
MIN_INPUT_LEN = 2
MIN_OUTPUT_LEN = 5
MAX_OUTPUT_LEN = 512


# ============ 数据清洗 ============

def load_raw_data(file_path: str) -> list:
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^[「『【]|[」』】]$", "", text).strip()
    return text


def is_valid(item: dict) -> bool:
    question = clean_text(item.get("user_question", ""))
    response = clean_text(item.get("agent_response", ""))
    if not item.get("agent_role"):
        return False
    if len(question) < MIN_INPUT_LEN or len(response) < MIN_OUTPUT_LEN:
        return False
    if len(response) > MAX_OUTPUT_LEN:
        return False
    return True


def build_context(item: dict) -> str:
    """将 user_question + more_dialogues 拼接为多轮上下文"""
    parts = [clean_text(item.get("user_question", ""))]
    for turn in item.get("more_dialogues", []):
        if isinstance(turn, str) and turn.strip():
            parts.append(clean_text(turn))
    return "\n".join(p for p in parts if p)


def to_sft(item: dict) -> dict:
    agent_role = item["agent_role"]
    instruction = (
        f'你现在扮演{agent_role}，请完全进入角色，'
        f'用{agent_role}的语气、口头禅和性格特点来回应对方。'
        f'不要跳出角色，不要说“我是AI”之类的话。'
    )
    return {
        "instruction": instruction,
        "input": build_context(item),
        "output": clean_text(item.get("agent_response", "")),
        "_meta": {
            "agent_role": agent_role,
            "user_role": item.get("user_role", ""),
            "source": item.get("question_source", ""),
        },
    }


# ============ 统计 ============

def compute_stats(data: list, raw: int, filtered: int, deduped: int) -> dict:
    role_counter = Counter(d["_meta"]["agent_role"] for d in data)
    in_lens = [len(d["input"]) for d in data]
    out_lens = [len(d["output"]) for d in data]
    return {
        "raw_count": raw,
        "filtered_out": filtered,
        "dedup_removed": deduped,
        "final_count": len(data),
        "role_count": len(role_counter),
        "top_roles": dict(role_counter.most_common(20)),
        "input_len": {"min": min(in_lens), "max": max(in_lens), "avg": round(sum(in_lens) / len(in_lens), 1)},
        "output_len": {"min": min(out_lens), "max": max(out_lens), "avg": round(sum(out_lens) / len(out_lens), 1)},
    }


# ============ 主流程 ============

def main():
    print("=" * 60)
    print("ChatHaruhi 数据处理流水线")
    print("=" * 60)

    # Step 1: 加载
    print(f"\n[1/5] 加载原始数据: {INPUT_FILE}")
    raw = load_raw_data(INPUT_FILE)
    raw_count = len(raw)
    print(f"  原始样本: {raw_count}")

    # Step 2: 过滤
    print("\n[2/5] 过滤无效样本...")
    valid = [item for item in raw if is_valid(item)]
    filtered = raw_count - len(valid)
    print(f"  过滤: {filtered} 条（空/过短/过长/缺角色）")
    print(f"  保留: {len(valid)} 条")

    # Step 3: 去重（按 output 去重）
    print("\n[3/5] 去重...")
    seen = set()
    deduped_list = []
    for item in valid:
        key = clean_text(item.get("agent_response", ""))
        if key not in seen:
            seen.add(key)
            deduped_list.append(item)
    dedup_removed = len(valid) - len(deduped_list)
    print(f"  去重移除: {dedup_removed} 条")
    print(f"  剩余:     {len(deduped_list)} 条")

    # Step 4: 转换为 SFT 格式
    print("\n[4/5] 转换为 SFT 格式（含多轮上下文）...")
    sft_data = [to_sft(item) for item in deduped_list]

    role_counter = Counter(d["_meta"]["agent_role"] for d in sft_data)
    print(f"  角色总数: {len(role_counter)}")
    print("  Top-10 角色:")
    for role, cnt in role_counter.most_common(10):
        print(f"    {role}: {cnt} 条")

    # Step 5: 划分并保存
    print("\n[5/5] 划分并保存...")
    random.seed(SEED)
    random.shuffle(sft_data)
    val_size = int(len(sft_data) * VAL_RATIO)
    val_data = sft_data[:val_size]
    train_data = sft_data[val_size:]
    print(f"  训练集: {len(train_data)} 条")
    print(f"  验证集: {len(val_data)} 条")

    def strip_meta(d):
        return {"instruction": d["instruction"], "input": d["input"], "output": d["output"]}

    Path(TRAIN_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump([strip_meta(d) for d in train_data], f, ensure_ascii=False, indent=2)

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump([strip_meta(d) for d in val_data], f, ensure_ascii=False, indent=2)

    stats = compute_stats(sft_data, raw_count, filtered, dedup_removed)
    with open(STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 训练集 → {TRAIN_FILE}")
    print(f"✓ 验证集 → {VAL_FILE}")
    print(f"✓ 统计   → {STATS_FILE}")

    print("\n--- 数据预览（前 3 条）---")
    for i, item in enumerate(train_data[:3]):
        print(f"\n[样本 {i + 1}]")
        print(f"  instruction: {item['instruction'][:70]}...")
        print(f"  input:  {item['input'][:80]}")
        print(f"  output: {item['output'][:80]}")


if __name__ == "__main__":
    main()
