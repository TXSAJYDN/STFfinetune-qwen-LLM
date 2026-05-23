"""
精简数据集：只保留 Top-N 高频角色，每角色上限 max_per_role 条
用于在小显存 GPU 上快速训练出有效模型
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# ============ 配置 ============
INPUT_TRAIN = "data/sft_train.json"
INPUT_VAL = "data/sft_val.json"
OUTPUT_TRAIN = "data/sft_train_slim.json"
OUTPUT_VAL = "data/sft_val_slim.json"

TOP_N_ROLES = 20       # 保留前 N 个高频角色
MAX_PER_ROLE = 500     # 每角色最多保留条数
SEED = 42


def extract_role(instruction: str) -> str:
    """从 instruction 中提取角色名"""
    if "扮演" in instruction:
        parts = instruction.split("扮演")
        if len(parts) >= 2:
            role = parts[1].split("，")[0].split(",")[0].strip()
            return role
    return ""


def slim_dataset(data: list, top_n: int, max_per_role: int) -> list:
    # 统计角色频率
    role_data = defaultdict(list)
    for item in data:
        role = extract_role(item["instruction"])
        if role:
            role_data[role].append(item)

    # 取 top-N 角色
    role_counts = {r: len(v) for r, v in role_data.items()}
    top_roles = sorted(role_counts, key=role_counts.get, reverse=True)[:top_n]

    print(f"  保留角色 ({len(top_roles)}):")
    result = []
    random.seed(SEED)
    for role in top_roles:
        items = role_data[role]
        if len(items) > max_per_role:
            items = random.sample(items, max_per_role)
        result.extend(items)
        print(f"    {role}: {len(items)} 条")

    random.shuffle(result)
    return result


def main():
    print("=" * 60)
    print("精简数据集生成")
    print(f"  Top-{TOP_N_ROLES} 角色, 每角色上限 {MAX_PER_ROLE} 条")
    print("=" * 60)

    with open(INPUT_TRAIN, "r", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(INPUT_VAL, "r", encoding="utf-8") as f:
        val_data = json.load(f)

    print(f"\n原始: 训练 {len(train_data)} 条, 验证 {len(val_data)} 条")

    print("\n[训练集精简]")
    slim_train = slim_dataset(train_data, TOP_N_ROLES, MAX_PER_ROLE)
    print(f"\n[验证集精简]")
    slim_val = slim_dataset(val_data, TOP_N_ROLES, MAX_PER_ROLE // 5)

    print(f"\n精简后: 训练 {len(slim_train)} 条, 验证 {len(slim_val)} 条")
    print(f"压缩比: {len(slim_train)/len(train_data)*100:.1f}%")

    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        json.dump(slim_train, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_VAL, "w", encoding="utf-8") as f:
        json.dump(slim_val, f, ensure_ascii=False, indent=2)

    print(f"\n✓ {OUTPUT_TRAIN}")
    print(f"✓ {OUTPUT_VAL}")


if __name__ == "__main__":
    main()
