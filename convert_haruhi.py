"""
将 ChatHaruhi-54K 数据集转换为项目 SFT 训练格式
输出: data/sft_train.json + data/sft_val.json
"""
import json
import random

INPUT_FILE = "data/ChatHaruhi-54K/Haruhi_54K_v1.jsonl"
TRAIN_FILE = "data/sft_train.json"
VAL_FILE = "data/sft_val.json"
VAL_RATIO = 0.1  # 10% 作为验证集
SEED = 42

def convert():
    # 读取 JSONL
    raw_data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_data.append(json.loads(line))

    print(f"原始数据: {len(raw_data)} 条")

    # 统计角色分布
    role_count = {}
    for item in raw_data:
        role = item.get("agent_role", "未知")
        role_count[role] = role_count.get(role, 0) + 1

    print(f"角色数: {len(role_count)}")
    for role, count in sorted(role_count.items(), key=lambda x: -x[1])[:10]:
        print(f"  {role}: {count} 条")

    # 转换为 SFT 格式
    sft_data = []
    for item in raw_data:
        agent_role = item.get("agent_role", "")
        user_question = item.get("user_question", "")
        agent_response = item.get("agent_response", "")

        if not user_question or not agent_response:
            continue

        # 构建 instruction: 告诉模型扮演什么角色
        instruction = f"请你扮演{agent_role}，用{agent_role}的语气和风格来回答问题。"

        sft_item = {
            "instruction": instruction,
            "input": user_question,
            "output": agent_response,
        }
        sft_data.append(sft_item)

    print(f"\n转换后数据: {len(sft_data)} 条")

    # 随机打乱并划分
    random.seed(SEED)
    random.shuffle(sft_data)

    val_size = int(len(sft_data) * VAL_RATIO)
    val_data = sft_data[:val_size]
    train_data = sft_data[val_size:]

    print(f"训练集: {len(train_data)} 条")
    print(f"验证集: {len(val_data)} 条")

    # 保存
    with open(TRAIN_FILE, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(VAL_FILE, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 训练集已保存: {TRAIN_FILE}")
    print(f"✓ 验证集已保存: {VAL_FILE}")

    # 预览几条
    print("\n--- 训练数据预览 ---")
    for i in range(min(3, len(train_data))):
        item = train_data[i]
        print(f"\n[样本 {i+1}]")
        print(f"  instruction: {item['instruction']}")
        print(f"  input: {item['input'][:100]}")
        print(f"  output: {item['output'][:100]}")


if __name__ == "__main__":
    convert()
