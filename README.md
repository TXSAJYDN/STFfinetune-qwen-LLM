# 🎭 角色扮演 LLM 微调与评测平台

基于 **Qwen2.5-3B-Instruct + QLoRA** 的角色扮演大模型全流程平台。覆盖数据工程、模型训练、自动评测、RAG 知识增强、安全防御、推理基准与可视化 Web Demo，数据集为 ChatHaruhi-54K（289 个角色、49K+ 对话）。

## 核心特性

| 模块 | 说明 |
|------|------|
| **数据工程** | 清洗 → 过滤 → 去重 → 多轮拼接 → 统计报告 → 训练/验证集划分 |
| **QLoRA 训练** | 4-bit 量化 + LoRA，8GB 显存即可训练 3B 模型；支持断点续训与增量训练 |
| **自动评测** | Base vs LoRA 对比：BLEU-4 / ROUGE-L / 角色一致性（本地评测，无需 API） |
| **多轮对话** | 完整历史管理 + Token 窗口截断，流式生成 |
| **RAG 知识增强** | 30 个角色知识库（背景/口头禅/性格/关系），自动注入 System Prompt |
| **安全防御** | 双层 Prompt 注入检测（关键词 + 正则），拦截越狱请求 |
| **推理基准** | 延迟 / 吞吐 / 显存占用一键测试 |
| **可视化平台** | 6 个 Web 页面：角色聊天、效果对比、数据分析、训练可视化、评测报告、推理基准 |

## 评测结果（最终版）

| 指标 | Base 模型 | LoRA 微调 | 提升 |
|------|----------|----------|------|
| **BLEU-4** | 0.1807 | **0.4064** | +0.2257 |
| **ROUGE-L** | 0.1124 | **0.2312** | +0.1188 |
| **角色一致性** | 1.0000 | **1.0000** | — |
| 平均输出长度 | 92.0 | 99.4 | +7.4 |
| 平均延迟 | 19.6s | **12.0s** | -7.6s |

> LoRA 微调后 BLEU-4 提升 **+125%**，ROUGE-L 提升 **+106%**，推理速度反而更快。

## 项目结构

```
finetune-qwen/
├── app/                           # Web 应用层
│   ├── api.py                    # FastAPI 后端（推理 + 数据 API）
│   ├── web.py                    # 页面路由
│   ├── templates/                # 前端页面
│   │   ├── base.html             #   导航基础模板
│   │   ├── chat.html             #   多轮角色对话（流式）
│   │   ├── compare.html          #   Base vs LoRA 效果对比
│   │   ├── data.html             #   数据集统计可视化
│   │   ├── training.html         #   训练 Loss/LR 曲线
│   │   ├── eval.html             #   评测报告（雷达图 + 表格）
│   │   └── benchmark.html        #   推理性能基准
│   └── static/                   # CSS / JS 静态资源
├── rag/                           # RAG 知识增强
│   ├── knowledge/characters.json #   30 个角色知识库
│   └── retriever.py              #   知识检索模块
├── safety/                        # 安全防御
│   └── detector.py               #   Prompt 注入检测
├── benchmark/                     # 推理基准
│   └── bench_inference.py        #   延迟 / 吞吐 / 显存测试脚本
├── configs/                       # 训练配置
│   ├── qlora_sft.yaml            #   QLoRA 配置（推荐，>=8GB）
│   └── lora_sft.yaml             #   LoRA 配置（>=24GB）
├── data/                          # 数据目录
│   ├── ChatHaruhi-54K/           #   原始数据集
│   ├── sft_train.json            #   训练集 (46K)
│   ├── sft_val.json              #   验证集 (5K)
│   ├── sft_train_slim.json       #   精简训练集 (10K, Top-20 角色)
│   └── data_stats.json           #   数据统计报告
├── models/                        # 模型权重
│   └── Qwen2.5-3B-Instruct/
├── outputs/                       # 训练产物
│   ├── qlora_sft/final/          #   LoRA 适配器权重
│   ├── eval_report.json          #   评测报告
│   └── bench_results.json        #   基准测试结果
├── data_process.py               # 数据处理流水线
├── data_slim.py                  # 数据精简工具（Top-N 角色采样）
├── train_sft.py                  # SFT 训练（支持断点续训 + 增量训练）
├── eval.py                       # 自动评测（BLEU / ROUGE / 角色一致性）
├── inference.py                  # CLI 交互式推理
├── merge_lora.py                 # LoRA 权重合并
├── run_web.py                    # Web 平台启动入口
└── requirements.txt
```

## 环境要求

- **GPU**: NVIDIA RTX 4060 8GB（或同级）
- **Python**: 3.10+
- **PyTorch**: 2.0+
- **核心依赖**: transformers, peft, trl, datasets, accelerate, bitsandbytes, fastapi, nltk

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据处理

```bash
# 完整数据处理流水线（清洗 → 去重 → 统计 → 划分）
python data_process.py

# 可选：生成精简数据集（Top-20 角色，加速训练）
python data_slim.py
```

### 3. 模型训练

```bash
# QLoRA 训练（推荐，8GB 显存）
python train_sft.py --config configs/qlora_sft.yaml

# 在已有 LoRA 基础上继续训练（增量训练）
python train_sft.py --config configs/qlora_sft.yaml --resume_lora outputs/qlora_sft/final

# 训练中断后恢复（从 checkpoint 继续）
python train_sft.py --config configs/qlora_sft.yaml
```

### 4. 评测

```bash
# Base vs LoRA 自动对比评测
python eval.py --load_in_4bit --num_samples 50
```

### 5. 推理基准

```bash
python -m benchmark.bench_inference --load_in_4bit
```

### 6. 启动 Web 平台

```bash
python run_web.py
# 访问 http://localhost:8000
```

Web 平台包含 6 个页面：

| 页面 | 路径 | 功能 |
|------|------|------|
| 首页 | `/` | 模型加载、系统状态 |
| 角色聊天 | `/chat` | 多轮流式对话、RAG 知识增强、安全拦截 |
| 效果对比 | `/compare` | Base vs LoRA 实时对比 |
| 数据分析 | `/data` | 数据集统计、角色分布、长度分布图表 |
| 训练可视化 | `/training` | Loss / Learning Rate / Eval Loss 曲线 |
| 评测报告 | `/eval` | 雷达图、指标表格、样本输出对比 |
| 推理基准 | `/benchmark` | 延迟分布、吞吐量、显存占用 |

### 7. 其他工具

```bash
# CLI 交互式推理
python inference.py --base_model ./models/Qwen2.5-3B-Instruct \
                    --lora_path outputs/qlora_sft/final --interactive --load_in_4bit

# 合并 LoRA 权重为完整模型
python merge_lora.py --base_model ./models/Qwen2.5-3B-Instruct \
                     --lora_path outputs/qlora_sft/final --output_dir outputs/merged_model
```

## 训练配置

### QLoRA 参数

| 参数 | 值 | 说明 |
|------|------|------|
| `r` | 32 | LoRA 秩 |
| `lora_alpha` | 64 | 缩放因子（通常为 r 的 2 倍） |
| `lora_dropout` | 0.05 | Dropout |
| `target_modules` | q/k/v/o/gate/up/down_proj | 应用 LoRA 的模块（全覆盖） |
| `load_in_4bit` | true | NF4 量化 + 双重量化 |

### 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| `learning_rate` | 1e-4 | 学习率 |
| `num_train_epochs` | 2 | 训练轮数 |
| `batch_size` | 1 | 单卡批次 |
| `gradient_accumulation` | 16 | 梯度累积（有效批次=16） |
| `max_seq_length` | 512 | 最大序列长度 |
| `lr_scheduler` | cosine | 学习率调度 |

### 8GB 显存优化建议

1. 使用 QLoRA（4-bit 量化）
2. `max_seq_length` ≤ 1024
3. `per_device_train_batch_size` = 1
4. 开启 `gradient_checkpointing`
5. 使用精简数据集（`data_slim.py`）

## 模型调参与优化过程

本项目经历了完整的**训练调参 → 推理调参 → 评测验证**迭代过程，以下是关键调参记录：

### 训练阶段调参

#### 第一次训练：基础 QLoRA（1 epoch）

| 参数 | 值 | 结果 |
|------|------|------|
| `max_seq_length` | 1024 | ✓ 正常训练 |
| `batch_size` | 1 | ✓ 8GB 显存刚好 |
| `gradient_accumulation` | 16 | 有效批次=16 |
| `learning_rate` | 2e-4 | 初始学习率 |
| `num_train_epochs` | 1 | 训练 3078 步 |

训练时间：约 17 小时（~20s/步）

#### OOM 问题与解决

尝试将 `batch_size` 增大到 2 或 `max_seq_length` 增大到 2048 时触发 **CUDA OOM**：

```
torch.OutOfMemoryError: Tried to allocate 980.00 MiB. GPU has 8.00 GiB total, 56.44 MiB free.
```

**解决方案**：保持 `batch_size=1`，通过 `gradient_accumulation_steps=16` 模拟大批次。

#### 增量训练：在已有 LoRA 基础上继续训练（+2 epoch）

| 参数 | 调整 | 原因 |
|------|------|------|
| `learning_rate` | 2e-4 → **1e-4** | 继续训练用更小学习率，避免破坏已学特征 |
| `max_seq_length` | 1024 → **512** | 数据平均长度 83 字，512 够用，节省显存 |
| `num_train_epochs` | 1 → **2** | 在已有 1 epoch 基础上补训 |
| 数据集 | 全量 49K → **精简 10K** | 只保留 Top-20 高频角色，强化核心能力 |

训练时间：约 5 小时（精简数据 + 短序列大幅加速）

### 推理阶段调参

推理生成参数对输出质量影响巨大，经过三轮实验对比：

| 参数 | V1（初始） | V2（过度） | V3（最终） | 调整原因 |
|------|-----------|-----------|-----------|---------|
| `temperature` | 0.7 | 0.6 | **0.6** | 降低随机性，输出更稳定 |
| `top_p` | 0.9 | 0.85 | **0.85** | 缩小采样范围，减少离题 |
| `repetition_penalty` | 1.1 | 1.3 | **1.2** | 1.3 过高导致乱码，1.2 折中 |
| `max_new_tokens` | 256 | 512 | **256** | 512 导致输出过长，稀释评测分数 |

#### 各版本评测对比

| 版本 | BLEU-4 | ROUGE-L | 平均输出长度 | 问题 |
|------|--------|---------|-------------|------|
| V1（初始） | 0.4343 | 0.2579 | 87.5 | 偶有重复文本 |
| V2（过度调参） | 0.3357 | 0.1909 | **204.6** | 输出过长，rep_penalty 过高产生乱码 |
| **V3（最终）** | **0.4064** | **0.2312** | 99.4 | 质量最优，无乱码，延迟最低 |

#### 关键发现

1. **`max_new_tokens` 对评测分数影响最大**：512 让模型生成过多冗余内容（输出 204 字 vs 参考 87 字），BLEU/ROUGE 大幅下降
2. **`repetition_penalty` 不宜过高**：1.3 导致模型强行避免重复词而产生无意义文本（如乱码人名）
3. **分数 vs 质量不完全一致**：V3 的 BLEU 略低于 V1，但人工观察生成质量更好（更贴合角色、无重复）——这是 n-gram 指标的固有局限性
4. **LoRA 推理反而比 Base 更快**：LoRA 模型学会了更简洁直接的回答方式，生成较短序列

### Prompt 工程优化

System Prompt 从简单一句话升级为结构化指令：

**优化前**：
```
你现在扮演韦小宝，请完全进入角色，用韦小宝的语气、口头禅和性格特点来回应对方。
不要跳出角色，不要说"我是AI"之类的话。
```

**优化后**：
```
你现在扮演韦小宝，请完全进入角色。要求：
1. 始终使用韦小宝的语气、口头禅和说话习惯
2. 回复应自然简洁，像真实对话，避免冗长解释
3. 保持角色性格一致，不要跳出角色
4. 不要说"我是AI"或任何破坏沉浸感的话
5. 根据对话情境做出符合角色身份的反应

【角色知识】
（RAG 检索的角色背景、口头禅、性格、关系等）
请严格参考以上信息来塑造角色。
```

结构化 Prompt + RAG 知识注入让角色扮演的稳定性和准确性明显提升。

## 技术架构

```
用户输入
  ↓
[安全检测] → 拦截越狱/注入 → 返回安全提示
  ↓ (通过)
[RAG 检索] → 角色知识库 → 注入 System Prompt
  ↓
[Token 截断] → 滑动窗口管理上下文
  ↓
[模型推理] → Qwen2.5-3B + LoRA → 流式输出
  ↓
前端渲染（Markdown + 代码高亮）
```

## 数据格式

### SFT 训练数据

```json
[
  {
    "instruction": "你现在扮演韦小宝，请完全进入角色...",
    "input": "你今天怎么了？",
    "output": "嘿嘿，我韦小宝能有什么事..."
  }
]
```

### RAG 知识库

```json
{
  "韦小宝": {
    "background": "金庸《鹿鼎记》主角...",
    "catchphrases": ["辣块妈妈", "他奶奶的"],
    "personality": "贪财好色、讲义气、胆大心细",
    "relationships": "康熙（好友）、陈近南（师父）..."
  }
}
```

## 支持的 Prompt 模板

- **qwen**: Qwen 系列 ChatML 格式（默认）
- **chatml**: 通用 ChatML 格式
- **alpaca**: Alpaca 指令格式
