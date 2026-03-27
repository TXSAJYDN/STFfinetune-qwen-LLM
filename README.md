# LLM Fine-Tuning (LoRA / QLoRA)

A fine-tuning project for Qwen-series large language models, supporting **LoRA** and **QLoRA** modes for **SFT (Supervised Fine-Tuning)**.

## Project Structure

```
newproject/
├── configs/                    # Training config files
│   ├── lora_sft.yaml          # LoRA SFT config (1.5B model, higher VRAM usage)
│   └── qlora_sft.yaml         # QLoRA SFT config (3B model, recommended ★)
├── data/                       # Data directory
│   ├── sft_train.json         # SFT training data
│   └── sft_val.json           # SFT validation data
├── train_sft.py               # SFT training script
├── merge_lora.py              # LoRA weight merging script
├── inference.py               # Inference script
├── requirements.txt           # Python dependencies
└── README.md
```

## Environment

- **GPU**: RTX 4060 (8GB VRAM)
- **Conda env**: `pytorch`
- **PyTorch**: 2.10.0+cu128
- **Core deps**: transformers, peft, datasets, accelerate, bitsandbytes, trl

> **Recommended for 8GB VRAM**: QLoRA + Qwen2.5-3B-Instruct, max_seq_length=1024, batch_size=1

## Quick Start

### 1. SFT Instruction Fine-Tuning

Prepare data (JSON format):
```json
[
  {
    "instruction": "Your instruction",
    "input": "Optional input context",
    "output": "Expected output"
  }
]
```

Start training:
```bash
# LoRA mode (requires >= 24GB VRAM)
python train_sft.py --config configs/lora_sft.yaml

# QLoRA mode (requires >= 8GB VRAM, recommended)
python train_sft.py --config configs/qlora_sft.yaml
```

### 2. Merge LoRA Weights

After training, merge the LoRA adapter into the base model:
```bash
python merge_lora.py \
    --base_model ./models/Qwen2.5-3B-Instruct \
    --lora_path outputs/qlora_sft/final \
    --output_dir outputs/merged_model
```

### 3. Inference

```bash
# With LoRA adapter (no merging needed)
python inference.py --base_model ./models/Qwen2.5-3B-Instruct --lora_path outputs/qlora_sft/final --load_in_4bit

# With merged model
python inference.py --model_path outputs/merged_model

# Interactive chat
python inference.py --model_path outputs/merged_model --interactive

# Single prompt
python inference.py --model_path outputs/merged_model --prompt "Hello"

# 4-bit quantized inference (saves VRAM)
python inference.py --base_model ./models/Qwen2.5-3B-Instruct --lora_path outputs/qlora_sft/final --load_in_4bit
```

## Configuration

### LoRA Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `r` | 32 | LoRA rank; higher = more capacity but more VRAM |
| `lora_alpha` | 64 | Scaling factor, typically 2x of `r` |
| `lora_dropout` | 0.05 | Dropout rate |
| `target_modules` | q/k/v/o/gate/up/down_proj | Modules to apply LoRA to |

### Training Parameters

| Parameter | Description |
|-----------|-------------|
| `per_device_train_batch_size` | Batch size per GPU; reduce if OOM |
| `gradient_accumulation_steps` | Gradient accumulation; increase to simulate larger batch |
| `learning_rate` | Learning rate; typically 1e-4 ~ 3e-4 for LoRA |
| `num_train_epochs` | Number of training epochs |
| `max_seq_length` | Maximum sequence length |

### Tips for Limited VRAM (8GB)

Configs are already optimized for RTX 4060 8GB. If you still get OOM:
1. Use a smaller model: `Qwen2.5-1.5B-Instruct`
2. Reduce `max_seq_length` to 512
3. Reduce LoRA `r` to 16
4. Ensure `gradient_checkpointing: true`

## Data Format

### SFT Data (JSON)

```json
[
  {
    "instruction": "Instruction or question",
    "input": "Optional input context (empty string if none)",
    "output": "Expected response"
  }
]
```

## Supported Prompt Templates

- **qwen**: Qwen-series ChatML format (default)
- **chatml**: Generic ChatML format
- **alpaca**: Alpaca instruction format
