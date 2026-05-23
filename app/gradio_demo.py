"""
Gradio 演示界面：角色扮演 LLM 微调与评测平台

Tab 1 - 角色扮演对话：与 LoRA 微调模型实时对话
Tab 2 - Base vs LoRA 效果对比：同一问题并列展示两个模型回复

启动：
  python app/gradio_demo.py
  # 访问 http://localhost:7860
"""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATS_FILE = PROJECT_ROOT / "data" / "data_stats.json"
DEFAULT_BASE = str(PROJECT_ROOT / "models" / "Qwen2.5-3B-Instruct")
DEFAULT_LORA = str(PROJECT_ROOT / "outputs" / "qlora_sft" / "final")


# ============ 模型持有器 ============

class ModelHolder:
    def __init__(self, label: str):
        self.label = label
        self.model = None
        self.tokenizer = None
        self.loaded = False


base_holder = ModelHolder("Base")
lora_holder = ModelHolder("LoRA")


# ============ 工具函数 ============

def load_roles() -> list:
    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return list(json.load(f).get("top_roles", {}).keys())
    return ["韦小宝", "孙悟空", "哈利波特", "爱丽丝"]


def build_instruction(role: str) -> str:
    return (
        f'你现在扮演{role}，请完全进入角色，'
        f'用{role}的语气、口头禅和性格特点来回应对方。'
        f'不要跳出角色，不要说“我是AI”之类的话。'
    )


def _load_model(base_path: str, lora_path: Optional[str], load_in_4bit: bool):
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bnb_config is None else None,
        device_map="auto",
        trust_remote_code=True,
    )
    if lora_path and Path(lora_path).exists():
        model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tokenizer


def generate(holder: ModelHolder, instruction: str, message: str,
             max_new_tokens: int, temperature: float) -> tuple:
    if not holder.loaded:
        return f"⚠️ {holder.label} 模型未加载", 0.0
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": message},
    ]
    text = holder.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = holder.tokenizer(text, return_tensors="pt").to(holder.model.device)
    t0 = time.time()
    with torch.no_grad():
        out_ids = holder.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
        )
    latency = time.time() - t0
    resp_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    response = holder.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    return response, round(latency, 2)


# ============ Gradio 回调 ============

def do_load_base(base_path: str, load_4bit: bool) -> str:
    try:
        base_holder.model, base_holder.tokenizer = _load_model(base_path, None, load_4bit)
        base_holder.loaded = True
        return "✅ Base 模型加载完成"
    except Exception as e:
        return f"❌ 加载失败: {e}"


def do_load_lora(base_path: str, lora_path: str, load_4bit: bool) -> str:
    try:
        lora_holder.model, lora_holder.tokenizer = _load_model(base_path, lora_path, load_4bit)
        lora_holder.loaded = True
        return "✅ LoRA 模型加载完成"
    except Exception as e:
        return f"❌ 加载失败: {e}"


def do_chat(history, message: str, role: str, max_tokens: int, temperature: float):
    if not message.strip():
        return history, ""
    holder = lora_holder if lora_holder.loaded else base_holder
    if not holder.loaded:
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": "⚠️ 请先在上方加载模型"})
        return history, ""
    instruction = build_instruction(role)
    response, latency = generate(holder, instruction, message, max_tokens, temperature)
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": f"{response}\n\n`[{holder.label} | {latency}s]`"})
    return history, ""


def do_compare(message: str, role: str, max_tokens: int, temperature: float):
    if not message.strip():
        return "(请输入内容)", "(请输入内容)", "请先输入消息"
    instruction = build_instruction(role)
    base_resp, base_lat = generate(base_holder, instruction, message, max_tokens, temperature)
    lora_resp, lora_lat = generate(lora_holder, instruction, message, max_tokens, temperature)
    status = f"✅ 生成完成 | Base: {base_lat}s | LoRA: {lora_lat}s"
    return base_resp, lora_resp, status


# ============ 构建 UI ============

def create_ui():
    roles = load_roles()
    default_role = roles[0] if roles else "韦小宝"

    with gr.Blocks(title="角色扮演 LLM 微调与评测平台") as demo:
        gr.Markdown("# 🎭 角色扮演 LLM 微调与评测平台")
        gr.Markdown(
            "基于 **Qwen2.5-3B-Instruct + QLoRA** 微调（ChatHaruhi-54K）\n\n"
            "支持角色扮演对话 · Base vs LoRA 效果对比"
        )

        # ─── 模型加载 ───
        with gr.Accordion("⚙️ 模型加载", open=True):
            with gr.Row():
                base_path_box = gr.Textbox(value=DEFAULT_BASE, label="Base 模型路径", scale=3)
                lora_path_box = gr.Textbox(value=DEFAULT_LORA, label="LoRA 适配器路径", scale=3)
                load_4bit_cb = gr.Checkbox(value=True, label="4-bit 量化", scale=1)
            with gr.Row():
                btn_load_base = gr.Button("加载 Base 模型", variant="secondary")
                btn_load_lora = gr.Button("加载 LoRA 模型", variant="primary")
            load_status = gr.Textbox(label="状态", interactive=False, lines=1)
            btn_load_base.click(do_load_base, [base_path_box, load_4bit_cb], load_status)
            btn_load_lora.click(do_load_lora, [base_path_box, lora_path_box, load_4bit_cb], load_status)

        with gr.Tabs():

            # ─── Tab 1: 角色扮演对话 ───
            with gr.Tab("🎭 角色扮演对话"):
                gr.Markdown(
                    "优先使用 LoRA 微调模型对话（若 LoRA 未加载则退回 Base 模型）\n\n"
                    "示例问题：「你今天怎么了？」「快跟我说说你的故事」"
                )
                with gr.Row():
                    role_dd = gr.Dropdown(choices=roles, value=default_role, label="选择角色", scale=2)
                    max_tok_sl = gr.Slider(64, 512, value=256, step=32, label="最大 token", scale=2)
                    temp_sl = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature", scale=2)

                chatbot = gr.Chatbot(label="对话", height=420)
                with gr.Row():
                    chat_input = gr.Textbox(placeholder="输入你的话...", show_label=False, scale=5)
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                clear_btn = gr.Button("🗑️ 清空对话")

                send_btn.click(
                    do_chat,
                    [chatbot, chat_input, role_dd, max_tok_sl, temp_sl],
                    [chatbot, chat_input],
                )
                chat_input.submit(
                    do_chat,
                    [chatbot, chat_input, role_dd, max_tok_sl, temp_sl],
                    [chatbot, chat_input],
                )
                clear_btn.click(lambda: ([], ""), None, [chatbot, chat_input])

            # ─── Tab 2: Base vs LoRA 对比 ───
            with gr.Tab("📊 Base vs LoRA 效果对比"):
                gr.Markdown(
                    "同一输入，左侧展示 Base 模型回复，右侧展示 LoRA 微调回复，直观感受微调效果\n\n"
                    "> 需要同时加载 Base 和 LoRA 两个模型"
                )
                with gr.Row():
                    cmp_role_dd = gr.Dropdown(choices=roles, value=default_role, label="角色", scale=2)
                    cmp_max_tok = gr.Slider(64, 512, value=200, step=32, label="最大 token", scale=2)
                    cmp_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature", scale=2)

                cmp_input = gr.Textbox(label="输入问题", placeholder="输入你想问角色的内容...", lines=2)
                cmp_btn = gr.Button("🔍 对比生成", variant="primary")

                with gr.Row():
                    base_out = gr.Textbox(label="🔹 Base 模型回复", lines=10, interactive=False)
                    lora_out = gr.Textbox(label="🔸 LoRA 微调回复", lines=10, interactive=False)

                cmp_status = gr.Textbox(label="状态", interactive=False, lines=1)

                cmp_btn.click(
                    do_compare,
                    [cmp_input, cmp_role_dd, cmp_max_tok, cmp_temp],
                    [base_out, lora_out, cmp_status],
                )

        gr.Markdown(
            "---\n"
            "**技术栈**: Qwen2.5-3B-Instruct | QLoRA (r=32, α=64) | PEFT | "
            "ChatHaruhi-54K | Gradio | FastAPI"
        )

    return demo


def main():
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
