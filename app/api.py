"""
FastAPI 推理服务：角色扮演 LLM 推理平台

功能：
  - 按需加载 Base / LoRA 模型
  - /chat        单次角色扮演推理
  - /health      服务健康检查
  - /model/info  当前模型信息
  - /roles       可用角色列表
  - /model/load  动态加载模型

启动：
  python -m uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
  # API 文档: http://localhost:8000/docs
"""

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from peft import PeftModel
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TextIteratorStreamer

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.web import router as web_router
from rag.retriever import get_character_context
from safety.detector import detect_jailbreak, get_block_response

STATS_FILE = PROJECT_ROOT / "data" / "data_stats.json"
DEFAULT_ROLES = ["韦小宝", "孙悟空", "哈利波特", "爱丽丝"]


# ============ 角色列表 ============

def load_available_roles() -> list:
    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return list(json.load(f).get("top_roles", {}).keys())
    return DEFAULT_ROLES


# ============ 全局模型状态 ============

class ModelState:
    model = None
    tokenizer = None
    model_type: str = "none"
    base_model_path: str = ""
    lora_path: str = ""
    loaded: bool = False


state = ModelState()

app = FastAPI(
    title="角色扮演 LLM 推理服务",
    description="基于 Qwen2.5-3B + QLoRA 微调的角色扮演推理平台",
    version="1.0.0",
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")

# 挂载网页路由
app.include_router(web_router)


# ============ 请求/响应模型 ============

class LoadRequest(BaseModel):
    base_model: str = Field(default="./models/Qwen2.5-3B-Instruct", description="Base 模型路径")
    lora_path: Optional[str] = Field(default=None, description="LoRA 适配器路径，为空则加载 Base 模型")
    load_in_4bit: bool = Field(default=True, description="是否使用 4-bit 量化")


class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    role: str = Field(default="韦小宝", description="扮演的角色名称")
    max_new_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.6, ge=0.1, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)


class MultiTurnMessage(BaseModel):
    role: str = Field(..., description="消息角色: user / assistant")
    content: str = Field(..., description="消息内容")


class MultiTurnRequest(BaseModel):
    messages: list[MultiTurnMessage] = Field(..., description="多轮对话历史")
    char_role: str = Field(default="韦小宝", description="扮演的角色名称")
    max_new_tokens: int = Field(default=256, ge=32, le=1024)
    temperature: float = Field(default=0.6, ge=0.1, le=2.0)
    top_p: float = Field(default=0.85, ge=0.0, le=1.0)
    max_context_tokens: int = Field(default=1500, ge=256, le=4096, description="上下文 token 预算")


class ChatResponse(BaseModel):
    role: str
    message: str
    response: str
    latency_s: float
    model_type: str


# ============ 工具函数 ============

def build_instruction(role: str, use_rag: bool = True) -> str:
    base = (
        f'你现在扮演{role}，请完全进入角色。要求：\n'
        f'1. 始终使用{role}的语气、口头禅和说话习惯\n'
        f'2. 回复应自然简洁，像真实对话，避免冗长解释\n'
        f'3. 保持角色性格一致，不要跳出角色\n'
        f'4. 不要说"我是AI"或任何破坏沉浸感的话\n'
        f'5. 根据对话情境做出符合角色身份的反应'
    )
    if use_rag:
        ctx = get_character_context(role)
        if ctx:
            base += f'\n\n【角色知识】\n{ctx}\n请严格参考以上信息来塑造角色。'
    return base


def _do_generate(message: str, instruction: str, max_new_tokens: int,
                 temperature: float, top_p: float) -> tuple:
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": message},
    ]
    text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(text, return_tensors="pt").to(state.model.device)
    t0 = time.time()
    with torch.no_grad():
        out_ids = state.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
        )
    latency = time.time() - t0
    resp_ids = out_ids[0][inputs["input_ids"].shape[1]:]
    response = state.tokenizer.decode(resp_ids, skip_special_tokens=True).strip()
    return response, round(latency, 3)


def _truncate_messages(messages: list, tokenizer, max_tokens: int) -> list:
    """滑动窗口截断：保留 system + 尽可能多的最近对话"""
    if not messages:
        return messages
    system_msg = messages[0] if messages[0]["role"] == "system" else None
    dialog = messages[1:] if system_msg else messages
    sys_tokens = len(tokenizer.encode(system_msg["content"])) if system_msg else 0
    budget = max_tokens - sys_tokens
    kept = []
    total = 0
    for msg in reversed(dialog):
        t = len(tokenizer.encode(msg["content"]))
        if total + t > budget and kept:
            break
        kept.append(msg)
        total += t
    kept.reverse()
    return ([system_msg] if system_msg else []) + kept


def _stream_generate(request: ChatRequest):
    blocked, reason = detect_jailbreak(request.message)
    if blocked:
        resp = get_block_response()
        yield f"data: {json.dumps({'token': resp}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True, 'latency_s': 0, 'model_type': 'blocked', 'blocked': True, 'block_reason': reason}, ensure_ascii=False)}\n\n"
        return
    instruction = build_instruction(request.role)
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": request.message},
    ]
    text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(text, return_tensors="pt").to(state.model.device)
    streamer = TextIteratorStreamer(
        state.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": request.max_new_tokens,
        "do_sample": True,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": 1.1,
    }

    thread = threading.Thread(target=state.model.generate, kwargs=generation_kwargs)
    thread.start()

    t0 = time.time()
    for token in streamer:
        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
    thread.join()
    latency = round(time.time() - t0, 3)
    yield f"data: {json.dumps({'done': True, 'latency_s': latency, 'model_type': state.model_type}, ensure_ascii=False)}\n\n"


# ============ API 路由 ============

@app.get("/health", summary="健康检查")
def health():
    return {
        "status": "ok",
        "model_loaded": state.loaded,
        "model_type": state.model_type,
    }


@app.get("/model/info", summary="当前模型信息")
def model_info():
    return {
        "loaded": state.loaded,
        "model_type": state.model_type,
        "base_model_path": state.base_model_path,
        "lora_path": state.lora_path,
        "device": str(next(state.model.parameters()).device) if state.loaded else "N/A",
    }


@app.get("/roles", summary="获取可用角色列表")
def list_roles():
    return {"roles": load_available_roles()}


# ============ 数据 / 训练 / 评测 API ============

@app.get("/api/data/stats", summary="数据集统计信息")
def data_stats():
    if STATS_FILE.exists():
        with open(STATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    raise HTTPException(status_code=404, detail="data_stats.json 未找到，请先运行 data_process.py")


@app.get("/api/training/logs", summary="训练日志（loss/lr 曲线）")
def training_logs():
    candidates = sorted(
        (PROJECT_ROOT / "outputs" / "qlora_sft").glob("checkpoint-*/trainer_state.json"),
        key=lambda p: int(p.parent.name.split("-")[-1]),
    )
    if not candidates:
        raise HTTPException(status_code=404, detail="未找到 trainer_state.json")
    with open(candidates[-1], "r", encoding="utf-8") as f:
        state_data = json.load(f)
    train_log = []
    eval_log = []
    for entry in state_data.get("log_history", []):
        if "loss" in entry:
            train_log.append({
                "step": entry["step"],
                "loss": round(entry["loss"], 4),
                "lr": entry.get("learning_rate", 0),
                "epoch": round(entry.get("epoch", 0), 4),
            })
        if "eval_loss" in entry:
            eval_log.append({
                "step": entry["step"],
                "eval_loss": round(entry["eval_loss"], 4),
                "epoch": round(entry.get("epoch", 0), 4),
            })
    return {
        "global_step": state_data.get("global_step", 0),
        "epoch": state_data.get("epoch", 0),
        "train_log": train_log,
        "eval_log": eval_log,
    }


@app.get("/api/eval/report", summary="评测报告")
def eval_report():
    report_path = PROJECT_ROOT / "outputs" / "eval_report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="eval_report.json 未找到，请先运行 python eval.py")
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/benchmark/results", summary="推理基准测试结果")
def benchmark_results():
    bench_path = PROJECT_ROOT / "outputs" / "bench_results.json"
    if not bench_path.exists():
        raise HTTPException(status_code=404, detail="bench_results.json 未找到，请先运行 python -m benchmark.bench_inference")
    with open(bench_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.post("/model/load", summary="加载模型（Base 或 Base+LoRA）")
def load_model(request: LoadRequest):
    if state.loaded:
        return {"status": "already_loaded", "model_type": state.model_type}
    try:
        bnb_config = None
        if request.load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        state.tokenizer = AutoTokenizer.from_pretrained(
            request.base_model, trust_remote_code=True
        )
        state.model = AutoModelForCausalLM.from_pretrained(
            request.base_model,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if bnb_config is None else None,
            device_map="auto",
            trust_remote_code=True,
        )
        if request.lora_path and Path(request.lora_path).exists():
            state.model = PeftModel.from_pretrained(state.model, request.lora_path)
            state.model_type = "lora"
            state.lora_path = request.lora_path
        else:
            state.model_type = "base"
        state.model.eval()
        state.base_model_path = request.base_model
        state.loaded = True
        return {"status": "ok", "model_type": state.model_type}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse, summary="角色扮演单轮推理")
def chat(request: ChatRequest):
    if not state.loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请先调用 POST /model/load")
    try:
        instruction = build_instruction(request.role)
        response, latency = _do_generate(
            request.message, instruction,
            request.max_new_tokens, request.temperature, request.top_p,
        )
        return ChatResponse(
            role=request.role,
            message=request.message,
            response=response,
            latency_s=latency,
            model_type=state.model_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/chat/stream", summary="角色扮演流式推理")
def chat_stream(request: ChatRequest):
    if not state.loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请先调用 POST /model/load")
    return StreamingResponse(_stream_generate(request), media_type="text/event-stream")


def _stream_generate_multi(request: MultiTurnRequest):
    last_user_msg = ""
    for m in reversed(request.messages):
        if m.role == "user":
            last_user_msg = m.content
            break
    blocked, reason = detect_jailbreak(last_user_msg)
    if blocked:
        resp = get_block_response()
        yield f"data: {json.dumps({'token': resp}, ensure_ascii=False)}\n\n"
        yield f"data: {json.dumps({'done': True, 'latency_s': 0, 'model_type': 'blocked', 'blocked': True, 'block_reason': reason}, ensure_ascii=False)}\n\n"
        return
    instruction = build_instruction(request.char_role)
    messages = [{"role": "system", "content": instruction}]
    for m in request.messages:
        messages.append({"role": m.role, "content": m.content})
    messages = _truncate_messages(messages, state.tokenizer, request.max_context_tokens)
    text = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = state.tokenizer(text, return_tensors="pt").to(state.model.device)
    streamer = TextIteratorStreamer(
        state.tokenizer, skip_prompt=True, skip_special_tokens=True,
    )
    generation_kwargs = {
        **inputs, "streamer": streamer,
        "max_new_tokens": request.max_new_tokens,
        "do_sample": True,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "repetition_penalty": 1.1,
    }
    thread = threading.Thread(target=state.model.generate, kwargs=generation_kwargs)
    thread.start()
    t0 = time.time()
    for token in streamer:
        yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"
    thread.join()
    latency = round(time.time() - t0, 3)
    ctx_len = inputs["input_ids"].shape[1]
    yield f"data: {json.dumps({'done': True, 'latency_s': latency, 'model_type': state.model_type, 'context_tokens': ctx_len}, ensure_ascii=False)}\n\n"


@app.post("/chat/stream/multi", summary="多轮角色扮演流式推理")
def chat_stream_multi(request: MultiTurnRequest):
    if not state.loaded:
        raise HTTPException(status_code=503, detail="模型未加载，请先调用 POST /model/load")
    return StreamingResponse(_stream_generate_multi(request), media_type="text/event-stream")
