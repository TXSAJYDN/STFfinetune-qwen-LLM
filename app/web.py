"""
页面路由：直接用 Jinja2 渲染 HTML 返回（绕开 Starlette TemplateResponse 兼容问题）
"""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader
from pathlib import Path

_template_dir = str(Path(__file__).parent / "templates")
_env = Environment(loader=FileSystemLoader(_template_dir))

router = APIRouter()


def _render(name: str, **ctx) -> HTMLResponse:
    template = _env.get_template(name)
    return HTMLResponse(template.render(**ctx))


@router.get("/")
async def index():
    return _render("index.html", nav_active="home")


@router.get("/chat")
async def chat():
    return _render("chat.html", nav_active="chat")


@router.get("/compare")
async def compare():
    return _render("compare.html", nav_active="compare")


@router.get("/data")
async def data_page():
    return _render("data.html", nav_active="data")


@router.get("/training")
async def training_page():
    return _render("training.html", nav_active="training")


@router.get("/eval")
async def eval_page():
    return _render("eval.html", nav_active="eval")


@router.get("/benchmark")
async def benchmark_page():
    return _render("benchmark.html", nav_active="benchmark")
