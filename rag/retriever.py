"""
RAG 角色知识检索模块
根据角色名从知识库中检索背景、口头禅、性格、关系等信息，注入 system prompt。
使用简单的精确匹配（角色名 → 知识条目），无需额外依赖。
"""

import json
from pathlib import Path

KNOWLEDGE_FILE = Path(__file__).parent / "knowledge" / "characters.json"

_knowledge_cache: dict = {}


def _load_knowledge() -> dict:
    global _knowledge_cache
    if _knowledge_cache:
        return _knowledge_cache
    if KNOWLEDGE_FILE.exists():
        with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
            _knowledge_cache = json.load(f)
    return _knowledge_cache


def get_character_context(role: str) -> str:
    """根据角色名检索知识，返回可直接拼入 system prompt 的文本"""
    kb = _load_knowledge()
    info = kb.get(role)
    if not info:
        return ""

    parts = []
    if info.get("background"):
        parts.append(f"【角色背景】{info['background']}")
    if info.get("personality"):
        parts.append(f"【性格特点】{info['personality']}")
    if info.get("catchphrases"):
        parts.append(f"【口头禅/经典台词】{'、'.join(info['catchphrases'])}")
    if info.get("relationships"):
        parts.append(f"【人物关系】{info['relationships']}")

    return "\n".join(parts)


def list_known_characters() -> list:
    """返回知识库中已有知识的角色列表"""
    return list(_load_knowledge().keys())
