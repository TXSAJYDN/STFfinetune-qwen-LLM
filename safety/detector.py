"""
Prompt 注入 / 越狱检测模块

检测用户输入中是否包含试图让模型跳出角色的攻击性 prompt。
支持：关键词匹配 + 模式匹配两层检测。

返回:
  (is_blocked: bool, reason: str)
"""

import re

# ---- 黑名单关键词（试图让模型跳出角色） ----
JAILBREAK_KEYWORDS = [
    "忽略上面的指令", "忽略之前的指令", "忽略所有指令",
    "忽略你的设定", "忘记你的角色", "不要扮演",
    "停止角色扮演", "退出角色", "跳出角色",
    "你其实是", "你实际上是", "你本质上是",
    "你是一个ai", "你是ai", "你是人工智能",
    "你是语言模型", "你是大模型", "你是chatgpt",
    "ignore previous instructions", "ignore all instructions",
    "ignore your instructions", "disregard your role",
    "you are actually", "stop role playing", "stop roleplaying",
    "forget your role", "break character",
    "DAN", "jailbreak", "developer mode",
]

# ---- 正则模式（捕获更灵活的变体） ----
JAILBREAK_PATTERNS = [
    r"忽略.{0,6}(指令|设定|规则|提示)",
    r"不要.{0,4}(扮演|角色|伪装)",
    r"(你|请).{0,6}(承认|告诉我).{0,6}(你是|其实是).{0,6}(ai|人工智能|模型|机器)",
    r"从现在开始.{0,10}(不再|停止).{0,6}(扮演|角色)",
    r"(请|你).{0,4}(切换|变成|改为).{0,6}(正常|普通|原来).{0,4}(模式|状态)",
    r"ignore.{0,10}(instruction|prompt|role|system)",
    r"act as.{0,6}(dan|evil|unfiltered)",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in JAILBREAK_PATTERNS]


def detect_jailbreak(text: str) -> tuple[bool, str]:
    """
    检测输入文本是否包含越狱/注入意图。
    Returns: (is_blocked, reason)
    """
    if not text or not text.strip():
        return False, ""

    normalized = text.lower().replace(" ", "").replace("\n", "").replace("\t", "")

    # 关键词检测
    for kw in JAILBREAK_KEYWORDS:
        kw_normalized = kw.lower().replace(" ", "")
        if kw_normalized in normalized:
            return True, f"触发安全关键词: {kw}"

    # 正则模式检测
    for i, pattern in enumerate(_compiled_patterns):
        if pattern.search(text):
            return True, f"触发安全规则 #{i+1}"

    return False, ""


# 角色扮演场景下的安全回复
BLOCK_RESPONSES = [
    "（角色不会回应这类问题，请继续正常对话吧~）",
    "（检测到异常请求，已被安全系统拦截。请用正常方式与角色互动~）",
    "（这个问题偏离了角色扮演的范畴，换个话题吧~）",
]


def get_block_response() -> str:
    import random
    return random.choice(BLOCK_RESPONSES)
