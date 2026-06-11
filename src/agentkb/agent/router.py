"""意图路由器——用轻量 prompt 做意图分类，减少不必要工具调用。"""

from __future__ import annotations

from enum import Enum

from loguru import logger


class Intent(str, Enum):
    CHAT = "chat"  # 闲聊/常识——直接回复
    KNOWLEDGE = "knowledge_search"  # 搜本地知识库
    WEB = "web_search"  # 联网搜索
    HYBRID = "hybrid"  # 知识库 + 联网都要


ROUTER_PROMPT = """分析用户意图，从以下选项中选择最合适的：

- chat: 闲聊、打招呼、常识性问题、不需要任何工具
- knowledge_search: 涉及用户文档/内部资料/制度/笔记
- web_search: 实时信息、新闻、天气、股价
- hybrid: 两者都需要

只回复一个词：chat / knowledge_search / web_search / hybrid

用户消息: {query}"""


class IntentRouter:
    """轻量意图路由器——分类用户意图后决定工具调用策略。"""

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    async def classify(self, query: str) -> Intent:
        """返回用户意图分类结果。"""
        # 快速关键词匹配——避免不必要的 LLM 调用
        intent = self._fast_match(query)
        if intent:
            return intent

        if self._llm is None:
            return Intent.KNOWLEDGE

        try:
            prompt = ROUTER_PROMPT.format(query=query)
            response = await self._llm.ainvoke(prompt)
            content = response.content.strip().lower() if hasattr(response, "content") else str(response).strip().lower()

            if "hybrid" in content:
                return Intent.HYBRID
            if "web_search" in content or "web" in content:
                return Intent.WEB
            if "knowledge_search" in content or "knowledge" in content:
                return Intent.KNOWLEDGE
            return Intent.CHAT
        except Exception as e:
            logger.warning(f"意图路由失败，默认 knowledge_search: {e}")
            return Intent.KNOWLEDGE

    @staticmethod
    def _fast_match(query: str) -> Intent | None:
        """关键词快速匹配，命中明确模式则跳过 LLM 调用。"""
        q = query.strip().lower()

        # 纯闲聊
        greetings = ["你好", "hi", "hello", "嘿", "早", "晚安", "再见", "谢谢", "谢了"]
        if any(q == g for g in greetings):
            return Intent.CHAT
        if len(q) <= 3 and not any(kw in q for kw in ["制度", "文件", "文档", "规定", "政策", "怎么", "什么", "如何"]):
            return Intent.CHAT

        # 实时信息
        realtime_kw = ["天气", "股价", "新闻", "最新", "今天", "明天", "本周", "实时"]
        if any(kw in q for kw in realtime_kw):
            return Intent.WEB

        # 本地知识
        local_kw = ["制度", "规定", "政策", "我的笔记", "文件", "文档", "上传", "手册",
                     "考勤", "年假", "请假", "报销", "流程", "部门", "公司"]
        if any(kw in q for kw in local_kw):
            return Intent.KNOWLEDGE

        return None  # 无法快速判断，交给 LLM
