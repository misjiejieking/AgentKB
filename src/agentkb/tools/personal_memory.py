"""长期个人记忆的显式搜索与保存工具。"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Generator, Literal

from pydantic import BaseModel, Field

from agentkb.config.settings import Settings
from agentkb.memory.long_term import LongTermMemory
from agentkb.tools.base import BaseTool, ToolResult

_memory_session_id: ContextVar[str] = ContextVar(
    "personal_memory_session_id",
    default="",
)


@contextmanager
def personal_memory_context(session_id: str) -> Generator[None, None, None]:
    """为当前工具调用绑定来源会话。"""
    token = _memory_session_id.set(session_id)
    try:
        yield
    finally:
        _memory_session_id.reset(token)


class SearchPersonalMemoryInput(BaseModel):
    query: str = Field(description="要回忆的用户偏好、事实或历史经验")
    top_k: int = Field(default=5, ge=1, le=10)


class SearchPersonalMemoryTool(BaseTool):
    @property
    def name(self) -> str:
        return "search_personal_memory"

    @property
    def description(self) -> str:
        return (
            "搜索用户明确保存过的跨会话个人记忆。"
            "当用户询问自己的偏好、身份信息或之前要求记住的事项时使用。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return SearchPersonalMemoryInput

    async def _execute(
        self,
        query: str = "",
        top_k: int = 5,
        **kwargs: Any,
    ) -> ToolResult:
        if not Settings.load().memory_long_term_enabled:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="长期记忆功能未启用",
            )
        memories = await asyncio.to_thread(
            LongTermMemory().search,
            query,
            top_k,
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": query, "memories": memories, "total": len(memories)},
        )


class SavePersonalMemoryInput(BaseModel):
    content: str = Field(description="用户明确要求长期记住的完整事实或偏好")
    category: Literal[
        "fact",
        "preference",
        "experience",
        "insight",
        "general",
    ] = "general"
    importance: float = Field(default=0.8, ge=0, le=1)


class SavePersonalMemoryTool(BaseTool):
    @property
    def name(self) -> str:
        return "save_personal_memory"

    @property
    def description(self) -> str:
        return (
            "保存跨会话个人记忆。"
            "只有用户明确说“记住”“以后记得”或明确要求保存偏好时才能使用，"
            "不得自动推断并保存敏感信息。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return SavePersonalMemoryInput

    async def _execute(
        self,
        content: str = "",
        category: str = "general",
        importance: float = 0.8,
        **kwargs: Any,
    ) -> ToolResult:
        cfg = Settings.load()
        if not cfg.memory_long_term_enabled:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="长期记忆功能未启用",
            )
        if importance < cfg.memory_long_term_min_importance:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=(
                    "记忆重要性低于持久化阈值 "
                    f"{cfg.memory_long_term_min_importance}"
                ),
            )

        memory_id = await asyncio.to_thread(
            LongTermMemory().save,
            content,
            category,
            importance,
            _memory_session_id.get(),
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"memory_id": memory_id, "saved": True},
        )
