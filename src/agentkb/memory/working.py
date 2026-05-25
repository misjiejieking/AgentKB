"""工作记忆——会话级别的短期上下文窗口。

自动维护最近 N 轮对话，支持上下文压缩（旧消息自动摘要）。
"""

from __future__ import annotations

from typing import Any
from collections import deque
from dataclasses import dataclass

from loguru import logger


@dataclass
class MemoryEntry:
    """单条记忆条目。"""
    role: str           # "user" | "assistant" | "tool" | "system"
    content: str
    importance: float = 0.5   # 重要性评分 0~1（越高越不容易被淘汰）
    tokens: int = 0


class WorkingMemory:
    """工作记忆——有限容量的会话上下文窗口。

    策略:
      - 保留最近 max_turns 轮完整对话
      - 超出容量时，对旧消息做自动摘要压缩
      - 高 importance 的消息优先保留
    """

    def __init__(self, max_turns: int = 10, max_tokens: int = 4096) -> None:
        self._entries: deque[MemoryEntry] = deque(maxlen=max_turns * 2)
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._summary: str = ""   # 被淘汰消息的摘要

    def add(self, role: str, content: str, importance: float = 0.5) -> None:
        """添加一条记忆。"""
        entry = MemoryEntry(
            role=role,
            content=content[:2048],
            importance=importance,
            tokens=len(content) // 2,
        )
        self._entries.append(entry)
        self._maybe_compress()

    def get_context(self, max_tokens: int | None = None) -> list[dict[str, str]]:
        """返回当前窗口内的对话上下文（用于组装 prompt）。"""
        limit = max_tokens or self._max_tokens
        result = []
        token_count = 0

        if self._summary:
            result.append({"role": "system", "content": f"历史对话摘要: {self._summary}"})
            token_count += len(self._summary) // 2

        for entry in reversed(self._entries):
            est = entry.tokens or len(entry.content) // 2
            if token_count + est > limit:
                break
            result.append({"role": entry.role, "content": entry.content})
            token_count += est

        result.reverse()
        return result

    def get_recent(self, n: int = 6) -> list[str]:
        """获取最近 N 条消息的纯文本列表。"""
        return [
            f"[{e.role}]: {e.content[:256]}"
            for e in list(self._entries)[-n:]
        ]

    def clear(self) -> None:
        self._entries.clear()
        self._summary = ""

    def _maybe_compress(self) -> None:
        """当容量超出限制时，压缩旧消息为摘要。"""
        max_entries = self._max_turns * 2
        if len(self._entries) <= max_entries:
            return

        # 取超出部分
        overflow = list(self._entries)[:len(self._entries) - max_entries]
        # 保留高重要性的
        keep = [e for e in overflow if e.importance > 0.8]
        to_compress = [e for e in overflow if e.importance <= 0.8]

        if to_compress:
            compressed = " ".join(e.content[:100] for e in to_compress)
            if self._summary:
                self._summary += f" | {compressed}"
            else:
                self._summary = compressed[:500]

        # 移除已压缩的消息（保留高重要性）
        for e in to_compress:
            try:
                self._entries.remove(e)
            except ValueError:
                pass

        logger.debug(f"工作记忆压缩: {len(to_compress)} → 摘要 ({len(self._summary)} chars)")
