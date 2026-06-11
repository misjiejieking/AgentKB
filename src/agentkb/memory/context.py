"""会话上下文窗口与持久化摘要。"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from agentkb.config.settings import Settings

SUMMARY_PROMPT = """请把已有摘要和新增对话合并成一份简洁、准确的会话摘要。

要求：
- 保留用户目标、约束、偏好、已确认结论和未完成事项
- 不记录寒暄、重复表达和工具过程噪声
- 不补充对话中没有的信息
- 直接输出摘要正文，不使用 Markdown 标题

已有摘要：
{summary}

新增对话：
{transcript}
"""


def select_conversation_context(
    messages: list[AnyMessage],
    summary: str,
    max_turns: int,
) -> list[AnyMessage]:
    """保留最近完整轮次，避免从 ToolMessage 中间截断协议链。"""
    if max_turns < 1:
        raise ValueError("max_turns 必须大于 0")

    human_indexes = [
        index
        for index, message in enumerate(messages)
        if isinstance(message, HumanMessage)
    ]
    start = human_indexes[-max_turns] if len(human_indexes) > max_turns else 0
    recent: list[AnyMessage] = [
        message
        for message in messages[start:]
        if not isinstance(message, SystemMessage)
    ]
    if summary and start > 0:
        return [SystemMessage(content=f"较早会话摘要：{summary}"), *recent]
    return recent


class SessionSummaryService:
    """增量压缩超出工作窗口的会话消息。"""

    def __init__(self, db=None) -> None:
        self._db = db

    @property
    def db(self):
        if self._db is None:
            from agentkb.storage.pg_database import get_db

            self._db = get_db()
        return self._db

    def get_summary(self, session_id: str) -> str:
        record = self.db.get_session_summary(session_id)
        return str(record["summary"]) if record else ""

    async def refresh(
        self,
        session_id: str,
        *,
        max_turns: int,
        llm=None,
    ) -> bool:
        """把窗口外且尚未覆盖的消息合并进摘要。"""
        rows = await asyncio.to_thread(self.db.get_messages, session_id)
        rows = [row for row in rows if str(row.get("content", "")).strip()]
        human_indexes = [
            index for index, row in enumerate(rows) if row["role"] == "human"
        ]
        if len(human_indexes) <= max_turns:
            return False

        cutoff = human_indexes[-max_turns]
        older_rows = rows[:cutoff]
        record = await asyncio.to_thread(self.db.get_session_summary, session_id)
        covered_sequence = int(record["covered_sequence"]) if record else 0
        pending_rows = [
            row
            for row in older_rows
            if int(row["sequence"]) > covered_sequence
        ]
        if not pending_rows:
            return False

        transcript = self._format_transcript(pending_rows)
        current_summary = str(record["summary"]) if record else "（无）"
        prompt = SUMMARY_PROMPT.format(
            summary=current_summary,
            transcript=transcript,
        )

        if llm is None:
            from agentkb.llm.factory import get_router_chat_model

            llm = get_router_chat_model(streaming=False)
        response = await asyncio.wait_for(
            llm.ainvoke(prompt),
            timeout=Settings.load().llm_request_timeout,
        )
        summary = str(getattr(response, "content", response)).strip()
        if not summary:
            raise ValueError("会话摘要模型返回空内容")

        await asyncio.to_thread(
            self.db.upsert_session_summary,
            session_id,
            summary[:4000],
            int(older_rows[-1]["sequence"]),
        )
        return True

    @staticmethod
    def _format_transcript(rows: list[dict[str, Any]]) -> str:
        labels = {"human": "用户", "ai": "助手", "tool": "工具"}
        lines = [
            f"{labels.get(str(row['role']), str(row['role']))}: "
            f"{str(row['content'])[:1000]}"
            for row in rows
        ]
        return "\n".join(lines)[-8000:]
