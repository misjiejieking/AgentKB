"""会话管理器：封装消息持久化与历史恢复，桥接 LangGraph checkpointer。"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from loguru import logger

from agentkb.storage.pg_database import Database, get_db
from agentkb.storage.models import new_id


class SessionManager:
    """管理会话与消息的持久化，以及 LangChain 消息的序列化/反序列化。"""

    def __init__(self, db: Database | None = None) -> None:
        self._db = db or get_db()

    # ── 会话 ───────────────────────────────────────────────────

    def ensure_session(self, session_id: str) -> None:
        """确保会话存在，不存在则创建。"""
        if self._db.get_session(session_id) is None:
            self._db.create_session(session_id)
            logger.info(f"创建新会话: {session_id}")

    def get_session_title(self, session_id: str) -> str:
        """返回会话标题，不存在则返回默认值。"""
        s = self._db.get_session(session_id)
        return s["title"] if s else "New Chat"

    def clear_session(self, session_id: str) -> None:
        """清空会话中所有消息。"""
        self._db.clear_messages(session_id)
        logger.info(f"已清空会话消息: {session_id}")

    # ── 消息持久化 ─────────────────────────────────────────────

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_calls: list[dict] | None = None,
        tool_results: list[dict] | None = None,
    ) -> str:
        """保存单条消息到数据库，返回消息 ID。"""
        msg_id = new_id()
        self._db.add_message(
            msg_id=msg_id,
            session_id=session_id,
            role=role,
            content=content,
            tool_calls=json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None,
            tool_results=json.dumps(tool_results, ensure_ascii=False) if tool_results else None,
        )
        return msg_id

    def load_messages(self, session_id: str) -> list[dict[str, Any]]:
        """从数据库加载会话的全部消息历史。"""
        rows = self._db.get_messages(session_id)
        attachment_rows = self._db.get_session_attachments(session_id)
        attachments_by_message: dict[str, list[dict[str, Any]]] = {}
        for attachment in attachment_rows:
            attachments_by_message.setdefault(
                str(attachment["message_id"]),
                [],
            ).append({
                "id": attachment["id"],
                "name": attachment["original_name"],
                "media_type": attachment["media_type"],
                "status": attachment["status"],
                "description": attachment["description"],
            })
        messages = []
        for row in rows:
            msg = {
                "role": row["role"],
                "content": row["content"],
            }
            attachments = attachments_by_message.get(str(row["id"]), [])
            if attachments:
                msg["attachments"] = attachments
            if row["tool_calls"]:
                msg["tool_calls"] = self._decode_jsonb(row["tool_calls"])
            if row["tool_results"]:
                msg["tool_results"] = self._decode_jsonb(row["tool_results"])
            messages.append(msg)
        return messages

    @staticmethod
    def _decode_jsonb(value: Any) -> Any:
        """JSONB 驱动结果可能是原生对象，也可能是序列化字符串。"""
        return json.loads(value) if isinstance(value, str) else value

    # ── LangChain 消息序列化 ───────────────────────────────────

    @staticmethod
    def langchain_to_dict(messages: list[BaseMessage]) -> list[dict[str, Any]]:
        """将 LangChain 消息列表转为前端可渲染的 dict 列表。"""
        result = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                result.append({"role": "user", "content": str(msg.content)})
            elif isinstance(msg, AIMessage):
                entry: dict[str, Any] = {"role": "assistant", "content": str(msg.content)}
                if msg.tool_calls:
                    entry["tool_calls"] = msg.tool_calls
                result.append(entry)
            elif isinstance(msg, ToolMessage):
                result.append({
                    "role": "tool",
                    "name": getattr(msg, "name", ""),
                    "content": str(msg.content),
                })
        return result

    @staticmethod
    def dict_to_langchain(messages: list[dict[str, Any]]) -> list[BaseMessage]:
        """将数据库加载的 dict 列表反序列化为 LangChain 消息。"""
        lc_messages: list[BaseMessage] = []
        for m in messages:
            role = m.get("role", "")
            content = m.get("content", "")
            if role in ("user", "human"):
                descriptions = [
                    attachment.get("description", "").strip()
                    for attachment in m.get("attachments", [])
                    if attachment.get("description", "").strip()
                ]
                if descriptions:
                    content = (
                        f"{content}\n\n[历史图片附件分析]\n"
                        + "\n\n".join(descriptions)
                    ).strip()
                lc_messages.append(HumanMessage(content=content))
            elif role in ("assistant", "ai"):
                ai_msg = AIMessage(content=content)
                if m.get("tool_calls"):
                    ai_msg.tool_calls = m["tool_calls"]
                # 跳过纯空内容的 AI 消息（草稿），但保留带 tool_calls 的
                if (not content or not content.strip()) and not m.get("tool_calls"):
                    continue
                lc_messages.append(ai_msg)
            elif role == "tool":
                lc_messages.append(ToolMessage(
                    content=content,
                    tool_call_id=m.get("tool_call_id", ""),
                    name=m.get("name", ""),
                ))
        return lc_messages
