from __future__ import annotations

from typing import cast

from agentkb.session.manager import SessionManager
from agentkb.storage.pg_database import Database


class _MessageDatabase:
    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows

    def get_messages(self, session_id: str) -> list[dict]:
        return self._rows

    def get_session_attachments(self, session_id: str) -> list[dict]:
        return []


def test_load_messages_accepts_native_and_serialized_jsonb() -> None:
    database = _MessageDatabase([
        {
            "id": "message-1",
            "role": "ai",
            "content": "",
            "tool_calls": [{"id": "native"}],
            "tool_results": '[{"success": true}]',
        },
    ])

    messages = SessionManager(cast(Database, database)).load_messages("session-1")

    assert messages == [{
        "role": "ai",
        "content": "",
        "tool_calls": [{"id": "native"}],
        "tool_results": [{"success": True}],
    }]
