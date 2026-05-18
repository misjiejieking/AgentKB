"""SQLite 连接管理与建表初始化。"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from loguru import logger


_DDL = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id           TEXT PRIMARY KEY,
    session_id   TEXT NOT NULL,
    role         TEXT NOT NULL CHECK(role IN ('human','ai','tool')),
    content      TEXT NOT NULL,
    tool_calls   TEXT,
    tool_results TEXT,
    created_at   TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS knowledge_files (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    file_size   INTEGER NOT NULL DEFAULT 0,
    chunk_count INTEGER DEFAULT 0,
    status      TEXT DEFAULT 'active' CHECK(status IN ('active','deleted')),
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_files_status
    ON knowledge_files(status);
"""


class Database:
    """SQLite 数据库封装，每次操作创建新连接，启用 WAL 模式与外键约束。"""

    def __init__(self, db_path: str = "data/agentkb.db") -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_DDL)
        logger.info(f"Database initialized at {self._path}")

    # ── sessions ─────────────────────────────────────────────

    def create_session(self, session_id: str, title: str = "New Chat") -> dict[str, Any]:
        from .models import now_iso

        now = now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (session_id, title, now, now),
            )
        return {"id": session_id, "title": title, "created_at": now}

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> None:
        from .models import now_iso

        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET title = ?, updated_at = ? WHERE id = ?",
                (title, now_iso(), session_id),
            )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM messages WHERE session_id = s.id) AS message_count "
                "FROM sessions s ORDER BY updated_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        return cur.rowcount > 0

    # ── messages ─────────────────────────────────────────────

    def add_message(
        self,
        msg_id: str,
        session_id: str,
        role: str,
        content: str,
        tool_calls: str | None = None,
        tool_results: str | None = None,
    ) -> dict[str, Any]:
        from .models import now_iso

        now = now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (id, session_id, role, content, tool_calls, tool_results, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (msg_id, session_id, role, content, tool_calls, tool_results, now),
            )
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            # 自动标题：如果是第一条 human 消息且标题为 New Chat
            if role == "human":
                row = conn.execute(
                    "SELECT title FROM sessions WHERE id = ?", (session_id,)
                ).fetchone()
                if row and row["title"] == "New Chat":
                    title = content[:40] + ("..." if len(content) > 40 else "")
                    conn.execute(
                        "UPDATE sessions SET title = ? WHERE id = ?",
                        (title, session_id),
                    )
        return {
            "id": msg_id,
            "session_id": session_id,
            "role": role,
            "content": content,
            "created_at": now,
        }

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY created_at ASC",
                (session_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def clear_messages(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))

    # ── knowledge files ──────────────────────────────────────

    def add_knowledge_file(
        self, file_id: str, filename: str, filepath: str, file_size: int, chunk_count: int
    ) -> None:
        from .models import now_iso

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_files (id, filename, filepath, file_size, chunk_count, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, 'active', ?)",
                (file_id, filename, filepath, file_size, chunk_count, now_iso()),
            )

    def get_knowledge_file(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_files WHERE id = ?", (file_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_knowledge_files(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_files WHERE status = 'active' ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def update_knowledge_file(self, file_id: str, **kwargs: Any) -> None:
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [file_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE knowledge_files SET {set_clause} WHERE id = ?", values
            )

    def delete_knowledge_file(self, file_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE knowledge_files SET status = 'deleted' WHERE id = ?",
                (file_id,),
            )
        return cur.rowcount > 0


# 模块级单例：避免重复初始化数据库连接
_db: Database | None = None


def get_db(db_path: str | None = None) -> Database:
    """获取或创建 Database 单例。"""
    global _db
    if _db is None:
        path = db_path or "data/agentkb.db"
        _db = Database(path)
    return _db
