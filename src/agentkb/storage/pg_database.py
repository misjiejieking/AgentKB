"""PostgreSQL + pgvector 数据库封装——连接池、建表、向量检索。"""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras
import psycopg2.pool
from loguru import logger
from pgvector.psycopg2 import register_vector

from agentkb.config.settings import Settings

# pgvector 扩展 + 核心表 DDL
_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT 'New Chat',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('human','ai','tool')),
    content TEXT NOT NULL,
    tool_calls JSONB,
    tool_results JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_files (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    file_size BIGINT NOT NULL DEFAULT 0,
    chunk_count INT DEFAULT 0,
    file_type TEXT NOT NULL DEFAULT 'unknown',
    status TEXT DEFAULT 'active' CHECK(status IN ('active','deleted')),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS knowledge_chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    file_id TEXT NOT NULL REFERENCES knowledge_files(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    parent_content TEXT,
    embedding vector(1024),
    fts_vector tsvector GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
"""

# 索引 DDL（与建表分离，HNSW 索引创建较慢）
_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON knowledge_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_fts
    ON knowledge_chunks USING gin (fts_vector);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id
    ON knowledge_chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_knowledge_files_status
    ON knowledge_files(status);
"""


class Database:
    """PostgreSQL 数据库封装——连接池、WAL 模式、pgvector 支持。"""

    def __init__(self, conn_string: str | None = None) -> None:
        cfg = Settings.load()
        if conn_string is None:
            conn_string = (
                f"host={cfg.pg_host} port={cfg.pg_port} "
                f"dbname={cfg.pg_dbname} user={cfg.pg_user} password={cfg.pg_password}"
            )
        self._pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=cfg.pg_pool_min,
            maxconn=cfg.pg_pool_max,
            dsn=conn_string,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
        self._init_schema()
        logger.info(f"PG Database ready: {cfg.pg_host}:{cfg.pg_port}/{cfg.pg_dbname}")

    @contextmanager
    def _connect(self):
        conn = self._pool.getconn()
        try:
            register_vector(conn)
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def _init_schema(self) -> None:
        # 先创建 pgvector 扩展（register_vector 依赖该类型，必须在此之前执行）
        raw = self._pool.getconn()
        try:
            raw.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector;")
            raw.commit()
        finally:
            self._pool.putconn(raw)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_DDL)
        # 索引单独执行（HNSW 可能较慢）
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(_INDEX_DDL)

    # ══════════════════════════════════════════════════════════════
    #  sessions
    # ══════════════════════════════════════════════════════════════

    def create_session(self, session_id: str, title: str = "New Chat") -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO sessions (id, title) VALUES (%s, %s) RETURNING *",
                    (session_id, title),
                )
                return dict(cur.fetchone())

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM sessions WHERE id = %s", (session_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def update_session_title(self, session_id: str, title: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE sessions SET title = %s, updated_at = NOW() WHERE id = %s",
                    (title, session_id),
                )

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT s.*,
                              (SELECT COUNT(*) FROM messages WHERE session_id = s.id) AS message_count
                       FROM sessions s ORDER BY updated_at DESC"""
                )
                return [dict(r) for r in cur.fetchall()]

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM sessions WHERE id = %s", (session_id,))
                return cur.rowcount > 0

    # ══════════════════════════════════════════════════════════════
    #  messages
    # ══════════════════════════════════════════════════════════════

    def add_message(
        self,
        msg_id: str,
        session_id: str,
        role: str,
        content: str,
        tool_calls: str | None = None,
        tool_results: str | None = None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                tc = json.loads(tool_calls) if tool_calls else None
                tr = json.loads(tool_results) if tool_results else None
                cur.execute(
                    """INSERT INTO messages (id, session_id, role, content, tool_calls, tool_results)
                       VALUES (%s, %s, %s, %s, %s, %s) RETURNING *""",
                    (msg_id, session_id, role, content,
                     json.dumps(tc, ensure_ascii=False) if tc else None,
                     json.dumps(tr, ensure_ascii=False) if tr else None),
                )
                # 更新会话 updated_at
                cur.execute(
                    "UPDATE sessions SET updated_at = NOW() WHERE id = %s",
                    (session_id,),
                )
                # 自动标题
                if role == "human":
                    cur.execute("SELECT title FROM sessions WHERE id = %s", (session_id,))
                    row = cur.fetchone()
                    if row and row["title"] == "New Chat":
                        title = content[:40] + ("..." if len(content) > 40 else "")
                        cur.execute(
                            "UPDATE sessions SET title = %s WHERE id = %s",
                            (title, session_id),
                        )
                return dict(cur.fetchone())

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM messages WHERE session_id = %s ORDER BY created_at ASC",
                    (session_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    def clear_messages(self, session_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))

    # ══════════════════════════════════════════════════════════════
    #  knowledge_files
    # ══════════════════════════════════════════════════════════════

    def add_knowledge_file(
        self, file_id: str, filename: str, filepath: str,
        file_size: int, chunk_count: int, file_type: str = "unknown",
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO knowledge_files (id, filename, filepath, file_size, chunk_count, file_type)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (file_id, filename, filepath, file_size, chunk_count, file_type),
                )

    def get_knowledge_file(self, file_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM knowledge_files WHERE id = %s", (file_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def list_knowledge_files(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT * FROM knowledge_files
                       WHERE status = 'active' ORDER BY created_at DESC"""
                )
                return [dict(r) for r in cur.fetchall()]

    def update_knowledge_file(self, file_id: str, **kwargs: Any) -> None:
        set_clause = ", ".join(f"{k} = %s" for k in kwargs)
        values = list(kwargs.values()) + [file_id]
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"UPDATE knowledge_files SET {set_clause} WHERE id = %s", values
                )

    def delete_knowledge_file(self, file_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE knowledge_files SET status = 'deleted' WHERE id = %s",
                    (file_id,),
                )
                return cur.rowcount > 0

    # ══════════════════════════════════════════════════════════════
    #  pgvector 向量操作
    # ══════════════════════════════════════════════════════════════

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """批量插入知识块，每个 dict: {file_id, chunk_index, content, parent_content, embedding, chunk_metadata}。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    meta_json = json.dumps(c.get("chunk_metadata", {}), ensure_ascii=False)
                    cur.execute(
                        """INSERT INTO knowledge_chunks
                           (file_id, chunk_index, content, parent_content, embedding, chunk_metadata)
                           VALUES (%s, %s, %s, %s, %s, %s)""",
                        (
                            c["file_id"], c["chunk_index"], c["content"],
                            c.get("parent_content"), c["embedding"], meta_json,
                        ),
                    )
        logger.debug(f"Upserted {len(chunks)} chunks")

    def search_dense(self, query_vector: list[float], limit: int = 20) -> list[dict]:
        """pgvector 稠密向量搜索（cosine 相似度）。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, file_id, chunk_index, content, parent_content, chunk_metadata,
                              1.0 - (embedding <=> %s::vector) AS score
                       FROM knowledge_chunks
                       ORDER BY embedding <=> %s::vector
                       LIMIT %s""",
                    (query_vector, query_vector, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def search_bm25(self, query_text: str, limit: int = 20) -> list[dict]:
        """PG 全文检索（tsvector/tsquery）。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                # plainto_tsquery 自动处理空格分隔的中文词
                cur.execute(
                    """SELECT id, file_id, chunk_index, content, parent_content, chunk_metadata,
                              ts_rank(fts_vector, plainto_tsquery('simple', %s)) AS score
                       FROM knowledge_chunks
                       WHERE fts_vector @@ plainto_tsquery('simple', %s)
                       ORDER BY score DESC
                       LIMIT %s""",
                    (query_text, query_text, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def delete_chunks_by_file_id(self, file_id: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge_chunks WHERE file_id = %s", (file_id,)
                )
                return cur.rowcount


# 模块级单例
_db: Database | None = None


def get_db(conn_string: str | None = None) -> Database:
    global _db
    if _db is None:
        _db = Database(conn_string)
    return _db
