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
from agentkb.storage.migrations import apply_migrations

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
    session_id TEXT NOT NULL,
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
    file_id TEXT NOT NULL,
    chunk_index INT NOT NULL,
    content TEXT NOT NULL,
    parent_content TEXT,
    embedding vector(1024),
    fts_vector tsvector,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT NOT NULL DEFAULT '',
    message_id TEXT NOT NULL DEFAULT '',
    rating TEXT NOT NULL CHECK(rating IN ('up','down')),
    reason TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- V2 新增: 长期记忆表
CREATE TABLE IF NOT EXISTS long_term_memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    importance REAL NOT NULL DEFAULT 0.5,
    source_session TEXT NOT NULL DEFAULT '',
    embedding vector(1024),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    access_count INT NOT NULL DEFAULT 0
);

-- V2 新增: Agent 执行日志表
CREATE TABLE IF NOT EXISTS agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id TEXT NOT NULL,
    session_id TEXT NOT NULL DEFAULT '',
    agent_name TEXT NOT NULL,
    intent TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    output TEXT NOT NULL DEFAULT '',
    success BOOLEAN NOT NULL DEFAULT true,
    elapsed_ms REAL NOT NULL DEFAULT 0,
    tokens_used INT NOT NULL DEFAULT 0,
    tool_calls_count INT NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- V2 新增: Trace 持久化表
CREATE TABLE IF NOT EXISTS traces (
    trace_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL DEFAULT '',
    query TEXT NOT NULL DEFAULT '',
    data JSONB NOT NULL DEFAULT '{}',
    total_elapsed_ms REAL NOT NULL DEFAULT 0,
    total_tokens INT NOT NULL DEFAULT 0,
    total_tool_calls INT NOT NULL DEFAULT 0,
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
CREATE INDEX IF NOT EXISTS idx_long_term_memories_embedding
    ON long_term_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_agent_executions_trace
    ON agent_executions(trace_id);
CREATE INDEX IF NOT EXISTS idx_agent_executions_session
    ON agent_executions(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_traces_session
    ON traces(session_id, created_at);
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
            apply_migrations(conn)
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
                cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))
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
                result = dict(cur.fetchone())
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
                return result

    def get_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM messages WHERE session_id = %s ORDER BY sequence ASC",
                    (session_id,),
                )
                return [dict(r) for r in cur.fetchall()]

    def clear_messages(self, session_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM messages WHERE session_id = %s", (session_id,))

    def update_message_content(self, msg_id: str, content: str) -> None:
        """更新消息内容（流式过程中持续回写）。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE messages SET content = %s WHERE id = %s",
                    (content, msg_id),
                )

    def get_session_summary(self, session_id: str) -> dict[str, Any] | None:
        """读取会话摘要及其已覆盖的消息序号。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM session_summaries WHERE session_id = %s",
                    (session_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def upsert_session_summary(
        self,
        session_id: str,
        summary: str,
        covered_sequence: int,
    ) -> None:
        """推进会话摘要覆盖范围，不允许旧任务回退摘要版本。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO session_summaries (
                        session_id, summary, covered_sequence
                    )
                    VALUES (%s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        summary = EXCLUDED.summary,
                        covered_sequence = EXCLUDED.covered_sequence,
                        updated_at = NOW()
                    WHERE session_summaries.covered_sequence < EXCLUDED.covered_sequence
                    """,
                    (session_id, summary, covered_sequence),
                )

    # ══════════════════════════════════════════════════════════════
    #  agent runs / SSE events
    # ══════════════════════════════════════════════════════════════

    def create_chat_run(
        self,
        *,
        run_id: str,
        session_id: str,
        human_message_id: str,
        ai_message_id: str,
        message: str,
        mode: str,
        attachment_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """原子创建会话消息、Agent Run 和首个 SSE 事件。"""
        attachment_ids = attachment_ids or []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, title)
                    VALUES (%s, 'New Chat')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (session_id,),
                )
                if attachment_ids:
                    cur.execute(
                        """
                        SELECT id
                        FROM chat_attachments
                        WHERE id = ANY(%s)
                          AND session_id = %s
                          AND message_id IS NULL
                        FOR UPDATE
                        """,
                        (attachment_ids, session_id),
                    )
                    claimed = {row["id"] for row in cur.fetchall()}
                    if claimed != set(attachment_ids):
                        raise ValueError("附件不存在、已被使用或不属于当前会话")
                cur.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content)
                    VALUES (%s, %s, 'human', %s)
                    """,
                    (human_message_id, session_id, message),
                )
                if attachment_ids:
                    cur.execute(
                        """
                        UPDATE chat_attachments
                        SET message_id = %s
                        WHERE id = ANY(%s)
                        """,
                        (human_message_id, attachment_ids),
                    )
                cur.execute(
                    """
                    INSERT INTO messages (id, session_id, role, content)
                    VALUES (%s, %s, 'ai', '')
                    """,
                    (ai_message_id, session_id),
                )
                cur.execute(
                    """
                    INSERT INTO agent_runs (
                        id, session_id, ai_message_id, mode, user_input, status,
                        last_event_id
                    )
                    VALUES (%s, %s, %s, %s, %s, 'queued', 1)
                    RETURNING *
                    """,
                    (run_id, session_id, ai_message_id, mode, message),
                )
                run = dict(cur.fetchone())
                cur.execute(
                    """
                    INSERT INTO run_events (run_id, event_id, payload)
                    VALUES (%s, 1, %s::jsonb)
                    """,
                    (
                        run_id,
                        json.dumps(
                            {
                                "type": "message_id",
                                "message_id": ai_message_id,
                                "run_id": run_id,
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                title = message[:40] + ("..." if len(message) > 40 else "")
                cur.execute(
                    """
                    UPDATE sessions
                    SET title = CASE WHEN title = 'New Chat' THEN %s ELSE title END,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (title, session_id),
                )
                return run

    def create_chat_attachment(
        self,
        *,
        attachment_id: str,
        session_id: str,
        original_name: str,
        filepath: str,
        media_type: str,
        file_size: int,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO sessions (id, title)
                    VALUES (%s, 'New Chat')
                    ON CONFLICT (id) DO NOTHING
                    """,
                    (session_id,),
                )
                cur.execute(
                    """
                    INSERT INTO chat_attachments (
                        id, session_id, original_name, filepath,
                        media_type, file_size
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING *
                    """,
                    (
                        attachment_id,
                        session_id,
                        original_name,
                        filepath,
                        media_type,
                        file_size,
                    ),
                )
                return dict(cur.fetchone())

    def get_chat_attachment(
        self,
        attachment_id: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM chat_attachments WHERE id = %s",
                    (attachment_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_chat_attachments(
        self,
        attachment_ids: list[str],
        session_id: str,
    ) -> list[dict[str, Any]]:
        if not attachment_ids:
            return []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM chat_attachments
                    WHERE id = ANY(%s) AND session_id = %s
                    ORDER BY created_at
                    """,
                    (attachment_ids, session_id),
                )
                return [dict(row) for row in cur.fetchall()]

    def get_session_attachments(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM chat_attachments
                    WHERE session_id = %s AND message_id IS NOT NULL
                    ORDER BY created_at
                    """,
                    (session_id,),
                )
                return [dict(row) for row in cur.fetchall()]

    def get_all_session_attachments(
        self,
        session_id: str,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM chat_attachments
                    WHERE session_id = %s
                    ORDER BY created_at
                    """,
                    (session_id,),
                )
                return [dict(row) for row in cur.fetchall()]

    def update_chat_attachment_analysis(
        self,
        attachment_id: str,
        *,
        status: str,
        description: str = "",
        error: str = "",
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_attachments
                    SET status = %s,
                        description = %s,
                        error = %s,
                        analyzed_at = CASE
                            WHEN %s = 'analyzed' THEN NOW()
                            ELSE analyzed_at
                        END
                    WHERE id = %s
                    """,
                    (status, description, error[:2000], status, attachment_id),
                )

    def delete_unclaimed_chat_attachment(
        self,
        attachment_id: str,
        session_id: str,
    ) -> str | None:
        """删除尚未发送的附件并返回文件路径。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chat_attachments
                    WHERE id = %s
                      AND session_id = %s
                      AND message_id IS NULL
                    RETURNING filepath
                    """,
                    (attachment_id, session_id),
                )
                row = cur.fetchone()
        return str(row["filepath"]) if row else None

    def cleanup_stale_chat_attachments(self) -> list[str]:
        """清理超过 24 小时仍未发送的附件记录。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chat_attachments
                    WHERE message_id IS NULL
                      AND created_at < NOW() - INTERVAL '24 hours'
                    RETURNING filepath
                    """
                )
                return [str(row["filepath"]) for row in cur.fetchall()]

    # ══════════════════════════════════════════════════════════════
    #  custom agents
    # ══════════════════════════════════════════════════════════════

    def create_custom_agent(
        self,
        *,
        agent_id: str,
        name: str,
        display_name: str,
        description: str,
        instructions: str,
        intents: list[str],
        allowed_tools: list[str],
        model_name: str | None,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO custom_agents (
                        id, name, display_name, description, instructions,
                        intents, allowed_tools, model_name
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s
                    )
                    RETURNING *
                    """,
                    (
                        agent_id,
                        name,
                        display_name,
                        description,
                        instructions,
                        json.dumps(intents, ensure_ascii=False),
                        json.dumps(allowed_tools, ensure_ascii=False),
                        model_name,
                    ),
                )
                return dict(cur.fetchone())

    def get_custom_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM custom_agents WHERE id = %s",
                    (agent_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def list_custom_agents(
        self,
        *,
        active_only: bool = False,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM custom_agents
                    WHERE NOT %s OR status = 'active'
                    ORDER BY created_at
                    """,
                    (active_only,),
                )
                return [dict(row) for row in cur.fetchall()]

    def set_custom_agent_status(
        self,
        agent_id: str,
        status: str,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE custom_agents
                    SET status = %s, updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (status, agent_id),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def delete_custom_agent(self, agent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM custom_agents
                    WHERE id = %s
                    RETURNING *
                    """,
                    (agent_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    # ══════════════════════════════════════════════════════════════
    #  MCP servers and tools
    # ══════════════════════════════════════════════════════════════

    def create_mcp_server(self, server: dict[str, Any]) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO mcp_servers (
                        id, name, transport, command, args, url, env, headers,
                        confirmation_policy
                    )
                    VALUES (
                        %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s::jsonb, %s
                    )
                    RETURNING *
                    """,
                    (
                        server["id"],
                        server["name"],
                        server["transport"],
                        server.get("command"),
                        json.dumps(server.get("args", []), ensure_ascii=False),
                        server.get("url"),
                        json.dumps(server.get("env", {}), ensure_ascii=False),
                        json.dumps(server.get("headers", {}), ensure_ascii=False),
                        server["confirmation_policy"],
                    ),
                )
                return dict(cur.fetchone())

    def get_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM mcp_servers WHERE id = %s", (server_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def list_mcp_servers(self, *, enabled_only: bool = False) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT s.*,
                           COUNT(t.remote_name) AS tool_count,
                           COUNT(t.remote_name) FILTER (WHERE t.enabled) AS enabled_tool_count
                    FROM mcp_servers s
                    LEFT JOIN mcp_tools t ON t.server_id = s.id
                    WHERE NOT %s OR s.status = 'enabled'
                    GROUP BY s.id
                    ORDER BY s.created_at
                    """,
                    (enabled_only,),
                )
                return [dict(row) for row in cur.fetchall()]

    def set_mcp_server_connection(
        self,
        server_id: str,
        *,
        connection_status: str,
        enabled: bool | None = None,
        error: str = "",
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mcp_servers
                    SET connection_status = %s,
                        status = CASE
                            WHEN %s IS NULL THEN status
                            WHEN %s THEN 'enabled'
                            ELSE 'disabled'
                        END,
                        last_error = %s,
                        last_connected_at = CASE
                            WHEN %s = 'connected' THEN NOW()
                            ELSE last_connected_at
                        END,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (
                        connection_status,
                        enabled,
                        enabled,
                        error[:2000],
                        connection_status,
                        server_id,
                    ),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def delete_mcp_server(self, server_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM mcp_servers WHERE id = %s RETURNING *",
                    (server_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def replace_mcp_tools(
        self,
        server_id: str,
        tools: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                remote_names = [tool["remote_name"] for tool in tools]
                cur.execute(
                    """
                    DELETE FROM mcp_tools
                    WHERE server_id = %s
                      AND NOT (remote_name = ANY(%s))
                    """,
                    (server_id, remote_names),
                )
                for tool in tools:
                    cur.execute(
                        """
                        INSERT INTO mcp_tools (
                            server_id, remote_name, local_name, description,
                            input_schema, annotations, requires_confirmation
                        )
                        VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                        ON CONFLICT (server_id, remote_name) DO UPDATE SET
                            local_name = EXCLUDED.local_name,
                            description = EXCLUDED.description,
                            input_schema = EXCLUDED.input_schema,
                            annotations = EXCLUDED.annotations,
                            requires_confirmation = EXCLUDED.requires_confirmation,
                            last_seen_at = NOW()
                        """,
                        (
                            server_id,
                            tool["remote_name"],
                            tool["local_name"],
                            tool["description"],
                            json.dumps(tool["input_schema"], ensure_ascii=False),
                            json.dumps(tool["annotations"], ensure_ascii=False),
                            tool["requires_confirmation"],
                        ),
                    )
                cur.execute(
                    """
                    SELECT *
                    FROM mcp_tools
                    WHERE server_id = %s
                    ORDER BY local_name
                    """,
                    (server_id,),
                )
                return [dict(row) for row in cur.fetchall()]

    def list_mcp_tools(self, server_id: str | None = None) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM mcp_tools
                    WHERE %s IS NULL OR server_id = %s
                    ORDER BY local_name
                    """,
                    (server_id, server_id),
                )
                return [dict(row) for row in cur.fetchall()]

    def set_mcp_tool_enabled(
        self,
        server_id: str,
        remote_name: str,
        enabled: bool,
    ) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE mcp_tools
                    SET enabled = %s
                    WHERE server_id = %s AND remote_name = %s
                    RETURNING *
                    """,
                    (enabled, server_id, remote_name),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM agent_runs WHERE id = %s", (run_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def get_active_run(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM agent_runs
                    WHERE session_id = %s
                      AND status IN ('queued', 'running', 'waiting_approval')
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_latest_run(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM agent_runs
                    WHERE session_id = %s
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (session_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def append_run_event(self, run_id: str, event: dict[str, Any]) -> int:
        """递增事件序号并持久化一条 SSE 事件。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_runs
                    SET last_event_id = last_event_id + 1,
                        updated_at = NOW()
                    WHERE id = %s
                    RETURNING last_event_id
                    """,
                    (run_id,),
                )
                row = cur.fetchone()
                if row is None:
                    raise KeyError(f"Agent Run 不存在: {run_id}")
                event_id = int(row["last_event_id"])
                cur.execute(
                    """
                    INSERT INTO run_events (run_id, event_id, payload)
                    VALUES (%s, %s, %s::jsonb)
                    """,
                    (run_id, event_id, json.dumps(event, ensure_ascii=False)),
                )
                return event_id

    def get_run_events(
        self,
        run_id: str,
        after_event_id: int = 0,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT event_id, payload, created_at
                    FROM run_events
                    WHERE run_id = %s AND event_id > %s
                    ORDER BY event_id ASC
                    LIMIT %s
                    """,
                    (run_id, after_event_id, limit),
                )
                return [dict(row) for row in cur.fetchall()]

    def update_run_status(
        self,
        run_id: str,
        status: str,
        error: str = "",
    ) -> None:
        allowed = {
            "queued",
            "running",
            "waiting_approval",
            "completed",
            "failed",
            "cancelled",
            "interrupted",
        }
        if status not in allowed:
            raise ValueError(f"非法 Agent Run 状态: {status}")

        terminal = status in {"completed", "failed", "cancelled", "interrupted"}
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_runs
                    SET status = %s,
                        error = %s,
                        started_at = CASE
                            WHEN %s = 'running' AND started_at IS NULL THEN NOW()
                            ELSE started_at
                        END,
                        completed_at = CASE
                            WHEN %s THEN NOW()
                            ELSE completed_at
                        END,
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, error[:2000], status, terminal, run_id),
                )

    def claim_waiting_run(self, run_id: str) -> bool:
        """原子抢占等待审批的 Run，防止重复恢复执行。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE agent_runs
                    SET status = 'running',
                        updated_at = NOW()
                    WHERE id = %s AND status = 'waiting_approval'
                    """,
                    (run_id,),
                )
                return cur.rowcount == 1

    def interrupt_incomplete_runs(self) -> int:
        """应用启动时终止上次进程遗留的未完成 Run。"""
        message = "服务已重启，本次回答被中断，请重新发送问题。"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, ai_message_id, last_event_id
                    FROM agent_runs
                    WHERE status IN ('queued', 'running')
                    FOR UPDATE
                    """
                )
                runs = list(cur.fetchall())
                for run in runs:
                    event_id = int(run["last_event_id"]) + 1
                    cur.execute(
                        """
                        INSERT INTO run_events (run_id, event_id, payload)
                        VALUES (%s, %s, %s::jsonb)
                        """,
                        (
                            run["id"],
                            event_id,
                            json.dumps(
                                {"type": "error", "message": message},
                                ensure_ascii=False,
                            ),
                        ),
                    )
                    cur.execute(
                        """
                        UPDATE messages
                        SET content = CASE WHEN content = '' THEN %s ELSE content END
                        WHERE id = %s
                        """,
                        (message, run["ai_message_id"]),
                    )
                    cur.execute(
                        """
                        UPDATE agent_runs
                        SET status = 'interrupted',
                            error = %s,
                            last_event_id = %s,
                            completed_at = NOW(),
                            updated_at = NOW()
                        WHERE id = %s
                        """,
                        (message, event_id, run["id"]),
                    )
                return len(runs)

    # ══════════════════════════════════════════════════════════════
    #  tool approvals
    # ══════════════════════════════════════════════════════════════

    def create_tool_approval(
        self,
        *,
        approval_id: str,
        run_id: str,
        session_id: str,
        thread_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """幂等创建工具审批，节点恢复重放时不会产生重复记录。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO tool_approvals (
                        id, run_id, session_id, thread_id, tool_call_id,
                        tool_name, arguments, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, 'pending')
                    ON CONFLICT (thread_id, tool_call_id) DO NOTHING
                    """,
                    (
                        approval_id,
                        run_id,
                        session_id,
                        thread_id,
                        tool_call_id,
                        tool_name,
                        json.dumps(arguments, ensure_ascii=False),
                    ),
                )
                cur.execute(
                    """
                    SELECT *
                    FROM tool_approvals
                    WHERE thread_id = %s AND tool_call_id = %s
                    """,
                    (thread_id, tool_call_id),
                )
                return dict(cur.fetchone())

    def get_tool_approval(self, approval_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM tool_approvals WHERE id = %s",
                    (approval_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def decide_tool_approval(
        self,
        approval_id: str,
        approved: bool,
    ) -> dict[str, Any] | None:
        """记录用户决策；重复提交相同决策保持幂等。"""
        target_status = "approved" if approved else "rejected"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tool_approvals
                    SET status = %s,
                        decided_at = COALESCE(decided_at, NOW())
                    WHERE id = %s AND status = 'pending'
                    """,
                    (target_status, approval_id),
                )
                cur.execute(
                    "SELECT * FROM tool_approvals WHERE id = %s",
                    (approval_id,),
                )
                row = cur.fetchone()
        if row is None or row["status"] != target_status:
            return None
        return dict(row)

    def complete_tool_approval(
        self,
        approval_id: str,
        *,
        success: bool,
        result: Any = None,
        error: str = "",
    ) -> None:
        status = "completed" if success else "failed"
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE tool_approvals
                    SET status = %s,
                        result = %s::jsonb,
                        error = %s,
                        completed_at = NOW()
                    WHERE id = %s AND status = 'approved'
                    """,
                    (
                        status,
                        json.dumps(result, ensure_ascii=False),
                        error[:4000],
                        approval_id,
                    ),
                )

    # ══════════════════════════════════════════════════════════════
    #  long-term memories
    # ══════════════════════════════════════════════════════════════

    def add_long_term_memory(
        self,
        *,
        memory_id: str,
        content: str,
        category: str,
        importance: float,
        source_session: str,
        embedding: list[float],
    ) -> str:
        """写入长期记忆；相同内容直接返回已有记录。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM long_term_memories
                    WHERE content = %s
                    LIMIT 1
                    """,
                    (content,),
                )
                existing = cur.fetchone()
                if existing:
                    return str(existing["id"])

                cur.execute(
                    """
                    INSERT INTO long_term_memories (
                        id, content, category, importance,
                        source_session, embedding
                    )
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        memory_id,
                        content,
                        category,
                        importance,
                        source_session,
                        embedding,
                    ),
                )
                return memory_id

    def search_long_term_memories(
        self,
        embedding: list[float],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """按向量相似度检索长期记忆，并记录访问次数。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, content, category, importance,
                           1.0 - (embedding <=> %s::vector) AS score
                    FROM long_term_memories
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (embedding, embedding, top_k),
                )
                rows = [dict(row) for row in cur.fetchall()]
                if rows:
                    cur.execute(
                        """
                        UPDATE long_term_memories
                        SET access_count = access_count + 1,
                            last_accessed = NOW()
                        WHERE id = ANY(%s)
                        """,
                        ([row["id"] for row in rows],),
                    )
                return rows

    def delete_long_term_memory(self, memory_id: str) -> bool:
        """删除指定长期记忆。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM long_term_memories WHERE id = %s",
                    (memory_id,),
                )
                return cur.rowcount > 0

    # ══════════════════════════════════════════════════════════════
    #  evaluation jobs
    # ══════════════════════════════════════════════════════════════

    def create_eval_job(
        self,
        job_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_jobs (id, status, params)
                    VALUES (%s, 'pending', %s::jsonb)
                    RETURNING *
                    """,
                    (job_id, json.dumps(params, ensure_ascii=False)),
                )
                return dict(cur.fetchone())

    def get_eval_job(self, job_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM eval_jobs WHERE id = %s", (job_id,))
                row = cur.fetchone()
        return dict(row) if row else None

    def list_eval_jobs(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM eval_jobs
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]

    def start_eval_job(self, job_id: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET status = 'running',
                        started_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s AND status = 'pending'
                    """,
                    (job_id,),
                )

    def update_eval_job_progress(
        self,
        job_id: str,
        *,
        progress: float,
        message: str,
        current_query: str | None,
        completed: int | None,
        total: int | None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET progress = %s,
                        progress_message = %s,
                        current_query = COALESCE(%s, current_query),
                        completed_queries = COALESCE(%s, completed_queries),
                        total_queries = COALESCE(%s, total_queries),
                        updated_at = NOW()
                    WHERE id = %s AND status = 'running'
                    """,
                    (
                        min(progress, 99.9),
                        message,
                        current_query,
                        completed,
                        total,
                        job_id,
                    ),
                )

    def complete_eval_job(
        self,
        job_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET status = 'done',
                        progress = 100,
                        result = %s::jsonb,
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (json.dumps(result, ensure_ascii=False), job_id),
                )

    def fail_eval_job(self, job_id: str, error: str) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET status = 'failed',
                        error = %s,
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE id = %s
                    """,
                    (error[:4000], job_id),
                )

    def interrupt_incomplete_eval_jobs(self) -> int:
        """应用启动时标记上次进程遗留的评估任务。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE eval_jobs
                    SET status = 'interrupted',
                        error = '服务重启导致评估任务中断',
                        finished_at = NOW(),
                        updated_at = NOW()
                    WHERE status IN ('pending', 'running')
                    """
                )
                return cur.rowcount

    def create_eval_baseline(
        self,
        *,
        baseline_id: str,
        name: str,
        scope: str,
        job_id: str,
        evaluation_signature: str,
        metrics: dict[str, Any],
        policy: dict[str, Any],
        activate: bool,
    ) -> dict[str, Any]:
        """创建不可变评估基线，并可原子切换当前激活基线。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"eval-baseline:{scope}",),
                )
                if activate:
                    cur.execute(
                        """
                        UPDATE eval_baselines
                        SET is_active = FALSE
                        WHERE scope = %s AND is_active
                        """,
                        (scope,),
                    )
                cur.execute(
                    """
                    INSERT INTO eval_baselines (
                        id, name, scope, job_id, evaluation_signature,
                        metrics, policy, is_active, activated_at
                    )
                    VALUES (
                        %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb,
                        %s, CASE WHEN %s THEN NOW() ELSE NULL END
                    )
                    RETURNING *
                    """,
                    (
                        baseline_id,
                        name,
                        scope,
                        job_id,
                        evaluation_signature,
                        json.dumps(metrics, ensure_ascii=False),
                        json.dumps(policy, ensure_ascii=False),
                        activate,
                        activate,
                    ),
                )
                return dict(cur.fetchone())

    def get_eval_baseline(self, baseline_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT * FROM eval_baselines WHERE id = %s",
                    (baseline_id,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def get_active_eval_baseline(self, scope: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM eval_baselines
                    WHERE scope = %s AND is_active
                    """,
                    (scope,),
                )
                row = cur.fetchone()
        return dict(row) if row else None

    def list_eval_baselines(
        self,
        scope: str | None,
        limit: int,
    ) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT *
                    FROM eval_baselines
                    WHERE %s IS NULL OR scope = %s
                    ORDER BY is_active DESC, created_at DESC
                    LIMIT %s
                    """,
                    (scope, scope, limit),
                )
                return [dict(row) for row in cur.fetchall()]

    def activate_eval_baseline(self, baseline_id: str) -> dict[str, Any] | None:
        """在同一事务内切换 scope 的唯一激活基线。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT scope FROM eval_baselines WHERE id = %s FOR UPDATE",
                    (baseline_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                cur.execute(
                    "SELECT pg_advisory_xact_lock(hashtext(%s))",
                    (f"eval-baseline:{row['scope']}",),
                )
                cur.execute(
                    """
                    UPDATE eval_baselines
                    SET is_active = FALSE
                    WHERE scope = %s AND is_active
                    """,
                    (row["scope"],),
                )
                cur.execute(
                    """
                    UPDATE eval_baselines
                    SET is_active = TRUE, activated_at = NOW()
                    WHERE id = %s
                    RETURNING *
                    """,
                    (baseline_id,),
                )
                return dict(cur.fetchone())

    def save_eval_gate_run(
        self,
        *,
        gate_id: str,
        baseline_id: str,
        current_job_id: str,
        status: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO eval_gate_runs (
                        id, baseline_id, current_job_id, status, result
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (baseline_id, current_job_id) DO UPDATE SET
                        status = EXCLUDED.status,
                        result = EXCLUDED.result,
                        created_at = NOW()
                    RETURNING *
                    """,
                    (
                        gate_id,
                        baseline_id,
                        current_job_id,
                        status,
                        json.dumps(result, ensure_ascii=False),
                    ),
                )
                return dict(cur.fetchone())

    def list_eval_gate_runs(self, limit: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT gate.*, baseline.name AS baseline_name,
                           baseline.scope AS baseline_scope
                    FROM eval_gate_runs AS gate
                    JOIN eval_baselines AS baseline ON baseline.id = gate.baseline_id
                    ORDER BY gate.created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [dict(row) for row in cur.fetchall()]

    # ══════════════════════════════════════════════════════════════
    #  traces
    # ══════════════════════════════════════════════════════════════

    def save_trace(self, trace_data: dict[str, Any]) -> None:
        """持久化完整 Trace，供 API 查询与离线分析。"""
        summary = trace_data.get("summary", {})
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO traces (
                        trace_id, session_id, query, data,
                        total_elapsed_ms, total_tokens, total_tool_calls
                    )
                    VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s)
                    ON CONFLICT (trace_id) DO UPDATE SET
                        session_id = EXCLUDED.session_id,
                        query = EXCLUDED.query,
                        data = EXCLUDED.data,
                        total_elapsed_ms = EXCLUDED.total_elapsed_ms,
                        total_tokens = EXCLUDED.total_tokens,
                        total_tool_calls = EXCLUDED.total_tool_calls
                    """,
                    (
                        trace_data["trace_id"],
                        trace_data.get("session_id", ""),
                        trace_data.get("query", ""),
                        json.dumps(trace_data, ensure_ascii=False),
                        trace_data.get("total_elapsed_ms", 0),
                        summary.get("total_tokens", 0),
                        summary.get("total_tool_calls", 0),
                    ),
                )

    def get_trace(self, trace_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data FROM traces WHERE trace_id = %s",
                    (trace_id,),
                )
                row = cur.fetchone()
        if row is None:
            return None
        data = row["data"]
        return json.loads(data) if isinstance(data, str) else dict(data)

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
                    "DELETE FROM knowledge_relations WHERE file_id = %s",
                    (file_id,),
                )
                cur.execute(
                    "DELETE FROM knowledge_entity_mentions WHERE file_id = %s",
                    (file_id,),
                )
                cur.execute(
                    """
                    DELETE FROM knowledge_entities e
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM knowledge_entity_mentions m
                        WHERE m.entity_id = e.id
                    )
                    """
                )
                cur.execute(
                    """
                    UPDATE knowledge_files
                    SET status = 'deleted'
                    WHERE id = %s AND status = 'active'
                    """,
                    (file_id,),
                )
                return cur.rowcount > 0

    # ══════════════════════════════════════════════════════════════
    #  pgvector 向量操作
    # ══════════════════════════════════════════════════════════════

    def upsert_chunks(self, chunks: list[dict]) -> None:
        """批量插入知识块，写入时 jieba 分词 → to_tsvector。"""
        import jieba
        with self._connect() as conn:
            with conn.cursor() as cur:
                for c in chunks:
                    content = c["content"]
                    segmented = " ".join(jieba.cut(content))
                    meta_json = json.dumps(c.get("chunk_metadata", {}), ensure_ascii=False)
                    cur.execute(
                        """INSERT INTO knowledge_chunks
                           (file_id, chunk_index, content, parent_content, embedding, chunk_metadata, fts_vector)
                           VALUES (%s, %s, %s, %s, %s, %s, to_tsvector('simple', %s))""",
                        (
                            c["file_id"], c["chunk_index"], content,
                            c.get("parent_content"), c["embedding"], meta_json,
                            segmented,
                        ),
                    )
        logger.debug(f"Upserted {len(chunks)} chunks")

    def get_chunks_by_file_id(
        self,
        file_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """按原始顺序读取文件分块，用于派生索引。"""
        query = """
            SELECT id, file_id, chunk_index, content, parent_content, chunk_metadata
            FROM knowledge_chunks
            WHERE file_id = %s
            ORDER BY chunk_index
        """
        params: list[Any] = [file_id]
        if limit is not None:
            query += " LIMIT %s"
            params.append(limit)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return [dict(row) for row in cur.fetchall()]

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
        """BM25 近似搜索：jieba 分词后走 PG tsvector/tsquery + GIN 倒排索引。

        写入时 jieba 分词已存入 fts_vector，查询时同样分词后用 tsquery 匹配。
        ts_rank 提供专业的 TF/IDF 排序，GIN 索引保证 O(logN) 而非全表扫描。
        """
        import jieba
        segmented = " ".join(jieba.cut(query_text))

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT id, file_id, chunk_index, content, parent_content, chunk_metadata,
                              ts_rank(fts_vector, plainto_tsquery('simple', %s)) AS score
                       FROM knowledge_chunks
                       WHERE fts_vector @@ plainto_tsquery('simple', %s)
                       ORDER BY score DESC
                       LIMIT %s""",
                    (segmented, segmented, limit),
                )
                return [dict(r) for r in cur.fetchall()]

    def delete_chunks_by_file_id(self, file_id: str) -> int:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge_chunks WHERE file_id = %s", (file_id,)
                )
                return cur.rowcount

    # ══════════════════════════════════════════════════════════════
    #  knowledge graph
    # ══════════════════════════════════════════════════════════════

    def claim_knowledge_graph_index(self, file_id: str) -> bool:
        """原子抢占一个待处理图谱索引任务。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE knowledge_files
                    SET graph_status = 'processing',
                        graph_error = '',
                        graph_updated_at = NOW()
                    WHERE id = %s
                      AND status = 'active'
                      AND graph_status = 'queued'
                    """,
                    (file_id,),
                )
                return cur.rowcount == 1

    def queue_knowledge_graph_index(self, file_id: str) -> bool:
        """将活跃文件重新加入图谱索引队列。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE knowledge_files
                    SET graph_status = 'queued',
                        graph_error = '',
                        graph_updated_at = NOW()
                    WHERE id = %s AND status = 'active'
                    """,
                    (file_id,),
                )
                return cur.rowcount == 1

    def reset_interrupted_knowledge_graph_indexes(self) -> int:
        """服务启动时恢复上次进程中断的图谱任务。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE knowledge_files
                    SET graph_status = 'queued',
                        graph_error = '服务重启，任务已重新排队',
                        graph_updated_at = NOW()
                    WHERE status = 'active' AND graph_status = 'processing'
                    """
                )
                return cur.rowcount

    def list_queued_knowledge_graph_files(self) -> list[str]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id
                    FROM knowledge_files
                    WHERE status = 'active' AND graph_status = 'queued'
                    ORDER BY created_at
                    """
                )
                return [str(row["id"]) for row in cur.fetchall()]

    def update_knowledge_graph_status(
        self,
        file_id: str,
        status: str,
        error: str = "",
    ) -> None:
        if status not in {"disabled", "queued", "processing", "ready", "failed"}:
            raise ValueError(f"非法知识图谱状态: {status}")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE knowledge_files
                    SET graph_status = %s,
                        graph_error = %s,
                        graph_updated_at = NOW()
                    WHERE id = %s
                    """,
                    (status, error[:2000], file_id),
                )

    def replace_knowledge_graph(
        self,
        file_id: str,
        records: list[dict[str, Any]],
    ) -> dict[str, int]:
        """在单个事务内替换指定文件的实体提及和关系。"""
        entity_ids: set[str] = set()
        relation_count = 0

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM knowledge_relations WHERE file_id = %s",
                    (file_id,),
                )
                cur.execute(
                    "DELETE FROM knowledge_entity_mentions WHERE file_id = %s",
                    (file_id,),
                )

                for record in records:
                    chunk_id = record["chunk_id"]
                    context = str(record.get("context", ""))[:1000]
                    chunk_entities: dict[str, str] = {}

                    for entity in record.get("entities", []):
                        normalized_name = entity["normalized_name"]
                        entity_type = entity["entity_type"]
                        cur.execute(
                            """
                            INSERT INTO knowledge_entities (
                                name, normalized_name, entity_type, description
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (normalized_name, entity_type) DO UPDATE SET
                                name = EXCLUDED.name,
                                description = CASE
                                    WHEN EXCLUDED.description <> ''
                                        THEN EXCLUDED.description
                                    ELSE knowledge_entities.description
                                END,
                                updated_at = NOW()
                            RETURNING id
                            """,
                            (
                                entity["name"],
                                normalized_name,
                                entity_type,
                                entity.get("description", ""),
                            ),
                        )
                        entity_id = str(cur.fetchone()["id"])
                        entity_ids.add(entity_id)
                        chunk_entities.setdefault(normalized_name, entity_id)
                        cur.execute(
                            """
                            INSERT INTO knowledge_entity_mentions (
                                entity_id, file_id, chunk_id, context
                            )
                            VALUES (%s, %s, %s, %s)
                            ON CONFLICT (entity_id, file_id, chunk_id) DO UPDATE SET
                                context = EXCLUDED.context
                            """,
                            (entity_id, file_id, chunk_id, context),
                        )

                    for relation in record.get("relations", []):
                        source_id = chunk_entities.get(relation["normalized_source"])
                        target_id = chunk_entities.get(relation["normalized_target"])
                        if source_id is None or target_id is None:
                            continue
                        cur.execute(
                            """
                            INSERT INTO knowledge_relations (
                                source_entity_id, target_entity_id,
                                predicate, normalized_predicate,
                                file_id, chunk_id, evidence, confidence
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (
                                source_entity_id, target_entity_id,
                                normalized_predicate, file_id, chunk_id
                            ) DO UPDATE SET
                                predicate = EXCLUDED.predicate,
                                evidence = EXCLUDED.evidence,
                                confidence = EXCLUDED.confidence
                            """,
                            (
                                source_id,
                                target_id,
                                relation["predicate"],
                                relation["normalized_predicate"],
                                file_id,
                                chunk_id,
                                relation.get("evidence", "")[:1000],
                                relation.get("confidence", 1),
                            ),
                        )
                        relation_count += 1

                cur.execute(
                    """
                    DELETE FROM knowledge_entities e
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM knowledge_entity_mentions m
                        WHERE m.entity_id = e.id
                    )
                    """
                )

        return {
            "entities": len(entity_ids),
            "relations": relation_count,
        }

    def search_knowledge_graph(
        self,
        query: str,
        limit: int = 30,
    ) -> dict[str, list[dict[str, Any]]]:
        """按实体名和关系词检索相邻子图。"""
        import jieba

        stop_words = {
            "什么", "怎么", "如何", "哪些", "关系", "关联", "之间",
            "以及", "和", "与", "的", "是", "有",
        }
        terms = {
            token.strip().lower()
            for token in jieba.cut(query)
            if len(token.strip()) >= 2 and token.strip().lower() not in stop_words
        }
        if query.strip():
            terms.add(query.strip().lower())
        patterns = [f"%{term}%" for term in terms]
        if not patterns:
            return {"nodes": [], "edges": []}

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH matched_entities AS (
                        SELECT id
                        FROM knowledge_entities
                        WHERE normalized_name ILIKE ANY(%s)
                    )
                    SELECT
                        r.id,
                        s.id AS source_id,
                        s.name AS source,
                        s.entity_type AS source_type,
                        t.id AS target_id,
                        t.name AS target,
                        t.entity_type AS target_type,
                        r.predicate,
                        r.evidence,
                        r.confidence,
                        r.file_id,
                        f.filename
                    FROM knowledge_relations r
                    JOIN knowledge_entities s ON s.id = r.source_entity_id
                    JOIN knowledge_entities t ON t.id = r.target_entity_id
                    JOIN knowledge_files f ON f.id = r.file_id
                    WHERE f.status = 'active'
                      AND (
                          r.source_entity_id IN (SELECT id FROM matched_entities)
                          OR r.target_entity_id IN (SELECT id FROM matched_entities)
                          OR r.normalized_predicate ILIKE ANY(%s)
                      )
                    ORDER BY r.confidence DESC, r.created_at DESC
                    LIMIT %s
                    """,
                    (patterns, patterns, limit),
                )
                edges = [dict(row) for row in cur.fetchall()]

        nodes: dict[str, dict[str, Any]] = {}
        for edge in edges:
            nodes[str(edge["source_id"])] = {
                "id": str(edge["source_id"]),
                "name": edge["source"],
                "type": edge["source_type"],
            }
            nodes[str(edge["target_id"])] = {
                "id": str(edge["target_id"]),
                "name": edge["target"],
                "type": edge["target_type"],
            }
            edge["id"] = str(edge["id"])
            edge["source_id"] = str(edge["source_id"])
            edge["target_id"] = str(edge["target_id"])
        return {"nodes": list(nodes.values()), "edges": edges}

    def get_knowledge_graph_stats(self) -> dict[str, int]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        (SELECT COUNT(*) FROM knowledge_entities) AS entities,
                        (
                            SELECT COUNT(*)
                            FROM knowledge_relations r
                            JOIN knowledge_files f ON f.id = r.file_id
                            WHERE f.status = 'active'
                        ) AS relations,
                        (
                            SELECT COUNT(*)
                            FROM knowledge_files
                            WHERE status = 'active' AND graph_status = 'ready'
                        ) AS indexed_files,
                        (
                            SELECT COUNT(*)
                            FROM knowledge_files
                            WHERE status = 'active'
                              AND graph_status IN ('queued', 'processing')
                        ) AS pending_files
                    """
                )
                return dict(cur.fetchone())


    # ══════════════════════════════════════════════════════════════
    #  feedback
    # ══════════════════════════════════════════════════════════════

    def add_feedback(
        self, session_id: str, rating: str, reason: str = "",
        query: str = "", message_id: str = "",
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO feedback (session_id, message_id, rating, reason, query)
                       VALUES (%s, %s, %s, %s, %s)""",
                    (session_id, message_id, rating, reason, query),
                )


# 模块级单例
_db: Database | None = None


def get_db(conn_string: str | None = None) -> Database:
    global _db
    if _db is None:
        _db = Database(conn_string)
    return _db
