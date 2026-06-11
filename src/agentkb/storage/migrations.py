"""PostgreSQL 顺序迁移。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


MIGRATIONS = (
    Migration(
        version=1,
        name="durable_agent_runs",
        sql="""
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'knowledge_chunks'
          AND column_name = 'fts_vector'
          AND is_generated = 'ALWAYS'
    ) THEN
        ALTER TABLE knowledge_chunks ALTER COLUMN fts_vector DROP EXPRESSION;
    END IF;
END $$;

CREATE TABLE agent_runs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ai_message_id TEXT NOT NULL,
    mode TEXT NOT NULL CHECK(mode IN ('auto', 'simple')),
    user_input TEXT NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('queued', 'running', 'completed', 'failed', 'cancelled', 'interrupted')
    ),
    error TEXT NOT NULL DEFAULT '',
    last_event_id BIGINT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX idx_agent_runs_active_session
    ON agent_runs(session_id)
    WHERE status IN ('queued', 'running');
CREATE INDEX idx_agent_runs_session_created
    ON agent_runs(session_id, created_at DESC);

CREATE TABLE run_events (
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    event_id BIGINT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, event_id)
);

CREATE INDEX idx_run_events_created
    ON run_events(run_id, created_at);
""",
    ),
    Migration(
        version=2,
        name="langgraph_checkpoints",
        sql="""
CREATE TABLE langgraph_checkpoints (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    parent_checkpoint_id TEXT,
    checkpoint_type TEXT NOT NULL,
    checkpoint_blob BYTEA NOT NULL,
    metadata_type TEXT NOT NULL,
    metadata_blob BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
);

CREATE INDEX idx_langgraph_checkpoints_latest
    ON langgraph_checkpoints(thread_id, checkpoint_ns, created_at DESC);

CREATE TABLE langgraph_checkpoint_writes (
    thread_id TEXT NOT NULL,
    checkpoint_ns TEXT NOT NULL DEFAULT '',
    checkpoint_id TEXT NOT NULL,
    task_id TEXT NOT NULL,
    task_path TEXT NOT NULL DEFAULT '',
    write_index INT NOT NULL,
    channel TEXT NOT NULL,
    value_type TEXT NOT NULL,
    value_blob BYTEA NOT NULL,
    PRIMARY KEY (
        thread_id, checkpoint_ns, checkpoint_id, task_id, write_index
    )
);

CREATE INDEX idx_langgraph_writes_checkpoint
    ON langgraph_checkpoint_writes(thread_id, checkpoint_ns, checkpoint_id);
""",
    ),
    Migration(
        version=3,
        name="stable_message_order",
        sql="""
ALTER TABLE messages
    ADD COLUMN sequence BIGSERIAL;

CREATE UNIQUE INDEX idx_messages_sequence
    ON messages(sequence);
CREATE INDEX idx_messages_session_sequence
    ON messages(session_id, sequence);
""",
    ),
    Migration(
        version=4,
        name="session_summaries",
        sql="""
CREATE TABLE session_summaries (
    session_id TEXT PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    summary TEXT NOT NULL,
    covered_sequence BIGINT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
""",
    ),
    Migration(
        version=5,
        name="persistent_eval_jobs",
        sql="""
CREATE TABLE eval_jobs (
    id TEXT PRIMARY KEY,
    status TEXT NOT NULL CHECK(
        status IN ('pending', 'running', 'done', 'failed', 'interrupted')
    ),
    progress REAL NOT NULL DEFAULT 0 CHECK(progress >= 0 AND progress <= 100),
    progress_message TEXT NOT NULL DEFAULT '',
    params JSONB NOT NULL,
    result JSONB,
    error TEXT NOT NULL DEFAULT '',
    current_query TEXT NOT NULL DEFAULT '',
    completed_queries INT NOT NULL DEFAULT 0,
    total_queries INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_eval_jobs_created
    ON eval_jobs(created_at DESC);
""",
    ),
    Migration(
        version=6,
        name="tool_approvals",
        sql="""
ALTER TABLE agent_runs
    DROP CONSTRAINT agent_runs_status_check;
ALTER TABLE agent_runs
    ADD CONSTRAINT agent_runs_status_check CHECK(
        status IN (
            'queued', 'running', 'waiting_approval', 'completed',
            'failed', 'cancelled', 'interrupted'
        )
    );

DROP INDEX idx_agent_runs_active_session;
CREATE UNIQUE INDEX idx_agent_runs_active_session
    ON agent_runs(session_id)
    WHERE status IN ('queued', 'running', 'waiting_approval');

CREATE TABLE tool_approvals (
    id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    thread_id TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    arguments JSONB NOT NULL,
    status TEXT NOT NULL CHECK(
        status IN ('pending', 'approved', 'rejected', 'completed', 'failed')
    ),
    result JSONB,
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    decided_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    UNIQUE(thread_id, tool_call_id)
);

CREATE INDEX idx_tool_approvals_run
    ON tool_approvals(run_id, created_at);
""",
    ),
    Migration(
        version=7,
        name="knowledge_graph",
        sql="""
ALTER TABLE knowledge_files
    ADD COLUMN graph_status TEXT NOT NULL DEFAULT 'queued'
        CHECK(graph_status IN ('disabled', 'queued', 'processing', 'ready', 'failed')),
    ADD COLUMN graph_error TEXT NOT NULL DEFAULT '',
    ADD COLUMN graph_updated_at TIMESTAMPTZ;

CREATE INDEX idx_knowledge_files_graph_status
    ON knowledge_files(graph_status)
    WHERE status = 'active';

CREATE TABLE knowledge_entities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(normalized_name, entity_type)
);

CREATE TABLE knowledge_entity_mentions (
    entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    file_id TEXT NOT NULL REFERENCES knowledge_files(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
    context TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(entity_id, file_id, chunk_id)
);

CREATE TABLE knowledge_relations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    target_entity_id UUID NOT NULL REFERENCES knowledge_entities(id) ON DELETE CASCADE,
    predicate TEXT NOT NULL,
    normalized_predicate TEXT NOT NULL,
    file_id TEXT NOT NULL REFERENCES knowledge_files(id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL REFERENCES knowledge_chunks(id) ON DELETE CASCADE,
    evidence TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1 CHECK(confidence >= 0 AND confidence <= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(
        source_entity_id, target_entity_id, normalized_predicate,
        file_id, chunk_id
    )
);

CREATE INDEX idx_knowledge_entities_name
    ON knowledge_entities(normalized_name);
CREATE INDEX idx_knowledge_mentions_file
    ON knowledge_entity_mentions(file_id);
CREATE INDEX idx_knowledge_relations_source
    ON knowledge_relations(source_entity_id);
CREATE INDEX idx_knowledge_relations_target
    ON knowledge_relations(target_entity_id);
CREATE INDEX idx_knowledge_relations_file
    ON knowledge_relations(file_id);
""",
    ),
    Migration(
        version=8,
        name="evaluation_quality_gates",
        sql="""
CREATE TABLE eval_baselines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scope TEXT NOT NULL,
    job_id TEXT NOT NULL REFERENCES eval_jobs(id) ON DELETE RESTRICT,
    evaluation_signature TEXT NOT NULL,
    metrics JSONB NOT NULL,
    policy JSONB NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    activated_at TIMESTAMPTZ,
    UNIQUE(scope, name)
);

CREATE UNIQUE INDEX idx_eval_baselines_active_scope
    ON eval_baselines(scope)
    WHERE is_active;
CREATE INDEX idx_eval_baselines_created
    ON eval_baselines(scope, created_at DESC);

CREATE TABLE eval_gate_runs (
    id TEXT PRIMARY KEY,
    baseline_id TEXT NOT NULL REFERENCES eval_baselines(id) ON DELETE RESTRICT,
    current_job_id TEXT NOT NULL REFERENCES eval_jobs(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK(status IN ('passed', 'failed')),
    result JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(baseline_id, current_job_id)
);

CREATE INDEX idx_eval_gate_runs_created
    ON eval_gate_runs(created_at DESC);
""",
    ),
    Migration(
        version=9,
        name="chat_image_attachments",
        sql="""
CREATE TABLE chat_attachments (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    message_id TEXT REFERENCES messages(id) ON DELETE CASCADE,
    original_name TEXT NOT NULL,
    filepath TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK(
        media_type IN ('image/jpeg', 'image/png', 'image/webp')
    ),
    file_size BIGINT NOT NULL CHECK(file_size > 0),
    status TEXT NOT NULL DEFAULT 'uploaded' CHECK(
        status IN ('uploaded', 'analyzed', 'failed')
    ),
    description TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analyzed_at TIMESTAMPTZ
);

CREATE INDEX idx_chat_attachments_session
    ON chat_attachments(session_id, created_at);
CREATE INDEX idx_chat_attachments_message
    ON chat_attachments(message_id)
    WHERE message_id IS NOT NULL;
""",
    ),
    Migration(
        version=10,
        name="custom_agents",
        sql="""
CREATE TABLE custom_agents (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE CHECK(name ~ '^[a-z][a-z0-9_]{2,39}$'),
    display_name TEXT NOT NULL,
    description TEXT NOT NULL,
    instructions TEXT NOT NULL,
    intents JSONB NOT NULL,
    allowed_tools JSONB NOT NULL,
    model_name TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(
        status IN ('active', 'disabled')
    ),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_custom_agents_status
    ON custom_agents(status, created_at);
""",
    ),
    Migration(
        version=11,
        name="mcp_servers_and_tools",
        sql="""
CREATE TABLE mcp_servers (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE CHECK(name ~ '^[a-z][a-z0-9_]{2,39}$'),
    transport TEXT NOT NULL CHECK(transport IN ('stdio', 'streamable_http')),
    command TEXT,
    args JSONB NOT NULL DEFAULT '[]',
    url TEXT,
    env JSONB NOT NULL DEFAULT '{}',
    headers JSONB NOT NULL DEFAULT '{}',
    confirmation_policy TEXT NOT NULL DEFAULT 'writes' CHECK(
        confirmation_policy IN ('always', 'writes', 'never')
    ),
    status TEXT NOT NULL DEFAULT 'disabled' CHECK(
        status IN ('enabled', 'disabled')
    ),
    connection_status TEXT NOT NULL DEFAULT 'disconnected' CHECK(
        connection_status IN ('disconnected', 'connecting', 'connected', 'error')
    ),
    last_error TEXT NOT NULL DEFAULT '',
    last_connected_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK(
        (transport = 'stdio' AND command IS NOT NULL AND url IS NULL)
        OR
        (transport = 'streamable_http' AND url IS NOT NULL AND command IS NULL)
    )
);

CREATE INDEX idx_mcp_servers_status
    ON mcp_servers(status, created_at);

CREATE TABLE mcp_tools (
    server_id TEXT NOT NULL REFERENCES mcp_servers(id) ON DELETE CASCADE,
    remote_name TEXT NOT NULL,
    local_name TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL DEFAULT '',
    input_schema JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    requires_confirmation BOOLEAN NOT NULL DEFAULT TRUE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(server_id, remote_name)
);

CREATE INDEX idx_mcp_tools_server
    ON mcp_tools(server_id, local_name);
""",
    ),
)


def apply_migrations(conn) -> None:
    """在事务内串行应用未执行的迁移。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INT PRIMARY KEY,
                name TEXT NOT NULL,
                checksum TEXT NOT NULL,
                applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (0x41474B42,))
        cur.execute("SELECT version, checksum FROM schema_migrations")
        applied = {row["version"]: row["checksum"] for row in cur.fetchall()}

        for migration in MIGRATIONS:
            checksum = applied.get(migration.version)
            if checksum:
                if checksum != migration.checksum:
                    raise RuntimeError(
                        f"数据库迁移 {migration.version} 校验值不一致"
                    )
                continue

            cur.execute(migration.sql)
            cur.execute(
                """
                INSERT INTO schema_migrations (version, name, checksum)
                VALUES (%s, %s, %s)
                """,
                (migration.version, migration.name, migration.checksum),
            )
