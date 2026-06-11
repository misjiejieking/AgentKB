"""基于现有 psycopg2 连接池的 LangGraph PostgreSQL Checkpointer。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)


class PostgresCheckpointSaver(BaseCheckpointSaver[int]):
    """将完整 Checkpoint 与中间写入持久化到 PostgreSQL。"""

    def __init__(self, db) -> None:
        super().__init__()
        self._db = db

    @staticmethod
    def _identity(config: RunnableConfig) -> tuple[str, str, str | None]:
        configurable = config["configurable"]
        return (
            str(configurable["thread_id"]),
            str(configurable.get("checkpoint_ns", "")),
            get_checkpoint_id(config),
        )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns, checkpoint_id = self._identity(config)
        with self._db._connect() as conn:
            with conn.cursor() as cur:
                if checkpoint_id:
                    cur.execute(
                        """
                        SELECT *
                        FROM langgraph_checkpoints
                        WHERE thread_id = %s
                          AND checkpoint_ns = %s
                          AND checkpoint_id = %s
                        """,
                        (thread_id, checkpoint_ns, checkpoint_id),
                    )
                else:
                    cur.execute(
                        """
                        SELECT *
                        FROM langgraph_checkpoints
                        WHERE thread_id = %s AND checkpoint_ns = %s
                        ORDER BY created_at DESC, checkpoint_id DESC
                        LIMIT 1
                        """,
                        (thread_id, checkpoint_ns),
                    )
                row = cur.fetchone()
                if row is None:
                    return None

                cur.execute(
                    """
                    SELECT task_id, channel, value_type, value_blob
                    FROM langgraph_checkpoint_writes
                    WHERE thread_id = %s
                      AND checkpoint_ns = %s
                      AND checkpoint_id = %s
                    ORDER BY task_id, write_index
                    """,
                    (thread_id, checkpoint_ns, row["checkpoint_id"]),
                )
                writes = list(cur.fetchall())

        checkpoint = self.serde.loads_typed(
            (row["checkpoint_type"], bytes(row["checkpoint_blob"]))
        )
        metadata = self.serde.loads_typed(
            (row["metadata_type"], bytes(row["metadata_blob"]))
        )
        checkpoint_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": row["checkpoint_id"],
            }
        }
        parent_config: RunnableConfig | None = None
        if row["parent_checkpoint_id"]:
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": row["parent_checkpoint_id"],
                }
            }
        pending_writes = [
            (
                write["task_id"],
                write["channel"],
                self.serde.loads_typed(
                    (write["value_type"], bytes(write["value_blob"]))
                ),
            )
            for write in writes
        ]
        return CheckpointTuple(
            checkpoint_config,
            checkpoint,
            metadata,
            parent_config,
            pending_writes,
        )

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        clauses = []
        params: list[Any] = []
        if config:
            thread_id, checkpoint_ns, _ = self._identity(config)
            clauses.extend(["thread_id = %s", "checkpoint_ns = %s"])
            params.extend([thread_id, checkpoint_ns])
        if before and (before_id := get_checkpoint_id(before)):
            clauses.append("checkpoint_id < %s")
            params.append(before_id)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        database_limit = limit if filter is None else None
        limit_sql = "LIMIT %s" if database_limit is not None else ""
        if database_limit is not None:
            params.append(database_limit)

        with self._db._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT thread_id, checkpoint_ns, checkpoint_id
                    FROM langgraph_checkpoints
                    {where}
                    ORDER BY created_at DESC, checkpoint_id DESC
                    {limit_sql}
                    """,
                    params,
                )
                rows = list(cur.fetchall())

        yielded = 0
        for row in rows:
            checkpoint = self.get_tuple({
                "configurable": {
                    "thread_id": row["thread_id"],
                    "checkpoint_ns": row["checkpoint_ns"],
                    "checkpoint_id": row["checkpoint_id"],
                }
            })
            if checkpoint is None:
                continue
            if filter and any(
                checkpoint.metadata.get(key) != value
                for key, value in filter.items()
            ):
                continue
            yield checkpoint
            yielded += 1
            if limit is not None and yielded >= limit:
                break

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = self._identity(config)
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        metadata_type, metadata_blob = self.serde.dumps_typed(
            get_checkpoint_metadata(config, metadata)
        )
        checkpoint_id = checkpoint["id"]

        with self._db._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO langgraph_checkpoints (
                        thread_id, checkpoint_ns, checkpoint_id,
                        parent_checkpoint_id, checkpoint_type, checkpoint_blob,
                        metadata_type, metadata_blob
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (thread_id, checkpoint_ns, checkpoint_id)
                    DO UPDATE SET
                        parent_checkpoint_id = EXCLUDED.parent_checkpoint_id,
                        checkpoint_type = EXCLUDED.checkpoint_type,
                        checkpoint_blob = EXCLUDED.checkpoint_blob,
                        metadata_type = EXCLUDED.metadata_type,
                        metadata_blob = EXCLUDED.metadata_blob
                    """,
                    (
                        thread_id,
                        checkpoint_ns,
                        checkpoint_id,
                        parent_checkpoint_id,
                        checkpoint_type,
                        checkpoint_blob,
                        metadata_type,
                        metadata_blob,
                    ),
                )

        return {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        thread_id, checkpoint_ns, checkpoint_id = self._identity(config)
        if checkpoint_id is None:
            raise ValueError("写入 Checkpoint 前缺少 checkpoint_id")

        with self._db._connect() as conn:
            with conn.cursor() as cur:
                for index, (channel, value) in enumerate(writes):
                    write_index = WRITES_IDX_MAP.get(channel, index)
                    value_type, value_blob = self.serde.dumps_typed(value)
                    cur.execute(
                        """
                        INSERT INTO langgraph_checkpoint_writes (
                            thread_id, checkpoint_ns, checkpoint_id, task_id,
                            task_path, write_index, channel, value_type, value_blob
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (
                            thread_id, checkpoint_ns, checkpoint_id,
                            task_id, write_index
                        )
                        DO UPDATE SET
                            task_path = EXCLUDED.task_path,
                            channel = EXCLUDED.channel,
                            value_type = EXCLUDED.value_type,
                            value_blob = EXCLUDED.value_blob
                        """,
                        (
                            thread_id,
                            checkpoint_ns,
                            checkpoint_id,
                            task_id,
                            task_path,
                            write_index,
                            channel,
                            value_type,
                            value_blob,
                        ),
                    )

    def delete_thread(self, thread_id: str) -> None:
        with self._db._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM langgraph_checkpoint_writes WHERE thread_id = %s",
                    (thread_id,),
                )
                cur.execute(
                    "DELETE FROM langgraph_checkpoints WHERE thread_id = %s",
                    (thread_id,),
                )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        checkpoints = await asyncio.to_thread(
            lambda: list(
                self.list(
                    config,
                    filter=filter,
                    before=before,
                    limit=limit,
                )
            )
        )
        for checkpoint in checkpoints:
            yield checkpoint

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)
