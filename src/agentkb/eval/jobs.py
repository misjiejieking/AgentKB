"""PostgreSQL 持久化评估任务管理器。"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable

from loguru import logger


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class EvalJob:
    """单个评估任务的持久化状态。"""

    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    progress_message: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: str = ""
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    current_query: str = ""
    completed_queries: int = 0
    total_queries: int = 0

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> EvalJob:
        return cls(
            job_id=str(row["id"]),
            status=JobStatus(row["status"]),
            progress=float(row["progress"]),
            progress_message=str(row["progress_message"]),
            params=dict(row["params"]),
            result=dict(row["result"]) if row["result"] else None,
            error=str(row["error"]),
            created_at=_format_timestamp(row["created_at"]),
            started_at=_format_timestamp(row["started_at"]),
            finished_at=_format_timestamp(row["finished_at"]),
            current_query=str(row["current_query"]),
            completed_queries=int(row["completed_queries"]),
            total_queries=int(row["total_queries"]),
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "job_id": self.job_id,
            "status": self.status.value,
            "progress": round(self.progress, 1),
            "progress_message": self.progress_message,
            "params": self.params,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "current_query": self.current_query,
            "completed_queries": self.completed_queries,
            "total_queries": self.total_queries,
        }
        if self.status == JobStatus.DONE and self.result:
            data["metrics_summary"] = {
                key: value
                for key, value in self.result.get("metrics", {}).items()
                if key in ("recall_at_k", "mrr", "ndcg_at_k")
            }
        return data


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    return value.isoformat() if hasattr(value, "isoformat") else str(value)


class EvalJobManager:
    """通过 PostgreSQL 管理任务状态，仅在进程内保留并发信号量。"""

    def __init__(self, db=None, max_concurrency: int = 3) -> None:
        self._db = db
        self._semaphore = asyncio.Semaphore(max_concurrency)

    @property
    def db(self):
        if self._db is None:
            from agentkb.storage.pg_database import get_db

            self._db = get_db()
        return self._db

    async def submit(self, params: dict[str, Any]) -> EvalJob:
        job_id = uuid.uuid4().hex[:12]
        row = await asyncio.to_thread(self.db.create_eval_job, job_id, params)
        logger.info(f"评估任务已提交: {job_id}")
        return EvalJob.from_row(row)

    async def start(
        self,
        job_id: str,
        run_fn: Callable[[EvalJob], Awaitable[None]],
    ) -> None:
        async with self._semaphore:
            await asyncio.to_thread(self.db.start_eval_job, job_id)
            job = await self.get(job_id)
            if job is None or job.status != JobStatus.RUNNING:
                return

            try:
                await run_fn(job)
                await asyncio.to_thread(
                    self.db.complete_eval_job,
                    job_id,
                    job.result or {},
                )
                logger.info(f"评估任务完成: {job_id}")
            except Exception as exc:
                await asyncio.to_thread(self.db.fail_eval_job, job_id, str(exc))
                logger.error(f"评估任务失败: {job_id}, error={exc}")

    async def get(self, job_id: str) -> EvalJob | None:
        row = await asyncio.to_thread(self.db.get_eval_job, job_id)
        return EvalJob.from_row(row) if row else None

    async def list_jobs(self, limit: int = 20) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(self.db.list_eval_jobs, limit)
        return [EvalJob.from_row(row).to_dict() for row in rows]

    async def update_progress(
        self,
        job_id: str,
        progress: float,
        message: str = "",
        current_query: str | None = None,
        completed: int | None = None,
        total: int | None = None,
    ) -> None:
        await asyncio.to_thread(
            self.db.update_eval_job_progress,
            job_id,
            progress=progress,
            message=message,
            current_query=current_query,
            completed=completed,
            total=total,
        )


_job_manager: EvalJobManager | None = None


def get_job_manager() -> EvalJobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = EvalJobManager()
    return _job_manager
