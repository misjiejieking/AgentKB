"""评估异步任务管理器——提交、查询进度、获取报告。"""

from __future__ import annotations

import asyncio
import json
import uuid
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from agentkb.config.settings import Settings


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


@dataclass
class EvalJob:
    """单个评估任务的完整状态。"""
    job_id: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0          # 0~100
    progress_message: str = ""
    # 请求参数
    params: dict = field(default_factory=dict)
    # 结果
    result: dict | None = None
    error: str = ""
    # 时间戳
    created_at: str = ""
    started_at: str = ""
    finished_at: str = ""
    # 中间统计
    current_query: str = ""
    completed_queries: int = 0
    total_queries: int = 0

    def to_dict(self) -> dict:
        base = {
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
            base["metrics_summary"] = {
                k: v for k, v in self.result.get("metrics", {}).items()
                if k in ("recall_at_k", "mrr", "ndcg_at_k")
            }
        return base


class EvalJobManager:
    """评估任务管理器——内存存储 + 并发控制。"""

    def __init__(self) -> None:
        self._jobs: dict[str, EvalJob] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(3)  # 最大并发评估任务数

    async def submit(self, params: dict) -> EvalJob:
        """提交评估任务，返回 job 对象。"""
        job = EvalJob(
            job_id=uuid.uuid4().hex[:12],
            params=params,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        async with self._lock:
            self._jobs[job.job_id] = job
        logger.info(f"评估任务已提交: {job.job_id}, params={params}")
        return job

    async def start(self, job_id: str, run_fn: Callable) -> None:
        """异步启动评估任务。"""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.status = JobStatus.RUNNING
        job.started_at = datetime.now().isoformat(timespec="seconds")

        async with self._semaphore:
            try:
                await run_fn(job)
                job.status = JobStatus.DONE
                job.progress = 100
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                logger.info(f"评估任务完成: {job_id}")
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)
                job.finished_at = datetime.now().isoformat(timespec="seconds")
                logger.error(f"评估任务失败: {job_id}, error={exc}")

    def get(self, job_id: str) -> EvalJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 20) -> list[dict]:
        """返回最近的任务列表。"""
        sorted_jobs = sorted(
            self._jobs.values(),
            key=lambda j: j.created_at,
            reverse=True,
        )
        return [j.to_dict() for j in sorted_jobs[:limit]]

    def update_progress(
        self, job_id: str, progress: float, message: str = "",
        current_query: str = "", completed: int = 0, total: int = 0,
    ) -> None:
        """更新任务进度（从 run_fn 回调）。"""
        job = self._jobs.get(job_id)
        if not job:
            return
        job.progress = min(progress, 99.9)
        job.progress_message = message
        if current_query:
            job.current_query = current_query
        if completed:
            job.completed_queries = completed
        if total:
            job.total_queries = total


# 模块级单例
_job_manager: EvalJobManager | None = None


def get_job_manager() -> EvalJobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = EvalJobManager()
    return _job_manager
