from __future__ import annotations

from datetime import datetime

from agentkb.eval.jobs import EvalJobManager, JobStatus


class FakeEvalDatabase:
    def __init__(self) -> None:
        self.rows = {}

    def create_eval_job(self, job_id, params):
        row = {
            "id": job_id,
            "status": "pending",
            "progress": 0,
            "progress_message": "",
            "params": params,
            "result": None,
            "error": "",
            "current_query": "",
            "completed_queries": 0,
            "total_queries": 0,
            "created_at": datetime.now(),
            "started_at": None,
            "finished_at": None,
        }
        self.rows[job_id] = row
        return dict(row)

    def start_eval_job(self, job_id):
        self.rows[job_id]["status"] = "running"
        self.rows[job_id]["started_at"] = datetime.now()

    def get_eval_job(self, job_id):
        row = self.rows.get(job_id)
        return dict(row) if row else None

    def list_eval_jobs(self, limit):
        return [dict(row) for row in list(self.rows.values())[:limit]]

    def update_eval_job_progress(self, job_id, **values):
        row = self.rows[job_id]
        row["progress"] = values["progress"]
        row["progress_message"] = values["message"]

    def complete_eval_job(self, job_id, result):
        row = self.rows[job_id]
        row["status"] = "done"
        row["progress"] = 100
        row["result"] = result
        row["finished_at"] = datetime.now()

    def fail_eval_job(self, job_id, error):
        self.rows[job_id]["status"] = "failed"
        self.rows[job_id]["error"] = error


async def test_eval_job_manager_persists_completed_result():
    manager = EvalJobManager(db=FakeEvalDatabase(), max_concurrency=1)
    job = await manager.submit({"k_values": [5]})

    async def run(current_job):
        await manager.update_progress(current_job.job_id, 50, "running")
        current_job.result = {"metrics": {"mrr": 0.8}}

    await manager.start(job.job_id, run)
    completed = await manager.get(job.job_id)

    assert completed is not None
    assert completed.status == JobStatus.DONE
    assert completed.progress == 100
    assert completed.result == {"metrics": {"mrr": 0.8}}
