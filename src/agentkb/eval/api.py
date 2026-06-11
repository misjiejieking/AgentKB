"""评测 HTTP API——提交任务、查询进度、获取报告、对比评估。"""

from __future__ import annotations

import asyncio
import time
from contextvars import Context
from pathlib import Path

from fastapi import APIRouter, HTTPException
from loguru import logger
from psycopg2 import IntegrityError
from pydantic import BaseModel, Field

from agentkb.eval.jobs import get_job_manager, EvalJob, JobStatus
from agentkb.eval.evaluator import Evaluator
from agentkb.eval.testset import TestSet
from agentkb.eval.metrics import EvalResult, compute_metrics
from agentkb.eval.reporter import render_result_markdown
from agentkb.eval.quality_gate import (
    GatePolicy,
    QualityGateService,
    build_evaluation_signature,
)
from agentkb.config.settings import Settings

router = APIRouter(prefix="/eval", tags=["evaluation"])
_eval_tasks: set[asyncio.Task] = set()


# ══════════════════════════════════════════════════════════════
#  请求模型
# ══════════════════════════════════════════════════════════════

class EvalSubmitRequest(BaseModel):
    """提交评估任务的请求体。"""
    # 测试集：直接传 queries 或指定已有测试集路径
    queries: list[dict] | None = Field(default=None, description="单条或多条 query + relevant_chunk_ids")
    testset_path: str | None = Field(default=None, description="已有测试集 JSON 文件路径")
    # 模型配置
    model_name: str | None = Field(default=None, description="覆盖默认 LLM 模型")
    # 评估参数
    k_values: list[int] = Field(
        default_factory=lambda: [5, 10, 20],
        description="Recall@K 的 K 值列表",
    )
    skip_reranker: bool = Field(default=False, description="跳过 Reranker 精排")
    include_generation_eval: bool = Field(default=False, description="是否包含生成质量评估")
    sample_size: int | None = Field(default=None, description="限制评估样本数")
    # 标识
    prompt_version: str = Field(default="default", description="Prompt 版本标记")
    tags: list[str] = Field(default_factory=list, description="任务标签")
    gate_scope: str | None = Field(
        default=None,
        min_length=1,
        description="完成后自动执行指定 scope 的质量门禁",
    )


class EvalCompareRequest(BaseModel):
    baseline_job_id: str
    current_job_id: str


class EvalBaselineCreateRequest(BaseModel):
    job_id: str
    name: str = Field(min_length=1, max_length=100)
    scope: str = Field(default="default", min_length=1, max_length=100)
    policy: dict | None = None
    activate: bool = True


class EvalGateRequest(BaseModel):
    current_job_id: str
    baseline_id: str | None = None
    scope: str = Field(default="default", min_length=1, max_length=100)


# ══════════════════════════════════════════════════════════════
#  端点
# ══════════════════════════════════════════════════════════════

@router.post("/submit")
async def eval_submit(req: EvalSubmitRequest):
    """提交评估任务——异步执行，返回 job_id 用于查询进度。"""
    mgr = get_job_manager()

    job = await mgr.submit(req.model_dump())

    task = asyncio.create_task(
        _run_eval_job(job.job_id, req),
        context=Context(),
    )
    _eval_tasks.add(task)
    task.add_done_callback(_eval_tasks.discard)

    return {"job_id": job.job_id, "status": "pending"}


@router.get("/{job_id}/status")
async def eval_status(job_id: str):
    """查询评估任务进度——轮询此端点获取实时状态。"""
    mgr = get_job_manager()
    job = await mgr.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    return job.to_dict()


@router.get("/{job_id}/report")
async def eval_report(job_id: str, format: str = "json"):
    """获取评估报告——任务完成后返回完整报告。"""
    mgr = get_job_manager()
    job = await mgr.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"任务不存在: {job_id}")
    if job.status == JobStatus.PENDING:
        raise HTTPException(status_code=425, detail="任务尚未开始")
    if job.status == JobStatus.RUNNING:
        return {
            "status": "running",
            "progress": job.progress,
            "message": "任务仍在执行中，请等待完成后再获取报告",
        }
    if job.status in {JobStatus.FAILED, JobStatus.INTERRUPTED}:
        return {"status": job.status.value, "error": job.error}

    if format == "md":
        md = render_result_markdown(_dict_to_evalresult(job.result), title="评估报告")
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(md, media_type="text/markdown")

    return {
        "status": "done",
        "job_id": job_id,
        "metrics": job.result.get("metrics", {}) if job.result else {},
        "per_query": job.result.get("per_query", []) if job.result else [],
        "generation_eval": job.result.get("generation_eval") if job.result else None,
        "timing": {
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
        },
        "params": job.params,
    }


@router.get("/jobs")
async def eval_list_jobs(limit: int = 20):
    """列出历史评估任务。"""
    mgr = get_job_manager()
    return {"jobs": await mgr.list_jobs(limit)}


@router.post("/baselines")
async def eval_create_baseline(req: EvalBaselineCreateRequest):
    """从已完成评估任务创建不可变基线快照。"""
    try:
        baseline = await asyncio.to_thread(
            QualityGateService().create_baseline,
            job_id=req.job_id,
            name=req.name,
            scope=req.scope,
            policy=GatePolicy.from_dict(req.policy),
            activate=req.activate,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        raise HTTPException(
            status_code=409,
            detail="同一 scope 下已存在同名基线",
        ) from exc
    return {"baseline": baseline}


@router.get("/baselines")
async def eval_list_baselines(scope: str | None = None, limit: int = 50):
    """列出评估基线。"""
    from agentkb.storage.pg_database import get_db
    rows = await asyncio.to_thread(get_db().list_eval_baselines, scope, limit)
    return {"baselines": rows}


@router.post("/baselines/{baseline_id}/activate")
async def eval_activate_baseline(baseline_id: str):
    """切换指定 scope 的激活基线。"""
    from agentkb.storage.pg_database import get_db
    baseline = await asyncio.to_thread(
        get_db().activate_eval_baseline,
        baseline_id,
    )
    if not baseline:
        raise HTTPException(status_code=404, detail="评估基线不存在")
    return {"baseline": baseline}


@router.post("/gates")
async def eval_run_gate(req: EvalGateRequest):
    """对已完成任务执行质量门禁。"""
    try:
        gate = await asyncio.to_thread(
            QualityGateService().run_gate,
            current_job_id=req.current_job_id,
            baseline_id=req.baseline_id,
            scope=req.scope,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"gate": gate}


@router.get("/gates")
async def eval_list_gates(limit: int = 50):
    """列出最近的质量门禁执行记录。"""
    from agentkb.storage.pg_database import get_db
    rows = await asyncio.to_thread(get_db().list_eval_gate_runs, limit)
    return {"gates": rows}


@router.post("/compare")
async def eval_compare(req: EvalCompareRequest):
    """对比两次评估任务的结果。"""
    mgr = get_job_manager()
    baseline_job, current_job = await asyncio.gather(
        mgr.get(req.baseline_job_id),
        mgr.get(req.current_job_id),
    )

    if not baseline_job or not current_job:
        raise HTTPException(status_code=404, detail="任务不存在")
    if baseline_job.status != JobStatus.DONE or current_job.status != JobStatus.DONE:
        raise HTTPException(status_code=400, detail="两个任务都必须已完成")

    baseline = _dict_to_evalresult(baseline_job.result)
    current = _dict_to_evalresult(current_job.result)

    diff = Evaluator.diff(baseline, current, req.baseline_job_id, req.current_job_id)

    # 标记退化
    regressions = [d for d in diff.diffs if d.is_regression]
    improvements = [d for d in diff.diffs if d.is_improvement]

    return {
        "diff": diff.to_dict(),
        "summary": {
            "total_metrics": len(diff.diffs),
            "improvements": len(improvements),
            "regressions": len(regressions),
            "stable": len(diff.diffs) - len(improvements) - len(regressions),
            "has_regression": len(regressions) > 0,
            "regression_details": [
                {"metric": d.name, "delta": round(d.delta, 4)}
                for d in regressions
            ],
        },
    }


# ══════════════════════════════════════════════════════════════
#  后台任务执行
# ══════════════════════════════════════════════════════════════

async def _run_eval_job(job_id: str, params: EvalSubmitRequest) -> None:
    """在后台执行完整的评估流程。"""
    mgr = get_job_manager()
    cfg = Settings.load()

    async def _run(job: EvalJob) -> None:
        # 1. 加载或构建测试集
        if params.testset_path and Path(params.testset_path).exists():
            testset = TestSet.load(params.testset_path)
        elif params.queries:
            testset = TestSet.from_queries(params.queries)
        else:
            testset = TestSet.load(cfg.eval_testset_path)

        if params.sample_size and len(testset.items) > params.sample_size:
            testset.items = testset.items[:params.sample_size]

        total = len(testset.items)
        if total == 0:
            raise ValueError("评估测试集不能为空")
        await mgr.update_progress(
            job_id,
            5,
            f"测试集加载完成，共 {total} 条",
            total=total,
        )

        # 2. 预热组件
        from agentkb.storage.pg_database import get_db
        from agentkb.knowledge.embedder import get_embedder
        get_db()
        get_embedder()
        await mgr.update_progress(job_id, 10, "组件预热完成", total=total)

        # 3. 逐条执行检索评估
        from agentkb.knowledge.retriever import get_retriever
        from agentkb.knowledge.reranker import get_reranker

        retriever = get_retriever()
        reranker = None
        if not params.skip_reranker:
            try:
                reranker = get_reranker()
            except Exception as e:
                logger.warning(f"Reranker 不可用: {e}")

        queries = []
        relevant_ids_per_query = []
        retrieved_ids_per_query = []
        per_query_timing = []
        retrieval_elapsed_values: list[float] = []
        retrieval_failures = 0

        for idx, item in enumerate(testset.items):
            queries.append(item.query)
            relevant_ids_per_query.append(set(item.relevant_chunk_ids))

            t0 = time.time()
            try:
                candidates = retriever.retrieve(item.query)

                if candidates and reranker:
                    try:
                        ranked = reranker.rerank(item.query, candidates, top_k=cfg.retrieval_final_k)
                        ranked_ids_set = {r["id"] for r in ranked}
                        remaining = sorted(
                            [c for c in candidates if c["id"] not in ranked_ids_set],
                            key=lambda x: x.get("rrf_score", 0), reverse=True,
                        )
                        ret_ids = [r["id"] for r in ranked] + [c["id"] for c in remaining]
                    except Exception:
                        sorted_candidates = sorted(
                            candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)
                        ret_ids = [c["id"] for c in sorted_candidates]
                elif candidates:
                    sorted_candidates = sorted(
                        candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)
                    ret_ids = [c["id"] for c in sorted_candidates]
                else:
                    ret_ids = []

            except Exception as e:
                logger.error(f"检索失败 [{item.query}]: {e}")
                ret_ids = []
                retrieval_failures += 1

            elapsed = (time.time() - t0) * 1000
            retrieved_ids_per_query.append(ret_ids)
            per_query_timing.append({"query": item.query, "elapsed_ms": round(elapsed, 1), "candidate_count": len(ret_ids)})
            retrieval_elapsed_values.append(elapsed)

            # 更新进度
            progress = 10 + (idx + 1) / total * 80
            await mgr.update_progress(
                job_id, progress,
                message=f"评估中: {idx + 1}/{total}",
                current_query=item.query[:60],
                completed=idx + 1,
                total=total,
            )

        # 4. 计算指标
        await mgr.update_progress(
            job_id,
            92,
            "计算评估指标…",
            completed=total,
            total=total,
        )

        result = compute_metrics(
            queries=queries,
            relevant_ids_per_query=relevant_ids_per_query,
            retrieved_ids_per_query=retrieved_ids_per_query,
            k_values=params.k_values,
        )

        # 5. 可选的生成质量评估
        generation_eval = None
        if params.include_generation_eval:
            await mgr.update_progress(
                job_id,
                95,
                "执行生成质量评估…",
                completed=total,
                total=total,
            )
            generation_errors: list[str] = []
            try:
                from agentkb.eval.generation_eval import GenerationEval
                from agentkb.llm.factory import get_chat_model, get_chat_model_for

                llm = (
                    get_chat_model_for(params.model_name)
                    if params.model_name
                    else get_chat_model(streaming=False)
                )
                gen_eval = GenerationEval(llm_client=llm)
                eval_items = []
                for item in testset.items[:min(10, total)]:
                    try:
                        candidates = retriever.retrieve(item.query)
                        contexts = [
                            c.get("parent_content") or c.get("content", "")[:1024]
                            for c in (candidates or [])[:5]
                        ]
                        resp = await llm.ainvoke(f"基于上下文回答问题：\n\n上下文：\n{chr(10).join(contexts[:3])}\n\n问题：{item.query}\n\n答案：")
                        eval_items.append({
                            "query": item.query,
                            "answer": resp.content if hasattr(resp, "content") else str(resp),
                            "contexts": contexts,
                        })
                    except Exception as exc:
                        generation_errors.append(
                            f"{item.query[:60]}: {exc}"
                        )
                if eval_items:
                    gen_result = await gen_eval.evaluate_batch(eval_items)
                    generation_eval = {
                        "status": "completed",
                        **gen_result.to_dict(),
                        "failed_samples": len(generation_errors),
                        "errors": generation_errors[:20],
                    }
                else:
                    generation_eval = {
                        "status": "failed",
                        "error": "没有可评估的生成样本",
                        "errors": generation_errors[:20],
                    }
            except Exception as e:
                logger.error(f"生成评估失败: {e}")
                generation_eval = {
                    "status": "failed",
                    "error": str(e),
                    "errors": generation_errors[:20],
                }

        # 6. 组装结果
        await mgr.update_progress(
            job_id,
            99,
            "生成报告…",
            completed=total,
            total=total,
        )

        retrieval_hit_rate = sum(
            1 for q in result.per_query if q.recall_at_k.get(5, 0) > 0
        ) / len(result.per_query) if result.per_query else 0

        job.result = {
            "metrics": {
                "recall_at_k": {str(k): v for k, v in result.recall_at_k.items()},
                "precision_at_k": {str(k): v for k, v in result.precision_at_k.items()},
                "mrr": result.mrr,
                "ndcg_at_k": {str(k): v for k, v in result.ndcg_at_k.items()},
                # 扩展指标
                "retrieval_hit_rate_at_5": round(retrieval_hit_rate, 4),
                "retrieval_failures": retrieval_failures,
                "avg_latency_ms": round(
                    sum(retrieval_elapsed_values)
                    / len(retrieval_elapsed_values)
                    if retrieval_elapsed_values
                    else 0,
                    1,
                ),
                "total_queries": total,
            },
            "per_query": [
                {
                    "query": q.query,
                    "recall_at_5": round(q.recall_at_k.get(5, 0), 4),
                    "mrr": round(q.reciprocal_rank, 4),
                    "first_relevant_rank": q.first_relevant_rank,
                    "relevant_count": q.relevant_count,
                }
                for q in result.per_query
            ],
            "per_query_timing": per_query_timing,
            "generation_eval": generation_eval,
            "config": {
                "model_name": params.model_name or cfg.llm_generator_model_name,
                "prompt_version": params.prompt_version,
                "k_values": params.k_values,
                "skip_reranker": params.skip_reranker,
                "evaluation_signature": build_evaluation_signature(
                    testset,
                    params.k_values,
                ),
            },
        }

    await mgr.start(job_id, _run)
    if params.gate_scope:
        try:
            await asyncio.to_thread(
                QualityGateService().run_gate,
                current_job_id=job_id,
                scope=params.gate_scope,
            )
        except ValueError as exc:
            logger.error(f"评估任务 {job_id} 自动门禁失败: {exc}")


def _dict_to_evalresult(data: dict | None) -> EvalResult:
    """从 job.result dict 重建 EvalResult。"""
    from agentkb.eval.metrics import EvalResult as ER
    if not data or "metrics" not in data:
        return ER(k_values=[5, 10, 20])
    m = data["metrics"]
    recall = {int(k): v for k, v in m.get("recall_at_k", {}).items()}
    precision = {int(k): v for k, v in m.get("precision_at_k", {}).items()}
    ndcg = {int(k): v for k, v in m.get("ndcg_at_k", {}).items()}
    return ER(
        k_values=sorted(recall.keys()),
        recall_at_k=recall,
        precision_at_k=precision,
        mrr=m.get("mrr", 0),
        ndcg_at_k=ndcg,
    )
