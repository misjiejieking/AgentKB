"""评估基线与质量门禁。"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from agentkb.eval.testset import TestSet

MetricDirection = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricRule:
    """单项质量规则。"""

    metric: str
    direction: MetricDirection
    max_regression: float
    min_value: float | None = None
    max_value: float | None = None


@dataclass(frozen=True)
class GatePolicy:
    """质量门禁策略。"""

    rules: tuple[MetricRule, ...] = field(default_factory=lambda: (
        MetricRule("recall_at_k.5", "higher", 0.02),
        MetricRule("precision_at_k.5", "higher", 0.02),
        MetricRule("mrr", "higher", 0.02),
        MetricRule("ndcg_at_k.5", "higher", 0.02),
    ))

    def to_dict(self) -> dict[str, Any]:
        return {"rules": [asdict(rule) for rule in self.rules]}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> GatePolicy:
        if not data:
            return cls()
        rules = tuple(MetricRule(**rule) for rule in data.get("rules", []))
        if not rules:
            raise ValueError("质量门禁至少需要一条规则")
        return cls(rules=rules)


def build_evaluation_signature(
    testset: TestSet,
    k_values: list[int],
) -> str:
    """生成与顺序无关的测试集和指标配置指纹。"""
    items = sorted(
        (
            {
                "query": item.query.strip(),
                "relevant_chunk_ids": sorted(item.relevant_chunk_ids),
            }
            for item in testset.items
        ),
        key=lambda item: (item["query"], item["relevant_chunk_ids"]),
    )
    payload = {
        "items": items,
        "k_values": sorted(set(k_values)),
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def evaluate_quality_gate(
    baseline_metrics: dict[str, Any],
    current_metrics: dict[str, Any],
    policy: GatePolicy,
) -> dict[str, Any]:
    """按统一策略评估当前结果，缺失指标视为失败。"""
    checks = []
    for rule in policy.rules:
        baseline = _metric_value(baseline_metrics, rule.metric)
        current = _metric_value(current_metrics, rule.metric)
        if baseline is None or current is None:
            checks.append({
                "metric": rule.metric,
                "passed": False,
                "reason": "基线或当前结果缺少必需指标",
            })
            continue

        regression = (
            baseline - current
            if rule.direction == "higher"
            else current - baseline
        )
        reasons = []
        if regression > rule.max_regression:
            reasons.append(
                f"退化 {regression:.4f} 超过阈值 {rule.max_regression:.4f}"
            )
        if rule.min_value is not None and current < rule.min_value:
            reasons.append(f"当前值低于下限 {rule.min_value:.4f}")
        if rule.max_value is not None and current > rule.max_value:
            reasons.append(f"当前值高于上限 {rule.max_value:.4f}")

        checks.append({
            "metric": rule.metric,
            "direction": rule.direction,
            "baseline": baseline,
            "current": current,
            "regression": round(regression, 6),
            "max_regression": rule.max_regression,
            "passed": not reasons,
            "reason": "；".join(reasons),
        })

    failed = [check for check in checks if not check["passed"]]
    return {
        "passed": not failed,
        "checks": checks,
        "failed_count": len(failed),
    }


class QualityGateService:
    """持久化基线管理与门禁执行。"""

    def __init__(self, db=None) -> None:
        if db is None:
            from agentkb.storage.pg_database import get_db
            db = get_db()
        self.db = db

    def create_baseline(
        self,
        *,
        job_id: str,
        name: str,
        scope: str,
        policy: GatePolicy,
        activate: bool,
    ) -> dict[str, Any]:
        job = self._completed_job(job_id)
        result = job["result"]
        signature = result.get("config", {}).get("evaluation_signature")
        if not signature:
            raise ValueError("该评估任务缺少测试集指纹，请重新执行评估")
        return self.db.create_eval_baseline(
            baseline_id=uuid.uuid4().hex,
            name=name.strip(),
            scope=scope.strip(),
            job_id=job_id,
            evaluation_signature=signature,
            metrics=result["metrics"],
            policy=policy.to_dict(),
            activate=activate,
        )

    def run_gate(
        self,
        *,
        current_job_id: str,
        baseline_id: str | None = None,
        scope: str = "default",
    ) -> dict[str, Any]:
        job = self._completed_job(current_job_id)
        baseline = (
            self.db.get_eval_baseline(baseline_id)
            if baseline_id
            else self.db.get_active_eval_baseline(scope)
        )
        if not baseline:
            raise ValueError("没有可用的评估基线")

        result = job["result"]
        current_signature = result.get("config", {}).get("evaluation_signature")
        if current_signature != baseline["evaluation_signature"]:
            raise ValueError("当前评估与基线使用的测试集或 K 值不一致")

        gate_result = evaluate_quality_gate(
            baseline["metrics"],
            result["metrics"],
            GatePolicy.from_dict(baseline["policy"]),
        )
        return self.db.save_eval_gate_run(
            gate_id=uuid.uuid4().hex,
            baseline_id=baseline["id"],
            current_job_id=current_job_id,
            status="passed" if gate_result["passed"] else "failed",
            result=gate_result,
        )

    def _completed_job(self, job_id: str) -> dict[str, Any]:
        job = self.db.get_eval_job(job_id)
        if not job:
            raise ValueError(f"评估任务不存在: {job_id}")
        if job["status"] != "done" or not job.get("result"):
            raise ValueError("评估任务尚未成功完成")
        return job


def _metric_value(metrics: dict[str, Any], path: str) -> float | None:
    value: Any = metrics
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)
