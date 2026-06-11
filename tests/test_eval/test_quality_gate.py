from __future__ import annotations

from datetime import datetime

import pytest

from agentkb.eval.quality_gate import (
    GatePolicy,
    MetricRule,
    QualityGateService,
    build_evaluation_signature,
    evaluate_quality_gate,
)
from agentkb.eval.testset import TestItem as EvalTestItem
from agentkb.eval.testset import TestSet as EvaluationTestSet


def test_evaluation_signature_is_independent_of_item_order():
    first = EvaluationTestSet(items=[
        EvalTestItem("问题二", ["b", "a"]),
        EvalTestItem("问题一", ["c"]),
    ])
    second = EvaluationTestSet(items=[
        EvalTestItem("问题一", ["c"]),
        EvalTestItem("问题二", ["a", "b"]),
    ])

    assert build_evaluation_signature(first, [10, 5]) == build_evaluation_signature(
        second,
        [5, 10],
    )


def test_quality_gate_supports_higher_and_lower_metrics():
    policy = GatePolicy(rules=(
        MetricRule("mrr", "higher", 0.02),
        MetricRule("latency", "lower", 10, max_value=120),
    ))

    result = evaluate_quality_gate(
        {"mrr": 0.9, "latency": 100},
        {"mrr": 0.87, "latency": 121},
        policy,
    )

    assert result["passed"] is False
    assert result["failed_count"] == 2


def test_quality_gate_fails_when_required_metric_is_missing():
    policy = GatePolicy(rules=(MetricRule("mrr", "higher", 0.02),))

    result = evaluate_quality_gate({"mrr": 0.9}, {}, policy)

    assert result["passed"] is False
    assert result["checks"][0]["reason"] == "基线或当前结果缺少必需指标"


class FakeQualityGateDatabase:
    def __init__(self) -> None:
        now = datetime.now()
        self.jobs = {
            "baseline-job": {
                "id": "baseline-job",
                "status": "done",
                "result": {
                    "metrics": {"mrr": 0.9},
                    "config": {"evaluation_signature": "same"},
                },
            },
            "current-job": {
                "id": "current-job",
                "status": "done",
                "result": {
                    "metrics": {"mrr": 0.89},
                    "config": {"evaluation_signature": "same"},
                },
            },
        }
        self.baseline = None
        self.now = now

    def get_eval_job(self, job_id):
        return self.jobs.get(job_id)

    def create_eval_baseline(self, **values):
        self.baseline = {
            "id": values["baseline_id"],
            "created_at": self.now,
            "is_active": values["activate"],
            **values,
        }
        return self.baseline

    def get_active_eval_baseline(self, scope):
        if self.baseline and self.baseline["scope"] == scope:
            return self.baseline
        return None

    def get_eval_baseline(self, baseline_id):
        if self.baseline and self.baseline["id"] == baseline_id:
            return self.baseline
        return None

    def save_eval_gate_run(self, **values):
        return {"id": values["gate_id"], **values}


def test_quality_gate_service_creates_baseline_and_persists_result():
    db = FakeQualityGateDatabase()
    service = QualityGateService(db=db)
    policy = GatePolicy(rules=(MetricRule("mrr", "higher", 0.02),))

    baseline = service.create_baseline(
        job_id="baseline-job",
        name="release",
        scope="default",
        policy=policy,
        activate=True,
    )
    gate = service.run_gate(current_job_id="current-job")

    assert baseline["is_active"] is True
    assert gate["status"] == "passed"
    assert gate["result"]["passed"] is True


def test_quality_gate_service_rejects_incomparable_evaluations():
    db = FakeQualityGateDatabase()
    service = QualityGateService(db=db)
    service.create_baseline(
        job_id="baseline-job",
        name="release",
        scope="default",
        policy=GatePolicy(rules=(MetricRule("mrr", "higher", 0.02),)),
        activate=True,
    )
    db.jobs["current-job"]["result"]["config"]["evaluation_signature"] = "different"

    with pytest.raises(ValueError, match="测试集或 K 值不一致"):
        service.run_gate(current_job_id="current-job")
