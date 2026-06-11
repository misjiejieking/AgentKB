from __future__ import annotations

from agentkb.observability.metrics import MetricsCollector


def test_metrics_keep_independent_label_series_and_summary_totals():
    metrics = MetricsCollector()

    metrics.record_tool_call("search", elapsed_ms=10, success=True)
    metrics.record_tool_call("search", elapsed_ms=20, success=False)
    metrics.record_tool_call("memory", elapsed_ms=5, success=True)

    text = metrics.to_prometheus_text()

    assert 'agentkb_tool_calls_total{tool="search"} 2' in text
    assert 'agentkb_tool_calls_total{tool="memory"} 1' in text
    assert 'agentkb_tool_call_errors_total{tool="search"} 1' in text
    assert 'agentkb_tool_call_duration_ms_count{tool="search"} 2' in text
    assert 'agentkb_tool_call_duration_ms_sum{tool="search"} 30' in text
