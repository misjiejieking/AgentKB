"""线程安全的 Prometheus 指标聚合。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any, Literal

MetricKind = Literal["counter", "gauge", "summary"]
LabelKey = tuple[tuple[str, str], ...]
SeriesKey = tuple[str, LabelKey]


@dataclass(frozen=True)
class MetricDefinition:
    description: str
    kind: MetricKind


class MetricsCollector:
    """按标签维护 Counter、Gauge 和 Summary，输出标准文本格式。"""

    _definitions: dict[str, MetricDefinition] = {
        "http_requests_total": MetricDefinition("HTTP 请求总数", "counter"),
        "http_request_duration_ms": MetricDefinition("HTTP 请求耗时", "summary"),
        "tool_calls_total": MetricDefinition("工具调用总数", "counter"),
        "tool_call_errors_total": MetricDefinition("工具调用失败总数", "counter"),
        "tool_call_duration_ms": MetricDefinition("工具调用耗时", "summary"),
        "retrieval_calls_total": MetricDefinition("检索调用总数", "counter"),
        "retrieval_duration_ms": MetricDefinition("检索耗时", "summary"),
        "cache_hits_total": MetricDefinition("语义缓存命中总数", "counter"),
        "cache_misses_total": MetricDefinition("语义缓存未命中总数", "counter"),
        "agent_runs_total": MetricDefinition("Agent Run 总数", "counter"),
        "agent_run_duration_ms": MetricDefinition("Agent Run 耗时", "summary"),
        "agent_first_token_ms": MetricDefinition("Agent 首 Token 延迟", "summary"),
        "agent_run_events_total": MetricDefinition("Agent Run 事件总数", "counter"),
        "feedback_total": MetricDefinition("用户反馈总数", "counter"),
        "llm_calls_total": MetricDefinition("LLM 调用总数", "counter"),
        "llm_prompt_tokens_total": MetricDefinition("LLM 输入 Token 总数", "counter"),
        "llm_completion_tokens_total": MetricDefinition(
            "LLM 输出 Token 总数",
            "counter",
        ),
        "llm_call_duration_ms": MetricDefinition("LLM 调用耗时", "summary"),
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._values: dict[SeriesKey, float] = {}
        self._summary_count: dict[SeriesKey, int] = {}
        self._summary_sum: dict[SeriesKey, float] = {}

    def record_http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        elapsed_ms: float,
    ) -> None:
        labels = {
            "method": method,
            "path": path,
            "status": str(status_code),
        }
        with self._lock:
            self._increment("http_requests_total", labels=labels)
            self._observe(
                "http_request_duration_ms",
                elapsed_ms,
                labels={"method": method, "path": path},
            )

    def record_tool_call(
        self,
        tool_name: str,
        elapsed_ms: float,
        success: bool,
    ) -> None:
        labels = {"tool": tool_name}
        with self._lock:
            self._increment("tool_calls_total", labels=labels)
            if not success:
                self._increment("tool_call_errors_total", labels=labels)
            self._observe("tool_call_duration_ms", elapsed_ms, labels=labels)

    def record_retrieval(self, elapsed_ms: float, method: str = "hybrid") -> None:
        labels = {"method": method}
        with self._lock:
            self._increment("retrieval_calls_total", labels=labels)
            self._observe("retrieval_duration_ms", elapsed_ms, labels=labels)

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            self._increment("cache_hits_total" if hit else "cache_misses_total")

    def record_agent_run(
        self,
        *,
        mode: str,
        status: str,
        elapsed_ms: float,
        first_token_ms: float | None,
        event_count: int,
    ) -> None:
        labels = {"mode": mode, "status": status}
        with self._lock:
            self._increment("agent_runs_total", labels=labels)
            self._observe("agent_run_duration_ms", elapsed_ms, labels=labels)
            self._increment(
                "agent_run_events_total",
                value=event_count,
                labels={"mode": mode},
            )
            if first_token_ms is not None:
                self._observe(
                    "agent_first_token_ms",
                    first_token_ms,
                    labels={"mode": mode},
                )

    def record_llm_call(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        elapsed_ms: float,
        model: str = "",
    ) -> None:
        labels = {"model": model or "unknown"}
        with self._lock:
            self._increment("llm_calls_total", labels=labels)
            self._increment(
                "llm_prompt_tokens_total",
                value=prompt_tokens,
                labels=labels,
            )
            self._increment(
                "llm_completion_tokens_total",
                value=completion_tokens,
                labels=labels,
            )
            self._observe("llm_call_duration_ms", elapsed_ms, labels=labels)

    def record_feedback(self, rating: str) -> None:
        with self._lock:
            self._increment("feedback_total", labels={"rating": rating})

    def _increment(
        self,
        name: str,
        value: float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = (name, self._label_key(labels))
        self._values[key] = self._values.get(key, 0) + value

    def _observe(
        self,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        key = (name, self._label_key(labels))
        self._summary_count[key] = self._summary_count.get(key, 0) + 1
        self._summary_sum[key] = self._summary_sum.get(key, 0) + value

    @staticmethod
    def _label_key(labels: dict[str, str] | None) -> LabelKey:
        return tuple(sorted((labels or {}).items()))

    @staticmethod
    def _render_labels(labels: LabelKey) -> str:
        if not labels:
            return ""
        rendered = ",".join(
            f'{key}="{value.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
            for key, value in labels
        )
        return f"{{{rendered}}}"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "values": dict(self._values),
                "summary_count": dict(self._summary_count),
                "summary_sum": dict(self._summary_sum),
            }

    def to_prometheus_text(self) -> str:
        with self._lock:
            lines: list[str] = []
            for name, definition in self._definitions.items():
                metric_name = f"agentkb_{name}"
                lines.append(f"# HELP {metric_name} {definition.description}")
                lines.append(f"# TYPE {metric_name} {definition.kind}")

                if definition.kind == "summary":
                    keys = sorted(
                        key for key in self._summary_count if key[0] == name
                    )
                    for key in keys:
                        labels = self._render_labels(key[1])
                        lines.append(
                            f"{metric_name}_count{labels} "
                            f"{self._summary_count[key]}"
                        )
                        lines.append(
                            f"{metric_name}_sum{labels} "
                            f"{self._summary_sum[key]}"
                        )
                    continue

                keys = sorted(key for key in self._values if key[0] == name)
                for key in keys:
                    labels = self._render_labels(key[1])
                    lines.append(f"{metric_name}{labels} {self._values[key]}")

            return "\n".join(lines) + "\n"


_metrics_collector: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
