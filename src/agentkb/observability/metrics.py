"""指标收集器——统计工具调用、LLM 调用、错误率等关键指标。

所有指标均可通过 PrometheusExporter 暴露到 /metrics 端点。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class MetricSnapshot:
    """单个指标的当前快照。"""
    name: str
    description: str
    metric_type: str  # "counter" | "gauge" | "histogram"
    value: float = 0.0
    labels: dict[str, str] = field(default_factory=dict)
    _samples: list[float] = field(default_factory=list)  # histogram 用

    def to_prometheus(self) -> str:
        """输出 Prometheus text format。"""
        safe_name = f"agentkb_{self.name.replace('-', '_').replace(' ', '_')}"
        label_str = ",".join(f'{k}="{v}"' for k, v in self.labels.items())
        tag = f"{{{label_str}}}" if label_str else ""
        return f"{safe_name}{tag} {self.value}"


class MetricsCollector:
    """线程安全的指标收集器——自动聚合工具调用、LLM 调用等关键指标。

    使用方式:
      metrics = get_metrics()
      metrics.record_tool_call("search_knowledge_base", elapsed_ms=45, success=True)
      metrics.record_llm_call(prompt_tokens=1024, completion_tokens=256, elapsed_ms=1200)
      print(metrics.snapshot())
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._metrics: dict[str, MetricSnapshot] = {}
        self._init_metrics()

    def _init_metrics(self) -> None:
        """初始化内置指标。"""
        defaults = [
            ("tool_calls_total", "总工具调用次数", "counter"),
            ("tool_calls_errors_total", "工具调用失败次数", "counter"),
            ("tool_call_duration_ms", "工具调用耗时", "histogram"),
            ("llm_calls_total", "总 LLM 调用次数", "counter"),
            ("llm_prompt_tokens_total", "LLM prompt tokens 总数", "counter"),
            ("llm_completion_tokens_total", "LLM completion tokens 总数", "counter"),
            ("llm_call_duration_ms", "LLM 调用耗时", "histogram"),
            ("request_duration_ms", "端到端请求耗时", "histogram"),
            ("retrieval_duration_ms", "检索耗时", "histogram"),
            ("cache_hits_total", "缓存命中次数", "counter"),
            ("cache_misses_total", "缓存未命中次数", "counter"),
        ]
        for name, desc, mtype in defaults:
            self._metrics[name] = MetricSnapshot(name=name, description=desc, metric_type=mtype)

    # ── 记录方法 ─────────────────────────────────────────────────

    def record_tool_call(self, tool_name: str, elapsed_ms: float, success: bool) -> None:
        with self._lock:
            self._incr("tool_calls_total", labels={"tool": tool_name})
            if not success:
                self._incr("tool_calls_errors_total", labels={"tool": tool_name})
            self._observe("tool_call_duration_ms", elapsed_ms, labels={"tool": tool_name})

    def record_llm_call(self, prompt_tokens: int, completion_tokens: int, elapsed_ms: float) -> None:
        with self._lock:
            self._incr("llm_calls_total")
            self._add("llm_prompt_tokens_total", prompt_tokens)
            self._add("llm_completion_tokens_total", completion_tokens)
            self._observe("llm_call_duration_ms", elapsed_ms)

    def record_retrieval(self, elapsed_ms: float, method: str = "hybrid") -> None:
        with self._lock:
            self._observe("retrieval_duration_ms", elapsed_ms, labels={"method": method})

    def record_cache(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self._incr("cache_hits_total")
            else:
                self._incr("cache_misses_total")

    def record_request(self, elapsed_ms: float) -> None:
        with self._lock:
            self._observe("request_duration_ms", elapsed_ms)

    # ── 内部操作 ─────────────────────────────────────────────────

    def _incr(self, name: str, value: float = 1.0, labels: dict | None = None) -> None:
        m = self._metrics.get(name)
        if m:
            m.value += value
            if labels:
                m.labels = labels

    def _add(self, name: str, value: float) -> None:
        m = self._metrics.get(name)
        if m:
            m.value += value

    def _observe(self, name: str, value: float, labels: dict | None = None) -> None:
        """Histogram: 记录样本值，更新均值。"""
        m = self._metrics.get(name)
        if m:
            m._samples.append(value)
            if len(m._samples) > 1000:
                m._samples = m._samples[-500:]
            m.value = sum(m._samples) / len(m._samples)  # 均值
            if labels:
                m.labels = labels

    # ── 快照与输出 ───────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """返回所有指标的当前快照。"""
        with self._lock:
            return {
                name: {
                    "value": m.value,
                    "type": m.metric_type,
                    "description": m.description,
                    "labels": m.labels,
                    "sample_count": len(m._samples),
                }
                for name, m in self._metrics.items()
            }

    def to_prometheus_text(self) -> str:
        """输出 Prometheus 文本格式。"""
        with self._lock:
            lines = []
            for name, m in self._metrics.items():
                safe = f"agentkb_{name.replace('-', '_').replace(' ', '_')}"
                lines.append(f"# HELP {safe} {m.description}")
                lines.append(f"# TYPE {safe} {m.metric_type}")
                label_str = ",".join(f'{k}="{v}"' for k, v in m.labels.items())
                tag = f"{{{label_str}}}" if label_str else ""
                lines.append(f"{safe}{tag} {m.value}")
            return "\n".join(lines) + "\n"


# 模块级单例
_metrics_collector: MetricsCollector | None = None


def get_metrics() -> MetricsCollector:
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
