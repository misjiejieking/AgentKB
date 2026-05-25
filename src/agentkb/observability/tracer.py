"""核心 Trace/Span 管理——OpenTelemetry 兼容格式，支持层级 Span。

Span 层级示例:
  Request (trace_id=abc, span_id=1)
  ├── Intent Classification (span_id=2, parent=1)
  │   ├── fast_match (event)
  │   └── llm_classify (event)
  ├── Tool: search_knowledge_base (span_id=3, parent=1)
  │   ├── dense_search (event)  elapsed=45ms
  │   └── bm25_search (event)  elapsed=12ms
  └── LLM Generation (span_id=4, parent=1)
      ├── prompt_tokens: 1024
      └── completion_tokens: 256
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from loguru import logger


# ══════════════════════════════════════════════════════════════
#  数据结构
# ══════════════════════════════════════════════════════════════

@dataclass
class SpanEvent:
    """Span 内的事件（不带独立 Span 开销的轻量埋点）。"""
    name: str
    timestamp: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0


@dataclass
class TraceSpan:
    """单个 Span——代表一次有意义的操作。"""
    span_id: str
    parent_span_id: str = ""
    name: str = ""
    # 时间
    start_time: str = ""
    end_time: str = ""
    elapsed_ms: float = 0.0
    # 属性
    attributes: dict[str, Any] = field(default_factory=dict)
    # 子事件
    events: list[SpanEvent] = field(default_factory=list)
    # 状态
    status: str = "ok"  # "ok" | "error"
    status_message: str = ""

    def add_event(self, name: str, attributes: dict | None = None, elapsed_ms: float = 0.0) -> None:
        """添加轻量事件到当前 Span。"""
        self.events.append(SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {},
            elapsed_ms=elapsed_ms,
        ))

    def set_error(self, message: str) -> None:
        self.status = "error"
        self.status_message = message

    def to_dict(self) -> dict:
        d = {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "status": self.status,
            "attributes": self.attributes,
        }
        if self.events:
            d["events"] = [
                {
                    "name": e.name,
                    "elapsed_ms": round(e.elapsed_ms, 2),
                    "attributes": e.attributes,
                }
                for e in self.events
            ]
        if self.status_message:
            d["status_message"] = self.status_message
        return d


@dataclass
class Trace:
    """一次完整的用户请求 Trace——包含多个层级 Span。"""
    trace_id: str
    session_id: str = ""
    query: str = ""
    # 根 Span
    root_span_id: str = ""
    # 所有 Span（含根）
    spans: list[TraceSpan] = field(default_factory=list)
    # 元数据
    start_time: str = ""
    end_time: str = ""
    total_elapsed_ms: float = 0.0
    # 统计
    total_tokens: int = 0
    total_tool_calls: int = 0
    tool_call_errors: int = 0

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "root_span_id": self.root_span_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_elapsed_ms": round(self.total_elapsed_ms, 2),
            "summary": {
                "total_tokens": self.total_tokens,
                "total_tool_calls": self.total_tool_calls,
                "tool_call_errors": self.tool_call_errors,
                "span_count": len(self.spans),
            },
            "spans": [s.to_dict() for s in self.spans],
        }


# ══════════════════════════════════════════════════════════════
#  Trace 管理器
# ══════════════════════════════════════════════════════════════

class TraceManager:
    """全局 Trace 管理器——创建/管理 Trace 生命周期，支持多导出器。

    使用方式:
      tracer = get_tracer()
      with tracer.start_trace(session_id="s1", query="你好") as trace:
          with tracer.span("intent_classification") as span:
              span.add_event("fast_match", {"matched": True})
              # ... 业务逻辑 ...
          with tracer.span("llm_generation") as span:
              span.attributes["prompt_tokens"] = 100
          # trace 结束时自动导出到所有注册的导出器
    """

    def __init__(self) -> None:
        self._active_traces: dict[str, Trace] = {}
        self._span_stack: dict[str, list[str]] = {}  # trace_id → [current_span_id]
        self._exporters: list = []  # 延迟初始化

    def _ensure_exporters(self) -> None:
        if not self._exporters:
            from agentkb.observability.exporters import get_exporters
            self._exporters = get_exporters()

    @contextmanager
    def start_trace(
        self, session_id: str = "", query: str = ""
    ) -> Generator[Trace, None, None]:
        """创建一个新 Trace 作为上下文管理器。"""
        trace_id = uuid.uuid4().hex[:16]
        root_span_id = uuid.uuid4().hex[:8]

        trace = Trace(
            trace_id=trace_id,
            session_id=session_id,
            query=query,
            root_span_id=root_span_id,
            start_time=time.time(),
        )

        # 创建根 Span
        root_span = TraceSpan(
            span_id=root_span_id,
            name="request",
            start_time=time.time(),
            attributes={"session_id": session_id, "query": query[:200]},
        )
        trace.spans.append(root_span)
        self._active_traces[trace_id] = trace
        self._span_stack[trace_id] = [root_span_id]

        try:
            yield trace
        except Exception as exc:
            root_span.set_error(str(exc))
            raise
        finally:
            # 关闭根 Span
            root_span.end_time = time.time()
            root_span.elapsed_ms = (root_span.end_time - root_span.start_time) * 1000
            trace.end_time = time.time()
            trace.total_elapsed_ms = (trace.end_time - trace.start_time) * 1000

            # 导出
            self._ensure_exporters()
            trace_data = trace.to_dict()
            for exporter in self._exporters:
                try:
                    exporter.export_trace(trace_data)
                except Exception as e:
                    logger.warning(f"Trace 导出到 {exporter.name} 失败: {e}")

            self._active_traces.pop(trace_id, None)
            self._span_stack.pop(trace_id, None)

    @contextmanager
    def span(self, name: str, attributes: dict | None = None) -> Generator[TraceSpan, None, None]:
        """在当前活跃 Trace 中创建一个子 Span。"""
        # 找到当前活跃的 trace（使用最近创建的）
        trace_id = list(self._active_traces.keys())[-1] if self._active_traces else ""
        if not trace_id:
            # 无活跃 trace，创建一个孤立 span
            span = TraceSpan(span_id=uuid.uuid4().hex[:8], name=name, start_time=time.time())
            if attributes:
                span.attributes.update(attributes)
            try:
                yield span
            finally:
                span.end_time = time.time()
                span.elapsed_ms = (span.end_time - span.start_time) * 1000
            return

        trace = self._active_traces[trace_id]
        parent_id = self._span_stack.get(trace_id, [""])[-1] if self._span_stack.get(trace_id) else ""

        span = TraceSpan(
            span_id=uuid.uuid4().hex[:8],
            parent_span_id=parent_id,
            name=name,
            start_time=time.time(),
        )
        if attributes:
            span.attributes.update(attributes)

        trace.spans.append(span)
        self._span_stack.setdefault(trace_id, []).append(span.span_id)

        try:
            yield span
        except Exception as exc:
            span.set_error(str(exc))
            raise
        finally:
            span.end_time = time.time()
            span.elapsed_ms = (span.end_time - span.start_time) * 1000
            # 弹出栈
            stack = self._span_stack.get(trace_id, [])
            if stack and stack[-1] == span.span_id:
                stack.pop()

    def get_active_trace(self) -> Trace | None:
        """获取当前活跃的 Trace（最新的）。"""
        keys = list(self._active_traces.keys())
        return self._active_traces[keys[-1]] if keys else None

    def add_metric(self, name: str, value: float, tags: dict | None = None) -> None:
        """直接记录一个指标（会导出到 Prometheus 等）。"""
        self._ensure_exporters()
        for exporter in self._exporters:
            if hasattr(exporter, "record_metric"):
                try:
                    exporter.record_metric(name, value, tags or {})
                except Exception:
                    pass


# 模块级单例
_tracer: TraceManager | None = None


def get_tracer() -> TraceManager:
    global _tracer
    if _tracer is None:
        _tracer = TraceManager()
    return _tracer


# 便捷上下文管理器——在无活跃 trace 时自动创建
@contextmanager
def trace_context(session_id: str = "", query: str = "") -> Generator[Trace, None, None]:
    """自动创建 Trace 的便捷方法——用于顶层调用。"""
    tracer = get_tracer()
    with tracer.start_trace(session_id=session_id, query=query) as trace:
        yield trace
