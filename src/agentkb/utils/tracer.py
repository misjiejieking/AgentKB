"""轻量级链路追踪——JSON 格式输出到 data/traces/，不引入外部依赖。"""

from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class SpanEvent:
    name: str
    data: dict[str, Any]
    elapsed_ms: float = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TraceRecord:
    trace_id: str
    session_id: str = ""
    query: str = ""
    events: list[SpanEvent] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def add_event(self, name: str, data: dict[str, Any], elapsed_ms: float = 0) -> None:
        self.events.append(SpanEvent(name=name, data=data, elapsed_ms=elapsed_ms))

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "query": self.query,
            "elapsed_total_ms": round((time.time() - self.started_at) * 1000),
            "events": [
                {
                    "name": e.name,
                    "data": e.data,
                    "elapsed_ms": round(e.elapsed_ms, 2),
                    "timestamp": e.timestamp,
                }
                for e in self.events
            ],
        }

    def save(self, directory: str = "data/traces") -> None:
        path = Path(directory) / f"{self.trace_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, trace_id: str, directory: str = "data/traces") -> TraceRecord | None:
        path = Path(directory) / f"{trace_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        record = cls(
            trace_id=data["trace_id"],
            session_id=data.get("session_id", ""),
            query=data.get("query", ""),
            started_at=data.get("events", [{}])[0].get("timestamp", time.time()) if data.get("events") else time.time(),
        )
        for e in data.get("events", []):
            record.add_event(e["name"], e["data"], e.get("elapsed_ms", 0))
        return record


# 当前活跃的 trace（模块级，单请求单 trace）
_active_trace: TraceRecord | None = None


def start_trace(session_id: str = "", query: str = "") -> TraceRecord:
    global _active_trace
    trace_id = uuid.uuid4().hex[:12]
    _active_trace = TraceRecord(trace_id=trace_id, session_id=session_id, query=query)
    logger.debug(f"Trace 开始: {trace_id}")
    return _active_trace


def get_active_trace() -> TraceRecord | None:
    return _active_trace


def finish_trace() -> None:
    global _active_trace
    if _active_trace:
        _active_trace.save()
        logger.debug(f"Trace 结束: {_active_trace.trace_id}, "
                     f"events={len(_active_trace.events)}, "
                     f"elapsed={_active_trace.to_dict()['elapsed_total_ms']}ms")
        _active_trace = None


@contextmanager
def trace_span(name: str):
    """上下文管理器：记录一段操作的耗时和数据。"""
    start = time.time()
    data_holder = {}

    class SpanProxy:
        def set_data(self, **kwargs):
            data_holder.update(kwargs)

    proxy = SpanProxy()
    try:
        yield proxy
    finally:
        elapsed = (time.time() - start) * 1000
        if _active_trace:
            _active_trace.add_event(name, data_holder, elapsed)
