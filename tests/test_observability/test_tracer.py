from __future__ import annotations

import asyncio

from agentkb.observability.tracer import TraceManager


class CaptureExporter:
    name = "capture"

    def __init__(self) -> None:
        self.traces: list[dict] = []

    def export_trace(self, trace_data: dict) -> None:
        self.traces.append(trace_data)


def test_trace_manager_uses_valid_external_trace_id():
    manager = TraceManager()
    manager._exporters = [CaptureExporter()]

    with manager.start_trace(trace_id="ABCDEF0123456789") as trace:
        assert trace.trace_id == "abcdef0123456789"


async def test_trace_context_is_isolated_between_concurrent_tasks():
    manager = TraceManager()
    exporter = CaptureExporter()
    manager._exporters = [exporter]

    first_started = asyncio.Event()
    second_started = asyncio.Event()
    first_finished = asyncio.Event()

    async def first_request():
        with manager.start_trace(session_id="session-1") as trace:
            first_started.set()
            await second_started.wait()
            with manager.span("first-child"):
                assert manager.get_active_trace() is trace
                await asyncio.sleep(0)
            first_finished.set()
            return trace

    async def second_request():
        await first_started.wait()
        with manager.start_trace(session_id="session-2") as trace:
            second_started.set()
            await first_finished.wait()
            with manager.span("second-child"):
                assert manager.get_active_trace() is trace
            return trace

    first_trace, second_trace = await asyncio.gather(
        first_request(),
        second_request(),
    )

    assert [span.name for span in first_trace.spans] == ["request", "first-child"]
    assert [span.name for span in second_trace.spans] == ["request", "second-child"]
    assert manager.get_active_trace() is None
    assert {trace["session_id"] for trace in exporter.traces} == {
        "session-1",
        "session-2",
    }
