from __future__ import annotations

import pytest
from starlette.requests import Request

from agentkb.observability.middleware import ObservabilityMiddleware
from agentkb.observability.tracer import TraceManager


class CaptureExporter:
    name = "capture"

    def export_trace(self, trace_data):
        return None


async def test_middleware_does_not_execute_failed_request_twice(monkeypatch):
    manager = TraceManager()
    manager._exporters = [CaptureExporter()]
    monkeypatch.setattr(
        "agentkb.observability.tracer.get_tracer",
        lambda: manager,
    )

    request = Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/side-effect",
            "raw_path": b"/side-effect",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1234),
            "server": ("testserver", 80),
            "root_path": "",
        }
    )
    calls = 0

    async def failed_call_next(current_request):
        nonlocal calls
        calls += 1
        raise RuntimeError("expected")

    middleware = ObservabilityMiddleware(app=lambda scope, receive, send: None)
    with pytest.raises(RuntimeError, match="expected"):
        await middleware.dispatch(request, failed_call_next)

    assert calls == 1
