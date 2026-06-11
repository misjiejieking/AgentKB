"""FastAPI 中间件——自动为每个 HTTP 请求注入 trace_id 和开始/结束 Span。

使用方式:
  from agentkb.observability import ObservabilityMiddleware
  app.add_middleware(ObservabilityMiddleware)
"""

from __future__ import annotations

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from loguru import logger


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """为每个 HTTP 请求自动创建 Trace 上下文 + 注入 trace_id 到响应头。

    功能:
      1. 从 X-Trace-ID 请求头读取上游 trace_id，没有则生成
      2. 将 trace_id 写入 X-Trace-ID 响应头
      3. 记录请求耗时、状态码等基础 Span
      4. 绑定 loguru 的 trace_id 上下文（结构化日志关联）
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        from agentkb.observability.metrics import get_metrics
        from agentkb.observability.tracer import get_tracer

        started_at = time.perf_counter()
        tracer = get_tracer()
        status_code = 500
        with tracer.start_trace(
            session_id=request.headers.get("X-Session-ID", ""),
            query=f"{request.method} {request.url.path}",
            trace_id=request.headers.get("X-Trace-ID"),
        ) as trace:
            with logger.contextualize(trace_id=trace.trace_id):
                request.state.trace_id = trace.trace_id
                request.state.trace = trace
                try:
                    response: Response = await call_next(request)
                    status_code = response.status_code
                except Exception as exc:
                    trace.root_span.set_error(str(exc))
                    raise
                finally:
                    elapsed_ms = (time.perf_counter() - started_at) * 1000
                    route = request.scope.get("route")
                    metric_path = getattr(route, "path", request.url.path)
                    get_metrics().record_http_request(
                        method=request.method,
                        path=metric_path,
                        status_code=status_code,
                        elapsed_ms=elapsed_ms,
                    )

                response.headers["X-Trace-ID"] = trace.trace_id
                response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.1f}"
                return response
