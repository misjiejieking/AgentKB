"""FastAPI 中间件——自动为每个 HTTP 请求注入 trace_id 和开始/结束 Span。

使用方式:
  from agentkb.observability import ObservabilityMiddleware
  app.add_middleware(ObservabilityMiddleware)
"""

from __future__ import annotations

import time
import uuid

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
        # 提取或生成 trace_id
        trace_id = request.headers.get("X-Trace-ID", uuid.uuid4().hex[:16])
        request.state.trace_id = trace_id

        # 绑定到 loguru（通过 extra 字段，需配合 format 中的 {extra[trace_id]}）
        logger.configure(extra={"trace_id": trace_id})

        t0 = time.time()

        # 开始请求 Span
        try:
            from agentkb.observability.tracer import get_tracer
            tracer = get_tracer()
            with tracer.start_trace(
                session_id=request.headers.get("X-Session-ID", ""),
                query=f"{request.method} {request.url.path}",
            ) as trace:
                # 将 trace_id 存到 request state 供下游使用
                request.state.trace = trace

                response: Response = await call_next(request)

                # 记录请求级别耗时
                elapsed_ms = (time.time() - t0) * 1000
                trace.total_elapsed_ms = elapsed_ms

                # 注入响应头
                response.headers["X-Trace-ID"] = trace_id
                response.headers["X-Elapsed-Ms"] = f"{elapsed_ms:.1f}"

                return response

        except ImportError:
            # 如果观测模块不可用，静默跳过
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
        except Exception as exc:
            logger.warning(f"观测中间件异常: {exc}")
            response = await call_next(request)
            response.headers["X-Trace-ID"] = trace_id
            return response
