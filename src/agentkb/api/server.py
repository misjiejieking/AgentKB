"""FastAPI 应用工厂——挂载路由与静态文件。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_app(graph, multi_agent_graph=None) -> FastAPI:
    """创建 FastAPI 应用，注入 AgentGraph/MultiAgentGraph 并挂载前端静态文件。"""
    from agentkb.api.deps import init_graph, init_multi_agent_graph

    init_graph(graph)
    if multi_agent_graph is not None:
        init_multi_agent_graph(multi_agent_graph)

    app = FastAPI(title="AgentKB API", version="0.1.0")

    from agentkb.api.routes import router

    app.include_router(router, prefix="/api")

    # 注册评估 API 路由
    try:
        from agentkb.eval.api import router as eval_router
        app.include_router(eval_router, prefix="/api")
    except ImportError:
        pass

    static_dir = Path(__file__).resolve().parent.parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
