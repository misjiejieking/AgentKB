"""FastAPI 应用工厂——挂载路由与静态文件。"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_app(graph) -> FastAPI:
    """创建 FastAPI 应用，注入 AgentGraph 并挂载前端静态文件。"""
    from agentkb.api.deps import init_graph

    init_graph(graph)

    app = FastAPI(title="AgentKB API", version="0.1.0")

    from agentkb.api.routes import router

    app.include_router(router, prefix="/api")

    static_dir = Path(__file__).resolve().parent.parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
