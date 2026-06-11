"""FastAPI 应用工厂——挂载路由与静态文件。"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles


def create_app(graph, multi_agent_graph=None) -> FastAPI:
    """创建 FastAPI 应用，注入 AgentGraph/MultiAgentGraph 并挂载前端静态文件。"""
    from agentkb.api.deps import init_graph, init_multi_agent_graph

    init_graph(graph)
    if multi_agent_graph is not None:
        init_multi_agent_graph(multi_agent_graph)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from agentkb.knowledge.graph import resume_knowledge_graph_indexing
        from agentkb.mcp_integration.manager import get_mcp_manager
        from agentkb.storage.pg_database import get_db

        await resume_knowledge_graph_indexing()
        stale_attachments = get_db().cleanup_stale_chat_attachments()
        for filepath in stale_attachments:
            Path(filepath).unlink(missing_ok=True)
        mcp_manager = get_mcp_manager()
        from agentkb.config.settings import Settings

        if Settings.load().mcp_enabled:
            await mcp_manager.start()
        try:
            yield
        finally:
            await mcp_manager.stop()

    app = FastAPI(
        title="AgentKB API",
        version="0.1.0",
        lifespan=lifespan,
    )

    from agentkb.api.routes import router

    app.include_router(router, prefix="/api")

    from agentkb.eval.api import router as eval_router

    app.include_router(eval_router, prefix="/api")

    from agentkb.agents.api import router as agents_router

    app.include_router(agents_router, prefix="/api")

    from agentkb.mcp_integration.api import router as mcp_router

    app.include_router(mcp_router, prefix="/api")

    static_dir = Path(__file__).resolve().parent.parent.parent.parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
