"""AgentKB 应用入口——初始化所有服务并启动 FastAPI 服务器。"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from agentkb.config.settings import Settings
from agentkb.utils.logger import setup_logger
from agentkb.tools.registry import ToolRegistry
from agentkb.tools.knowledge_search import KnowledgeSearchTool
from agentkb.tools.web_search import WebSearchTool


def _ensure_data_dirs() -> None:
    """创建运行时所需的全部数据目录。"""
    dirs = ["data/uploads", "data/logs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _register_tools(cfg: Settings) -> None:
    """将 MVP 工具注册到 ToolRegistry。"""
    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool())
    registry.register(WebSearchTool(
        max_results=cfg.web_search_max_results,
        timeout=cfg.web_search_timeout,
    ))


def main() -> None:
    """启动 AgentKB。"""
    # 1. 加载配置
    cfg = Settings.load()

    # 2. 初始化日志
    from loguru import logger
    setup_logger(
        level=cfg.logging_level,
        log_file=cfg.logging_file,
        rotation=cfg.logging_rotation,
        retention=cfg.logging_retention,
        console=cfg.logging_console,
    )

    # 3. 创建数据目录
    _ensure_data_dirs()

    # 4. 初始化 PG 数据库（建表 + pgvector 扩展）
    try:
        from agentkb.storage.pg_database import get_db
        get_db()
    except Exception as e:
        logger.error(f"PostgreSQL 连接失败: {e}")
        logger.error("请确保 PostgreSQL 已启动，且 pgvector 扩展可用")
        raise

    # 5. 注册工具
    _register_tools(cfg)
    logger.info(f"已注册 {len(ToolRegistry().list_tools())} 个工具")

    # 6. 验证 LLM 连接
    try:
        from agentkb.llm.factory import create_llm
        provider = create_llm(cfg)
        provider.validate_connection()
    except Exception as e:
        logger.warning(f"LLM 连接检查失败: {e}")
        logger.warning("请确保 Ollama 已启动且模型已下载")

    # 7. 预热向量模型
    try:
        from agentkb.knowledge.embedder import get_embedder
        get_embedder(
            model_name=cfg.embedding_model_name,
            device=cfg.embedding_device,
            normalize=cfg.embedding_normalize,
            batch_size=cfg.embedding_batch_size,
        )
    except Exception as e:
        logger.warning(f"向量模型加载失败: {e}")

    # 8. 构建 LangGraph
    import asyncio
    from agentkb.agent.graph import AgentGraph
    agent_graph = asyncio.run(AgentGraph.create())

    # 10. 构建 FastAPI 应用
    from agentkb.api.server import create_app
    api_app = create_app(graph=agent_graph)

    # 11. 启动 uvicorn
    import uvicorn
    if cfg.app_auto_open_browser:
        webbrowser.open(f"http://{cfg.app_host}:{cfg.app_port}")

    logger.info(f"AgentKB 启动于 http://{cfg.app_host}:{cfg.app_port}")
    uvicorn.run(
        api_app,
        host=cfg.app_host,
        port=cfg.app_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
