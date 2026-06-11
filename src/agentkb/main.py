"""AgentKB 应用入口——初始化所有服务并启动 FastAPI 服务器。"""

from __future__ import annotations

import webbrowser
from pathlib import Path

from agentkb.config.settings import Settings
from agentkb.utils.logger import setup_logger


def _ensure_data_dirs() -> None:
    """创建运行时所需的全部数据目录。"""
    dirs = ["data/uploads", "data/logs", "data/traces", "data/eval", "data/eval/reports"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def _init_database() -> None:
    """初始化 PG 数据库（建表 + pgvector 扩展）。"""
    try:
        from agentkb.storage.pg_database import get_db
        db = get_db()
        interrupted = db.interrupt_incomplete_runs()
        if interrupted:
            from loguru import logger
            logger.warning(f"已终止 {interrupted} 个上次进程遗留的 Agent Run")
        interrupted_evals = db.interrupt_incomplete_eval_jobs()
        if interrupted_evals:
            from loguru import logger
            logger.warning(f"已终止 {interrupted_evals} 个上次进程遗留的评估任务")
    except Exception as e:
        from loguru import logger
        logger.error(f"PostgreSQL 连接失败: {e}")
        logger.error("请确保 PostgreSQL 已启动，且 pgvector 扩展可用")
        raise


def _register_tools(cfg: Settings) -> None:
    """注册所有工具到 ToolRegistry。"""
    from agentkb.tools.registry import ToolRegistry
    from agentkb.tools.knowledge_search import KnowledgeSearchTool
    from agentkb.tools.knowledge_graph import KnowledgeGraphQueryTool
    from agentkb.tools.web_search import WebSearchTool
    from agentkb.tools.code_executor import CodeExecutorTool
    from agentkb.tools.personal_memory import (
        SavePersonalMemoryTool,
        SearchPersonalMemoryTool,
    )
    from agentkb.tools.web_browser import WebBrowserTool

    registry = ToolRegistry()
    registry.register(KnowledgeSearchTool())
    if cfg.knowledge_graph_enabled:
        registry.register(KnowledgeGraphQueryTool())
    if cfg.web_search_enabled:
        registry.register(WebSearchTool(
            max_results=cfg.web_search_max_results,
            timeout=cfg.web_search_timeout,
        ))
    registry.register(CodeExecutorTool())
    registry.register(WebBrowserTool())
    registry.register(SearchPersonalMemoryTool())
    registry.register(SavePersonalMemoryTool())


def _register_agents() -> None:
    """注册所有 Specialist Agent 到 AgentRegistry。"""
    from agentkb.agents.registry import get_agent_registry
    from agentkb.agents.knowledge_agent import KnowledgeAgent
    from agentkb.agents.content_creator import ContentCreatorAgent
    from agentkb.agents.task_manager import TaskManagerAgent
    from agentkb.agents.learning_tutor import LearningTutorAgent
    from agentkb.agents.memory_agent import MemoryAgent
    from agentkb.agents.social_writer import SocialWriterAgent

    registry = get_agent_registry()
    registry.register(KnowledgeAgent())
    registry.register(ContentCreatorAgent())
    registry.register(TaskManagerAgent())
    registry.register(LearningTutorAgent())
    registry.register(SocialWriterAgent())
    registry.register(MemoryAgent())


def _init_llm(cfg: Settings) -> None:
    """验证 LLM 连接并预热模型。"""
    from loguru import logger
    try:
        from agentkb.llm.factory import create_llm
        provider = create_llm(cfg)
        provider.validate_connection()
    except Exception as e:
        logger.warning(f"LLM 连接检查失败: {e}")
        logger.warning(
            f"请检查 LLM Provider '{cfg.llm_provider}' 的地址、模型和密钥配置"
        )


def _init_embedder(cfg: Settings) -> None:
    """预热向量模型。"""
    from loguru import logger
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


def _init_observability(app, cfg: Settings) -> None:
    """注册可观测性中间件。"""
    from loguru import logger
    try:
        obs_cfg = cfg._val("observability", "enabled", default=True)
        if obs_cfg:
            from agentkb.observability.middleware import ObservabilityMiddleware
            app.add_middleware(ObservabilityMiddleware)
            logger.info("可观测性中间件已启用")
    except Exception as e:
        logger.warning(f"可观测性中间件初始化失败: {e}")


def main() -> None:
    """启动 AgentKB。"""
    from loguru import logger

    # 1. 加载配置
    cfg = Settings.load()

    # 2. 初始化日志
    setup_logger(
        level=cfg.logging_level,
        log_file=cfg.logging_file,
        rotation=cfg.logging_rotation,
        retention=cfg.logging_retention,
        console=cfg.logging_console,
    )

    # 3. 创建数据目录
    _ensure_data_dirs()

    # 4. 初始化数据库
    _init_database()

    # 5. 注册工具 + Agent
    _register_tools(cfg)
    _register_agents()
    from agentkb.agents.custom_service import CustomAgentService
    custom_agent_count = CustomAgentService().load_active()
    from agentkb.tools.registry import ToolRegistry
    from agentkb.agents.registry import get_agent_registry
    logger.info(f"已注册 {len(ToolRegistry().list_tools())} 个工具")
    logger.info(f"已注册 {len(get_agent_registry().list_all())} 个 Specialist Agent")
    if custom_agent_count:
        logger.info(f"已恢复 {custom_agent_count} 个自定义 Agent")

    # 6. 验证 LLM + 预热向量模型
    _init_llm(cfg)
    _init_embedder(cfg)

    # 7. 构建 LangGraph 单 Agent 图
    import asyncio
    from agentkb.agent.graph import AgentGraph
    from agentkb.storage.checkpointer import PostgresCheckpointSaver
    from agentkb.storage.pg_database import get_db

    checkpointer = PostgresCheckpointSaver(get_db())
    agent_graph = asyncio.run(AgentGraph.create(checkpointer))

    # 8. 构建 LangGraph Multi-Agent 图
    from agentkb.agents.graph import MultiAgentGraph
    multi_agent_graph = asyncio.run(MultiAgentGraph.create(checkpointer))

    # 9. 构建 FastAPI 应用
    from agentkb.api.server import create_app
    api_app = create_app(graph=agent_graph, multi_agent_graph=multi_agent_graph)

    # 10. 注册可观测性中间件
    _init_observability(api_app, cfg)

    # 11. 启动 uvicorn
    import uvicorn
    if cfg.app_auto_open_browser:
        webbrowser.open(f"http://{cfg.app_host}:{cfg.app_port}")

    logger.info(f"AgentKB V2 启动于 http://{cfg.app_host}:{cfg.app_port}")
    logger.info(
        "可用端点: /api/chat/stream | /api/agents | /api/mcp/servers "
        "| /api/eval/* | /api/metrics | /api/health"
    )
    uvicorn.run(
        api_app,
        host=cfg.app_host,
        port=cfg.app_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
