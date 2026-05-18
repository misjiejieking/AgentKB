"""FastAPI 依赖注入——全局单例引用。"""

from __future__ import annotations

from agentkb.agent.graph import AgentGraph
from agentkb.config.settings import Settings
from agentkb.session.manager import SessionManager

_agent_graph: AgentGraph | None = None


def init_graph(graph: AgentGraph) -> None:
    """启动时注入已构建的 AgentGraph。"""
    global _agent_graph
    _agent_graph = graph


def get_settings() -> Settings:
    return Settings.load()


def get_graph() -> AgentGraph:
    assert _agent_graph is not None, "Graph not initialized"
    return _agent_graph


def get_session_mgr() -> SessionManager:
    return SessionManager()
