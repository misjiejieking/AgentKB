"""Multi-Agent 协作层——Supervisor + Specialist Agents 架构。"""

from agentkb.agents.base import SpecialistAgent, AgentResult
from agentkb.agents.supervisor import SupervisorAgent, TaskDecomposition
from agentkb.agents.registry import AgentRegistry, get_agent_registry
from agentkb.agents.orchestrator import MultiAgentOrchestrator
from agentkb.agents.reflection import ReflectionModule

__all__ = [
    "SpecialistAgent",
    "AgentResult",
    "SupervisorAgent",
    "TaskDecomposition",
    "AgentRegistry",
    "get_agent_registry",
    "MultiAgentOrchestrator",
    "ReflectionModule",
]
