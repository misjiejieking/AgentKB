"""Agent 注册表——插件化注册和管理所有 Specialist Agent。

支持:
  - 按意图自动路由到匹配的 Agent
  - 动态注册/卸载 Agent
  - 多 Agent 投票机制
"""

from __future__ import annotations

from agentkb.agents.base import SpecialistAgent
from loguru import logger


class AgentRegistry:
    """Agent 注册表单例——管理所有 Specialist Agent 的生命周期。"""

    _instance: AgentRegistry | None = None
    _agents: dict[str, SpecialistAgent]

    def __new__(cls) -> AgentRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（测试用）。"""
        cls._instance = None

    def register(self, agent: SpecialistAgent) -> None:
        """注册一个 Specialist Agent。"""
        self._agents[agent.name] = agent
        logger.info(f"Agent 已注册: {agent.name} — {agent.description}")

    def unregister(self, name: str) -> None:
        """移除指定 Agent。"""
        self._agents.pop(name, None)

    def get(self, name: str) -> SpecialistAgent | None:
        return self._agents.get(name)

    def list_all(self) -> list[SpecialistAgent]:
        return list(self._agents.values())

    def find_by_intent(self, intent: str) -> list[SpecialistAgent]:
        """根据意图找到匹配的 Agent 列表。"""
        return [a for a in self._agents.values() if intent in a.intents]

    def find_best(self, intent: str) -> SpecialistAgent | None:
        """找到最匹配的 Agent（第一个声明支持该意图的）。"""
        candidates = self.find_by_intent(intent)
        return candidates[0] if candidates else None

    def get_agent_descriptions(self) -> str:
        """生成所有已注册 Agent 的描述文本（供 Supervisor prompt 使用）。"""
        lines = []
        for agent in self._agents.values():
            lines.append(
                f"- **{agent.name}** ({', '.join(agent.intents)}): {agent.description}"
            )
        return "\n".join(lines) if lines else "（无已注册的专业 Agent）"


# 模块级单例
def get_agent_registry() -> AgentRegistry:
    return AgentRegistry()
