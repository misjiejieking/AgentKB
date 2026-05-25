"""SpecialistAgent 基类——所有专业 Agent 的抽象基类。

每个 Specialist Agent 负责一个专业领域：
  - 声明自己擅长的意图（通过 intents 属性）
  - 实现 execute() 方法执行专业任务
  - 返回统一的 AgentResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class AgentResult:
    """Agent 执行结果——统一格式，方便 Supervisor 聚合。"""
    agent_name: str
    success: bool
    output: str = ""                         # 给用户看的最终输出
    intermediate_steps: list[dict] = field(default_factory=list)  # 中间步骤（调试用）
    data: dict[str, Any] = field(default_factory=dict)            # 结构化数据
    error: str = ""
    # 指标
    elapsed_ms: float = 0.0
    tokens_used: int = 0
    tool_calls_count: int = 0
    # Reflection
    self_critique: str = ""                  # 自检结果
    confidence: float = 1.0                  # 置信度 0~1

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "output": self.output[:2048],
            "data": self.data,
            "error": self.error,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tokens_used": self.tokens_used,
            "tool_calls_count": self.tool_calls_count,
            "self_critique": self.self_critique[:512],
            "confidence": round(self.confidence, 3),
        }


class SpecialistAgent(ABC):
    """专业 Agent 抽象基类。

    子类必须实现:
      - name: 唯一名称
      - description: 一句话描述
      - intents: 擅长处理的意图列表
      - execute(): 核心执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent 唯一名称，如 'content_creator'。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent 一句话描述，供 Supervisor 做路由决策。"""
        ...

    @property
    def intents(self) -> list[str]:
        """该 Agent 擅长处理的意图类型。"""
        return []

    @abstractmethod
    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        """执行专业任务。

        Args:
            task: 用户原始任务或 Supervisor 分解后的子任务描述
            context: 上下文（对话历史、知识库内容、其他 Agent 输出等）
            tools: 该 Agent 可用的工具列表，None 表示使用全部已注册工具
        """
        ...

    @property
    def llm(self):
        """获取此 Agent 的 LLM 实例。"""
        from agentkb.llm.factory import get_chat_model
        return get_chat_model(streaming=False)

    def log(self, message: str) -> None:
        logger.info(f"[{self.name}] {message}")
