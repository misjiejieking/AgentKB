"""跨会话个人记忆 Agent。"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from agentkb.agents.base import AgentResult, SpecialistAgent
from agentkb.config.settings import Settings
from agentkb.memory.long_term import LongTermMemory


class MemoryAgent(SpecialistAgent):
    @property
    def name(self) -> str:
        return "memory_agent"

    @property
    def description(self) -> str:
        return "保存和检索用户明确授权的跨会话偏好、事实与经验"

    @property
    def intents(self) -> list[str]:
        return ["personal_memory", "memory_management"]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        started_at = time.perf_counter()
        cfg = Settings.load()
        if not cfg.memory_long_term_enabled:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error="长期记忆功能未启用",
            )

        context = context or {}
        memory = LongTermMemory()
        lowered = task.lower()

        if any(keyword in lowered for keyword in ("忘记", "删除记忆", "forget")):
            return AgentResult(
                agent_name=self.name,
                success=True,
                output="删除长期记忆属于不可逆操作，需要你明确确认后才能执行。",
                data={"requires_confirmation": True, "action": "delete_memory"},
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
            )

        if any(
            keyword in lowered
            for keyword in ("记住", "记得", "保存偏好", "remember")
        ):
            category = (
                "preference"
                if any(keyword in lowered for keyword in ("喜欢", "偏好", "习惯"))
                else "fact"
            )
            memory_id = await asyncio.to_thread(
                memory.save,
                task,
                category,
                max(0.8, cfg.memory_long_term_min_importance),
                str(context.get("session_id", "")),
            )
            return AgentResult(
                agent_name=self.name,
                success=True,
                output="已保存为跨会话记忆。",
                data={"memory_id": memory_id, "category": category},
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
            )

        memories = await asyncio.to_thread(memory.search, task, 5)
        if not memories:
            output = "没有找到与你的问题相关的已保存记忆。"
        else:
            output = "找到以下相关记忆：\n" + "\n".join(
                f"- {item['content']}" for item in memories
            )
        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            data={"memories": memories},
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
        )
