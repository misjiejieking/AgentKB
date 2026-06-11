from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import HumanMessage

from agentkb.agents.base import AgentResult, SpecialistAgent
from agentkb.agents.graph import _agent_executor_node
from agentkb.agents.registry import AgentRegistry


class ParallelAgent(SpecialistAgent):
    def __init__(self) -> None:
        self.roots_started: list[str] = []
        self.roots_finished: list[str] = []
        self.both_started = asyncio.Event()
        self.dependent_context: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "parallel_agent"

    @property
    def description(self) -> str:
        return "并发测试 Agent"

    async def execute(self, task, context=None, tools=None):
        if task in {"root-1", "root-2"}:
            self.roots_started.append(task)
            if len(self.roots_started) == 2:
                self.both_started.set()
            await asyncio.wait_for(self.both_started.wait(), timeout=0.5)
            self.roots_finished.append(task)
        else:
            self.dependent_context = context or {}
            assert set(self.roots_finished) == {"root-1", "root-2"}

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=f"{task}-done",
        )


class FailingAgent(SpecialistAgent):
    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def name(self) -> str:
        return "failing_agent"

    @property
    def description(self) -> str:
        return "失败测试 Agent"

    async def execute(self, task, context=None, tools=None):
        self.calls.append(task)
        return AgentResult(
            agent_name=self.name,
            success=False,
            error="预期失败",
        )


async def test_executor_runs_ready_subtasks_in_parallel():
    AgentRegistry.reset()
    agent = ParallelAgent()
    AgentRegistry().register(agent)

    result = await _agent_executor_node({
        "messages": [HumanMessage(content="组合任务")],
        "subtasks": [
            {
                "id": 1,
                "description": "root-1",
                "assigned_agent": agent.name,
                "dependencies": [],
            },
            {
                "id": 2,
                "description": "root-2",
                "assigned_agent": agent.name,
                "dependencies": [],
            },
            {
                "id": 3,
                "description": "dependent",
                "assigned_agent": agent.name,
                "dependencies": [1, 2],
            },
        ],
    })

    assert [item["id"] for item in result["agent_results"]] == [1, 2, 3]
    assert all(item["success"] for item in result["agent_results"])
    assert set(agent.dependent_context["completed_subtasks"]) == {"1", "2"}


async def test_executor_blocks_subtask_after_dependency_failure():
    AgentRegistry.reset()
    agent = FailingAgent()
    AgentRegistry().register(agent)

    result = await _agent_executor_node({
        "messages": [HumanMessage(content="失败任务")],
        "subtasks": [
            {
                "id": 1,
                "description": "root",
                "assigned_agent": agent.name,
                "dependencies": [],
            },
            {
                "id": 2,
                "description": "dependent",
                "assigned_agent": agent.name,
                "dependencies": [1],
            },
        ],
    })

    assert agent.calls == ["root"]
    assert result["agent_results"][0]["success"] is False
    assert result["agent_results"][1]["success"] is False
    assert "依赖子任务执行失败" in result["agent_results"][1]["error"]
