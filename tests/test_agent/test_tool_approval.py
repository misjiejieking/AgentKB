from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

import agentkb.agent.graph as graph_module
from agentkb.observability.tracer import TraceManager
from agentkb.tools.base import BaseTool, ToolResult
from agentkb.tools.registry import ToolRegistry


class RiskInput(BaseModel):
    value: int


class RiskTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "risk_tool"

    @property
    def description(self) -> str:
        return "审批测试工具"

    @property
    def args_schema(self) -> type[BaseModel]:
        return RiskInput

    @property
    def requires_confirmation(self) -> bool:
        return True

    async def _execute(self, value: int) -> ToolResult:
        self.calls += 1
        return ToolResult(self.name, True, data={"value": value})


class FakeApprovalDatabase:
    def __init__(self) -> None:
        self.created = []
        self.completed = []

    def create_tool_approval(self, **kwargs):
        self.created.append(kwargs)
        return {**kwargs, "status": "pending"}

    def complete_tool_approval(self, approval_id, **kwargs):
        self.completed.append((approval_id, kwargs))


class CaptureExporter:
    name = "capture"

    def export_trace(self, trace_data):
        return None


async def test_risk_tool_only_executes_after_graph_resume(monkeypatch):
    async def fake_agent(state, config):
        if isinstance(state["messages"][-1], HumanMessage):
            return {
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "id": "call-1",
                                "name": "risk_tool",
                                "args": {"value": 7},
                            }
                        ],
                    )
                ]
            }
        assert isinstance(state["messages"][-1], ToolMessage)
        return {"messages": [AIMessage(content="执行完成")]}

    db = FakeApprovalDatabase()
    tool = RiskTool()
    ToolRegistry.reset()
    ToolRegistry().register(tool)
    monkeypatch.setattr(graph_module, "agent_node", fake_agent)
    monkeypatch.setattr("agentkb.storage.pg_database.get_db", lambda: db)
    tracer = TraceManager()
    tracer._exporters = [CaptureExporter()]
    monkeypatch.setattr(graph_module, "get_tracer", lambda: tracer)

    graph = await graph_module.AgentGraph.create(InMemorySaver())
    first_events = [
        event
        async for event in graph.stream(
            "执行测试",
            session_id="session-1",
            run_id="run-1",
            thread_id="approval-thread",
        )
    ]

    approval = next(
        event for event in first_events if event["type"] == "approval_required"
    )
    assert tool.calls == 0
    assert db.created[0]["run_id"] == "run-1"

    resumed_events = [
        event
        async for event in graph.stream(
            "执行测试",
            session_id="session-1",
            run_id="run-1",
            thread_id="approval-thread",
            resume={
                "approved": True,
                "approval_id": approval["approval_id"],
            },
        )
    ]

    assert tool.calls == 1
    assert db.completed[0][1]["success"] is True
    assert any(
        event["type"] == "token" and event["content"] == "执行完成"
        for event in resumed_events
    )
    assert resumed_events[-1]["type"] == "done"
