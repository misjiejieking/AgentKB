from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agentkb.agent.graph as graph_module


async def test_single_agent_graph_preserves_custom_state(monkeypatch):
    async def fake_agent(state, config):
        return {
            "messages": [AIMessage(content="完成")],
            "tool_calls": [{"id": "call-1", "name": "fake", "args": {}}],
            "tool_results": [{"name": "fake", "success": True}],
        }

    monkeypatch.setattr(graph_module, "agent_node", fake_agent)
    graph = await graph_module.build_graph(InMemorySaver())

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "messages": [HumanMessage(content="执行任务")],
        },
        {"configurable": {"thread_id": "state-test"}},
    )

    assert result["session_id"] == "session-1"
    assert result["tool_calls"] == [
        {"id": "call-1", "name": "fake", "args": {}},
    ]
    assert result["tool_results"] == [{"name": "fake", "success": True}]
    assert [message.content for message in result["messages"]] == ["执行任务", "完成"]
