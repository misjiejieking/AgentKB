from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

import agentkb.agents.graph as graph_module


async def test_multi_agent_graph_preserves_complete_state(monkeypatch):
    async def fake_supervisor(state):
        reply = "状态已完整保留"
        return {
            "intent": "chat",
            "reasoning": "direct",
            "direct_reply": reply,
            "subtasks": [],
            "agent_results": [],
            "final_output": reply,
            "messages": [AIMessage(content=reply)],
        }

    monkeypatch.setattr(graph_module, "_supervisor_node", fake_supervisor)
    graph = await graph_module.build_multi_agent_graph(InMemorySaver())

    result = await graph.ainvoke(
        {
            "session_id": "session-1",
            "messages": [HumanMessage(content="你好")],
        },
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert result["final_output"] == "状态已完整保留"
    assert result["intent"] == "chat"
    assert result["subtasks"] == []
    assert result["agent_results"] == []
    assert [message.content for message in result["messages"]] == [
        "你好",
        "状态已完整保留",
    ]
