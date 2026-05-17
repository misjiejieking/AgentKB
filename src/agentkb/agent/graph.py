"""LangGraph 图构建与 AgentGraph 流式封装。"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, END
from loguru import logger

from agentkb.agent.state import AgentState
from agentkb.agent.nodes import agent_node, tools_node
from agentkb.config.settings import Settings


def _should_continue(state: AgentState) -> str:
    """条件路由：最后一条 AI 消息有 tool_calls 则进入 tools，否则结束。"""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


def build_graph(checkpointer_path: str = "data/checkpoints.db") -> StateGraph:
    """构建并编译 LangGraph 状态图。"""
    workflow = StateGraph(dict)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    Path(checkpointer_path).parent.mkdir(parents=True, exist_ok=True)
    checkpointer = SqliteSaver.from_conn_string(checkpointer_path)

    max_recursion = Settings.load().langgraph_max_recursion_limit
    return workflow.compile(checkpointer=checkpointer)


class AgentGraph:
    """封装已编译的 LangGraph 图，暴露异步流式 API。"""

    def __init__(self, graph: StateGraph | None = None) -> None:
        self._graph = graph or build_graph()

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        thread_id: str = "default",
    ) -> AsyncGenerator[dict[str, Any], None]:
        """
        流式执行 Agent 并逐事件 yield。

        Yields:
            {"type": "token", "content": str}     — LLM 流式 token
            {"type": "tool_start", "name": str, "input": dict}
            {"type": "tool_end", "name": str, "output": str}
            {"type": "done", "session_id": str}
            {"type": "error", "message": str}
        """
        input_state = {
            "session_id": session_id,
            "messages": [HumanMessage(content=user_input)],
        }
        config = {"configurable": {"thread_id": thread_id}}

        try:
            async for event in self._graph.astream_events(
                input_state, config, version="v2"
            ):
                kind = event.get("event", "")

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        yield {"type": "token", "content": chunk.content}

                elif kind == "on_tool_start":
                    yield {
                        "type": "tool_start",
                        "name": event["name"],
                        "input": event["data"].get("input", {}),
                    }

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    yield {
                        "type": "tool_end",
                        "name": event["name"],
                        "output": str(output)[:2048],
                    }

            yield {"type": "done", "session_id": session_id}

        except Exception as exc:
            logger.error(f"Agent 流式执行出错: {exc}", exc_info=True)
            yield {
                "type": "error",
                "message": f"处理请求时出错：{exc}",
            }
