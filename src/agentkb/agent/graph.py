"""LangGraph 图构建与 AgentGraph 流式封装。"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, END
from loguru import logger

from agentkb.agent.state import AgentState
from agentkb.agent.nodes import agent_node, tools_node
from agentkb.config.settings import Settings
from agentkb.utils.tracer import start_trace, get_active_trace, finish_trace, trace_span


def _should_continue(state: AgentState) -> str:
    """条件路由：最后一条 AI 消息有 tool_calls 则进入 tools，否则结束。"""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


async def build_graph(checkpointer_path: str = "data/checkpoints.db") -> StateGraph:
    """构建并编译 LangGraph 状态图（MemorySaver，避免序列化丢字段）。"""
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.graph import MessagesState

    workflow = StateGraph(MessagesState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=MemorySaver())


class AgentGraph:
    """封装已编译的 LangGraph 图，暴露异步流式 API。"""

    def __init__(self, graph: StateGraph | None = None) -> None:
        self._graph = graph

    @classmethod
    async def create(cls, checkpointer_path: str = "data/checkpoints.db") -> AgentGraph:
        """异步工厂方法——构建图并返回 AgentGraph 实例。"""
        graph = await build_graph(checkpointer_path)
        return cls(graph)

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        thread_id: str = "default",
        history: list | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式执行 Agent 并逐事件 yield。

        history: 可选的历史消息列表（LangChain 格式），替换模型后首次调用时传入，
                避免 checkpointer 隔离导致 LLM 丢失上下文。
        """
        if self._graph is None:
            self._graph = await build_graph()

        trace = start_trace(session_id=session_id, query=user_input)

        input_messages = list(history) if history else []
        input_messages.append(HumanMessage(content=user_input))
        input_state = {"session_id": session_id, "messages": input_messages}
        config = {"configurable": {"thread_id": thread_id}}

        has_tokens = False
        try:
            async for event in self._graph.astream_events(
                input_state, config, version="v2"
            ):
                kind = event.get("event", "")
                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if hasattr(chunk, "content") and chunk.content:
                        has_tokens = True
                        yield {"type": "token", "content": chunk.content, "trace_id": trace.trace_id}

                elif kind == "on_tool_start":
                    yield {
                        "type": "tool_start",
                        "name": event["name"],
                        "input": event["data"].get("input", {}),
                        "trace_id": trace.trace_id,
                    }

                elif kind == "on_tool_end":
                    output = event["data"].get("output", "")
                    elapsed = _get_event_elapsed(event)
                    trace.add_event(
                        f"tool:{event['name']}",
                        {"output": str(output)[:512]},
                        elapsed_ms=elapsed,
                    )
                    yield {
                        "type": "tool_end",
                        "name": event["name"],
                        "output": str(output)[:2048],
                        "trace_id": trace.trace_id,
                        "trace": trace.to_dict(),
                    }

            # 如果没有流式 token（ainvoke 模式），从最终 state 提取回复
            if not has_tokens:
                final_state = self._graph.get_state(config)
                if final_state and final_state.values:
                    final_msgs = final_state.values.get("messages", [])
                    for m in reversed(final_msgs):
                        if isinstance(m, AIMessage) and getattr(m, "content", ""):
                            yield {"type": "token", "content": str(m.content), "trace_id": trace.trace_id}
                            break

            trace.add_event("done", {"session_id": session_id})
            yield {"type": "done", "session_id": session_id, "trace_id": trace.trace_id, "trace": trace.to_dict()}

        except Exception as exc:
            logger.error(f"Agent 流式执行出错: {exc}", exc_info=True)
            trace.add_event("error", {"message": str(exc)})
            yield {
                "type": "error",
                "message": f"处理请求时出错：{exc}",
                "trace_id": trace.trace_id,
            }
        finally:
            finish_trace()


def _get_event_elapsed(event: dict) -> float:
    """从 LangGraph event metadata 中提取耗时。"""
    meta = event.get("metadata", {}) or {}
    return float(meta.get("elapsed_ms", 0))
