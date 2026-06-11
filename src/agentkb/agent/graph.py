"""LangGraph 图构建与 AgentGraph 流式封装。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from loguru import logger

from agentkb.agent.state import AgentState
from agentkb.agent.nodes import agent_node, tools_node
from agentkb.observability.metrics import get_metrics
from agentkb.observability.tracer import get_tracer


def _should_continue(state: AgentState) -> str:
    """条件路由：最后一条 AI 消息有 tool_calls 则进入 tools，否则结束。"""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        return "tools"
    return END


async def build_graph(checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """使用外部 Checkpointer 构建并编译 LangGraph 状态图。"""
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tools_node)

    workflow.set_entry_point("agent")

    workflow.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)


class AgentGraph:
    """封装已编译的 LangGraph 图，暴露异步流式 API。"""

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    @classmethod
    async def create(cls, checkpointer: BaseCheckpointSaver) -> AgentGraph:
        """异步工厂方法——构建图并返回 AgentGraph 实例。"""
        graph = await build_graph(checkpointer)
        return cls(graph)

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        run_id: str = "",
        thread_id: str = "default",
        history: list[AnyMessage] | None = None,
        conversation_summary: str = "",
        resume: dict[str, Any] | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式执行 Agent 并逐事件 yield。

        history: 可选的历史消息列表（LangChain 格式），替换模型后首次调用时传入，
                避免 checkpointer 隔离导致 LLM 丢失上下文。
        """
        tracer = get_tracer()
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        if resume is None:
            input_messages: list[AnyMessage] = []
            if history:
                snapshot = self._graph.get_state(config)
                if not snapshot.values.get("messages"):
                    input_messages.extend(history)
            input_messages.append(HumanMessage(content=user_input))
            graph_input: AgentState | Command = {
                "session_id": session_id,
                "run_id": run_id,
                "messages": input_messages,
                "conversation_summary": conversation_summary,
            }
        else:
            graph_input = Command(resume=resume)

        with tracer.start_trace(session_id=session_id, query=user_input) as trace:
            has_tokens = False
            try:
                async for event in self._graph.astream_events(
                    graph_input, config, version="v2"
                ):
                    kind = event.get("event", "")
                    if kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            has_tokens = True
                            yield {
                                "type": "token",
                                "content": chunk.content,
                                "trace_id": trace.trace_id,
                            }

                    elif kind == "on_chat_model_end":
                        prompt_tokens, completion_tokens = _get_token_usage(event)
                        elapsed = _get_event_elapsed(event)
                        model = str(
                            (event.get("metadata", {}) or {}).get(
                                "ls_model_name",
                                "",
                            )
                        )
                        trace.total_tokens += prompt_tokens + completion_tokens
                        get_metrics().record_llm_call(
                            prompt_tokens,
                            completion_tokens,
                            elapsed,
                            model,
                        )

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
                        trace.root_span.add_event(
                            f"tool:{event['name']}",
                            {"output": str(output)[:512]},
                            elapsed_ms=elapsed,
                        )
                        trace.total_tool_calls += 1
                        yield {
                            "type": "tool_end",
                            "name": event["name"],
                            "output": str(output)[:2048],
                            "trace_id": trace.trace_id,
                            "trace": trace.to_dict(),
                        }

                    elif kind == "on_chain_stream":
                        chunk = (event.get("data", {}) or {}).get("chunk", {})
                        interrupts = (
                            chunk.get("__interrupt__", ())
                            if isinstance(chunk, Mapping)
                            else ()
                        )
                        if interrupts:
                            approval = interrupts[0].value
                            trace.root_span.add_event(
                                "approval_required",
                                {"approval_id": approval.get("approval_id", "")},
                            )
                            yield {
                                **approval,
                                "approval_kind": approval.get("type", ""),
                                "type": "approval_required",
                                "trace_id": trace.trace_id,
                            }
                            return

                if not has_tokens:
                    final_state = self._graph.get_state(config)
                    if final_state and final_state.values:
                        final_msgs = final_state.values.get("messages", [])
                        for message in reversed(final_msgs):
                            content = message.content if isinstance(message, AIMessage) else ""
                            if isinstance(content, str) and content:
                                yield {
                                    "type": "token",
                                    "content": content,
                                    "trace_id": trace.trace_id,
                                }
                                break

                trace.root_span.add_event("done", {"session_id": session_id})
                yield {
                    "type": "done",
                    "session_id": session_id,
                    "trace_id": trace.trace_id,
                    "trace": trace.to_dict(),
                }

            except Exception as exc:
                logger.error(f"Agent 流式执行出错: {exc}", exc_info=True)
                trace.root_span.add_event("error", {"message": str(exc)})
                yield {
                    "type": "error",
                    "message": f"处理请求时出错：{exc}",
                    "trace_id": trace.trace_id,
                }


def _get_event_elapsed(event: Mapping[str, Any]) -> float:
    """从 LangGraph event metadata 中提取耗时。"""
    meta = event.get("metadata", {}) or {}
    return float(meta.get("elapsed_ms", 0))


def _get_token_usage(event: Mapping[str, Any]) -> tuple[int, int]:
    """兼容 LangChain 标准 usage_metadata 与 Provider token_usage。"""
    output = (event.get("data", {}) or {}).get("output")
    usage = getattr(output, "usage_metadata", None) or {}
    if usage:
        return (
            int(usage.get("input_tokens", 0)),
            int(usage.get("output_tokens", 0)),
        )

    response_metadata = getattr(output, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage", {})
    return (
        int(token_usage.get("prompt_tokens", 0)),
        int(token_usage.get("completion_tokens", 0)),
    )
