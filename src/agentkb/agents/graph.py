"""Multi-Agent LangGraph 图——将 Supervisor→Agent→聚合→Reflection 编码为 StateGraph。"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, END
from loguru import logger

from agentkb.config.settings import Settings
from agentkb.observability.tracer import get_tracer


# ══════════════════════════════════════════════════════════════
#  节点定义
# ══════════════════════════════════════════════════════════════


async def _supervisor_node(state: dict) -> dict:
    """Supervisor 节点——意图分析 + 任务分解 + Agent 路由。

    输入: state["messages"] 最后一条 HumanMessage 为当前 query
    输出: state["intent"], state["direct_reply"], state["subtasks"]
    """
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "chat", "direct_reply": "你好！有什么可以帮助你的吗？"}

    last_msg = messages[-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # 提取对话历史（当前消息之前的）
    history = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:256]}"
        for m in messages[:-1]  # 排除当前消息
        if hasattr(m, "content") and m.content
    ]

    from agentkb.agents.supervisor import SupervisorAgent
    from agentkb.llm.factory import get_chat_model

    llm = get_chat_model(streaming=False)
    supervisor = SupervisorAgent(llm_client=llm)
    decomposition = await supervisor.analyze(query, history)

    subtasks_dict = []
    for st in decomposition.subtasks:
        subtasks_dict.append({
            "id": st.id,
            "description": st.description,
            "assigned_agent": st.assigned_agent,
            "dependencies": st.dependencies,
        })

    result = {
        "intent": decomposition.intent,
        "reasoning": decomposition.reasoning,
        "direct_reply": decomposition.direct_reply or "",
        "subtasks": subtasks_dict,
        "current_subtask_index": 0,
        "agent_results": [],
        "final_output": "",
    }

    # 闲聊直接回复：将回复追加到 messages
    if decomposition.direct_reply:
        result["messages"] = [AIMessage(content=decomposition.direct_reply)]
        result["final_output"] = decomposition.direct_reply

    logger.info(
        f"Supervisor: intent={decomposition.intent}, "
        f"subtasks={len(subtasks_dict)}, direct={bool(decomposition.direct_reply)}"
    )
    return result


async def _agent_executor_node(state: dict) -> dict:
    """Agent 执行节点——执行当前 subtask，推结果到 agent_results。

    每次调用执行一个 subtask，执行完 index++。
    """
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return {"final_output": "所有子任务已完成"}

    st = subtasks[idx]
    agent_name = st.get("assigned_agent", "")
    task_desc = st.get("description", "")

    # 获取对话历史作为上下文
    messages = state.get("messages", [])
    history = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:256]}"
        for m in messages
        if hasattr(m, "content") and m.content
    ]

    # 附加已完成子任务的结果
    completed = {
        str(r.get("id", i)): {"output": r.get("output", "")}
        for i, r in enumerate(state.get("agent_results", []))
    }

    # 路由到 Specialist Agent
    from agentkb.agents.registry import get_agent_registry
    from agentkb.agents.base import AgentResult

    registry = get_agent_registry()
    agent = registry.get(agent_name)
    if agent is None:
        candidates = registry.find_by_intent(agent_name)
        agent = candidates[0] if candidates else None

    if agent is None:
        logger.warning(f"Agent '{agent_name}' 未注册")
        result = AgentResult(
            agent_name=agent_name,
            success=False,
            error=f"Agent '{agent_name}' 未注册",
        )
    else:
        context = {
            "original_query": task_desc,
            "history": history,
            "completed_subtasks": completed,
        }
        tracer = get_tracer()
        with tracer.span(f"agent:{agent.name}"):
            result = await agent.execute(task=task_desc, context=context)

    agent_results = list(state.get("agent_results", []))
    agent_results.append({
        "id": st.get("id"),
        "agent_name": agent_name,
        "success": result.success,
        "output": result.output,
        "data": result.data,
        "error": result.error,
        "elapsed_ms": result.elapsed_ms,
        "tokens_used": result.tokens_used,
    })

    logger.info(
        f"Agent '{agent_name}' subtask {idx + 1}/{len(subtasks)}: "
        f"success={result.success}, elapsed={result.elapsed_ms:.0f}ms"
    )

    return {
        "current_subtask_index": idx + 1,
        "agent_results": agent_results,
    }


async def _aggregator_node(state: dict) -> dict:
    """聚合节点——运行 Reflection 自检 + 生成最终回复。

    会添加一条 AIMessage 到 messages 供下次对话使用。
    """
    agent_results = state.get("agent_results", [])
    messages = state.get("messages", [])

    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            query = m.content[:200]
            break

    from agentkb.agents.reflection import ReflectionModule
    from agentkb.agents.supervisor import SupervisorAgent
    from agentkb.llm.factory import get_chat_model

    llm = get_chat_model(streaming=False)
    reflection = ReflectionModule()
    supervisor = SupervisorAgent(llm_client=llm)

    # Reflection 自检
    refined = await reflection.critique(
        query=query,
        agent_results=[
            type("R", (), {
                "output": r.get("output", ""),
                "error": r.get("error", ""),
                "to_dict": lambda self, r=r: r,
            })()
            for r in agent_results
        ],
        llm_client=llm,
    )

    if refined.get("needs_revision") and refined.get("revised_output"):
        logger.info("Reflection 触发修订")
        final_output = refined["revised_output"]
    else:
        # Agent 聚合
        final_output = await supervisor.aggregate(agent_results, query)

    logger.info(f"Aggregator: final_output length={len(final_output)}")

    return {
        "final_output": final_output,
        "messages": [AIMessage(content=final_output)],
    }


# ══════════════════════════════════════════════════════════════
#  条件路由
# ══════════════════════════════════════════════════════════════

def _route_after_supervisor(state: dict) -> str:
    """Supervisor 之后的分支：chat 直接结束，有 subtask 则进入 agent_executor。"""
    if state.get("direct_reply"):
        return END
    subtasks = state.get("subtasks", [])
    if subtasks:
        return "agent_executor"
    return END


def _route_after_executor(state: dict) -> str:
    """Agent 执行后的分支：还有 subtask 继续，否则进入聚合。"""
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    if subtasks and idx < len(subtasks):
        return "agent_executor"
    return "aggregator"


# ══════════════════════════════════════════════════════════════
#  图构建
# ══════════════════════════════════════════════════════════════

async def build_multi_agent_graph(
    checkpointer_path: str = "data/checkpoints_multi.db",
) -> StateGraph:
    """构建 Multi-Agent LangGraph 状态图。"""
    workflow = StateGraph(MessagesState)

    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("agent_executor", _agent_executor_node)
    workflow.add_node("aggregator", _aggregator_node)

    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"agent_executor": "agent_executor", END: END},
    )

    workflow.add_conditional_edges(
        "agent_executor",
        _route_after_executor,
        {"agent_executor": "agent_executor", "aggregator": "aggregator"},
    )

    workflow.add_edge("aggregator", END)

    from langgraph.checkpoint.memory import MemorySaver
    return workflow.compile(checkpointer=MemorySaver())


# ══════════════════════════════════════════════════════════════
#  MultiAgentGraph 流式封装
# ══════════════════════════════════════════════════════════════

class MultiAgentGraph:
    """封装已编译的 Multi-Agent LangGraph 图，暴露异步流式 API。"""

    def __init__(self, graph=None) -> None:
        self._graph = graph

    @classmethod
    async def create(cls, checkpointer_path: str = "data/checkpoints_multi.db") -> MultiAgentGraph:
        graph = await build_multi_agent_graph(checkpointer_path)
        return cls(graph)

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        thread_id: str = "default",
        history: list | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """流式执行多 Agent 图。

        astream_events v2 事件类型:
          - on_chat_model_stream → 推送 token
          - on_chain_start/end → non-LLM 节点边界（用于进度提示）
        """
        if self._graph is None:
            self._graph = await build_multi_agent_graph()

        from agentkb.observability.tracer import get_tracer
        tracer = get_tracer()

        input_messages = list(history) if history else []
        input_messages.append(HumanMessage(content=user_input))
        input_state = {"session_id": session_id, "messages": input_messages}
        config = {"configurable": {"thread_id": thread_id}}

        # 跟踪是否有 LLM 流式 token 输出过
        streamed_tokens = False

        try:
            with tracer.start_trace(session_id=session_id, query=user_input):
                async for event in self._graph.astream_events(
                    input_state, config, version="v2"
                ):
                    kind = event.get("event", "")

                    # ── 节点边界：推送工具状态 ──
                    if kind == "on_chain_start":
                        node_name = event.get("name", "")
                        if node_name in ("supervisor", "agent_executor", "aggregator"):
                            if node_name == "supervisor":
                                yield {
                                    "type": "tool_start",
                                    "name": "supervisor",
                                    "input": {"query": user_input},
                                }
                            elif node_name == "agent_executor":
                                yield {
                                    "type": "tool_start",
                                    "name": "subtask_executor",
                                    "input": {"step": "executing"},
                                }

                    if kind == "on_chain_end":
                        node_name = event.get("name", "")
                        if node_name == "supervisor":
                            output = event.get("data", {}).get("output", {})
                            yield {
                                "type": "tool_end",
                                "name": "supervisor",
                                "output": json.dumps({
                                    "intent": output.get("intent", ""),
                                    "subtasks_count": len(output.get("subtasks", [])),
                                    "direct_reply": output.get("direct_reply", ""),
                                }, ensure_ascii=False)[:2048],
                            }
                        elif node_name == "aggregator":
                            final = event.get("data", {}).get("output", {})
                            final_output = final.get("final_output", "")
                            # 如果 LLM 没有流式输出 token（结构化输出/非流式模式），
                            # 把最终结果作为 token 补发
                            if final_output and not streamed_tokens:
                                yield {
                                    "type": "token",
                                    "content": final_output,
                                }

                    # ── LLM token 流 ──
                    elif kind == "on_chat_model_stream":
                        chunk = event["data"]["chunk"]
                        if hasattr(chunk, "content") and chunk.content:
                            streamed_tokens = True
                            yield {
                                "type": "token",
                                "content": chunk.content,
                            }

                # ── 完成 ──
                yield {"type": "done", "session_id": session_id}

        except Exception as exc:
            logger.error(f"Multi-Agent 流式执行出错: {exc}", exc_info=True)
            yield {"type": "error", "message": f"处理请求时出错：{exc}"}
