"""Multi-Agent LangGraph 图——将 Supervisor→Agent→聚合→Reflection 编码为 StateGraph。

图结构:
  START → supervisor_node (意图+分解)
           ├─ chat/direct_reply → END
           └─ subtasks → agent_executor_node (依赖分批并行)
                          → aggregator_node (Reflection+聚合) → END

流式策略:
  不用 astream_events 捕获节点内 LLM token（因为 Specialist Agent 内部创建
  的 LLM 实例与 LangGraph callback chain 不连通）。改用 ainvoke 拿到完整
  final_output 后分块推送模拟流式。
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from loguru import logger

from agentkb.config.settings import Settings
from agentkb.memory.context import select_conversation_context
from agentkb.observability.tracer import get_tracer


class MultiAgentState(MessagesState):
    """多 Agent 图在节点间持久化的完整状态。"""

    session_id: str
    intent: str
    reasoning: str
    direct_reply: str
    subtasks: list[dict[str, Any]]
    agent_results: list[dict[str, Any]]
    final_output: str
    conversation_summary: str


# ══════════════════════════════════════════════════════════════
#  节点定义
# ══════════════════════════════════════════════════════════════

async def _supervisor_node(state: MultiAgentState) -> dict[str, Any]:
    """Supervisor 节点——意图分析 + 任务分解 + Agent 路由。"""
    messages = select_conversation_context(
        list(state.get("messages", [])),
        state.get("conversation_summary", ""),
        Settings.load().memory_working_max_turns,
    )
    if not messages:
        return {"intent": "chat", "direct_reply": "你好！有什么可以帮助你的吗？"}

    last_msg = messages[-1]
    query = str(last_msg.content) if hasattr(last_msg, "content") else str(last_msg)

    history = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:256]}"
        for m in messages[:-1]
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
            "id": st.id, "description": st.description,
            "assigned_agent": st.assigned_agent, "dependencies": st.dependencies,
        })

    result: dict[str, Any] = {
        "intent": decomposition.intent,
        "reasoning": decomposition.reasoning,
        "direct_reply": decomposition.direct_reply or "",
        "subtasks": subtasks_dict,
        "agent_results": [],
        "final_output": "",
    }

    if decomposition.direct_reply:
        result["messages"] = [AIMessage(content=decomposition.direct_reply)]
        result["final_output"] = decomposition.direct_reply

    logger.info(
        f"Supervisor: intent={decomposition.intent}, "
        f"subtasks={len(subtasks_dict)}, direct={bool(decomposition.direct_reply)}"
    )
    return result


async def _run_subtask(
    st: dict[str, Any],
    query: str,
    history: list[str],
    completed: dict[int, dict[str, Any]],
    session_id: str,
) -> dict[str, Any]:
    """执行单个子任务并返回稳定的结构化结果。"""
    agent_name = st.get("assigned_agent", "")
    task_desc = st.get("description", "")

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
            "original_query": query,
            "history": history,
            "session_id": session_id,
            "completed_subtasks": {
                str(task_id): {"output": item.get("output", "")}
                for task_id, item in completed.items()
            },
        }
        tracer = get_tracer()
        try:
            with tracer.span(
                f"agent:{agent.name}",
                {"subtask_id": st.get("id"), "task": task_desc[:200]},
            ):
                result = await agent.execute(task=task_desc, context=context)
        except Exception as exc:
            logger.error(f"Agent '{agent.name}' 执行异常: {exc}")
            result = AgentResult(
                agent_name=agent.name,
                success=False,
                error=str(exc),
            )

    return {
        "id": st.get("id"), "agent_name": agent_name,
        "success": result.success, "output": result.output,
        "data": result.data, "error": result.error,
        "elapsed_ms": result.elapsed_ms, "tokens_used": result.tokens_used,
    }


def _dependency_failure(st: dict[str, Any], failed_ids: list[int]) -> dict[str, Any]:
    """构造依赖失败结果，阻止下游任务在无效输入上继续执行。"""
    return {
        "id": st.get("id"),
        "agent_name": st.get("assigned_agent", ""),
        "success": False,
        "output": "",
        "data": {},
        "error": f"依赖子任务执行失败: {failed_ids}",
        "elapsed_ms": 0,
        "tokens_used": 0,
    }


async def _agent_executor_node(state: MultiAgentState) -> dict[str, Any]:
    """按依赖关系分批并行执行全部子任务。"""
    subtasks = state.get("subtasks", [])
    if not subtasks:
        return {"agent_results": []}

    messages = select_conversation_context(
        list(state.get("messages", [])),
        state.get("conversation_summary", ""),
        Settings.load().memory_working_max_turns,
    )
    history = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:256]}"
        for m in messages
        if hasattr(m, "content") and m.content
    ]
    query_value = next(
        (
            m.content
            for m in reversed(messages)
            if isinstance(m, HumanMessage) and m.content
        ),
        "",
    )
    query = query_value if isinstance(query_value, str) else str(query_value)

    pending = {
        int(st["id"]): st
        for st in subtasks
        if isinstance(st.get("id"), int)
    }
    completed: dict[int, dict[str, Any]] = {}

    while pending:
        blocked_ids = [
            task_id
            for task_id, st in pending.items()
            if any(
                dependency in completed and not completed[dependency].get("success")
                for dependency in st.get("dependencies", [])
            )
        ]
        for task_id in blocked_ids:
            st = pending.pop(task_id)
            failed_dependencies = [
                dependency
                for dependency in st.get("dependencies", [])
                if dependency in completed and not completed[dependency].get("success")
            ]
            completed[task_id] = _dependency_failure(st, failed_dependencies)

        ready_ids = [
            task_id
            for task_id, st in pending.items()
            if all(
                dependency in completed and completed[dependency].get("success")
                for dependency in st.get("dependencies", [])
            )
        ]
        if not ready_ids:
            for task_id, st in pending.items():
                completed[task_id] = {
                    **_dependency_failure(st, st.get("dependencies", [])),
                    "error": f"子任务依赖不存在或形成循环: {st.get('dependencies', [])}",
                }
            break

        completed_snapshot = dict(completed)
        ready_tasks = [pending.pop(task_id) for task_id in ready_ids]
        results = await asyncio.gather(
            *(
                _run_subtask(
                    st,
                    query,
                    history,
                    completed_snapshot,
                    state.get("session_id", ""),
                )
                for st in ready_tasks
            )
        )
        for task_id, result in zip(ready_ids, results):
            completed[task_id] = result

    logger.info(
        f"子任务执行完成: total={len(subtasks)}, "
        f"success={sum(bool(item.get('success')) for item in completed.values())}"
    )

    return {
        "agent_results": [
            completed[int(st["id"])]
            for st in subtasks
            if isinstance(st.get("id"), int) and int(st["id"]) in completed
        ]
    }


async def _aggregator_node(state: MultiAgentState) -> dict[str, Any]:
    """聚合节点——Reflection 自检 + 生成最终回复。"""
    agent_results = state.get("agent_results", [])
    messages = select_conversation_context(
        list(state.get("messages", [])),
        state.get("conversation_summary", ""),
        Settings.load().memory_working_max_turns,
    )

    query = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage) and m.content:
            query = (
                m.content[:200]
                if isinstance(m.content, str)
                else str(m.content)[:200]
            )
            break

    from agentkb.agents.reflection import ReflectionModule
    from agentkb.agents.supervisor import SupervisorAgent
    from agentkb.llm.factory import get_chat_model

    reflection = ReflectionModule()
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
        llm_client=None,
    )

    if refined.get("needs_revision") and refined.get("revised_output"):
        logger.info("Reflection 触发修订")
        final_output = str(refined["revised_output"])
    else:
        llm = get_chat_model(streaming=False)
        supervisor = SupervisorAgent(llm_client=llm)
        final_output = await supervisor.aggregate(agent_results, query)

    logger.info(f"Aggregator: final_output length={len(final_output)}")

    return {
        "final_output": final_output,
        "messages": [AIMessage(content=final_output)],
    }


# ══════════════════════════════════════════════════════════════
#  条件路由
# ══════════════════════════════════════════════════════════════

def _route_after_supervisor(state: MultiAgentState) -> str:
    if state.get("direct_reply"):
        return END
    if state.get("subtasks"):
        return "agent_executor"
    return END


# ══════════════════════════════════════════════════════════════
#  图构建
# ══════════════════════════════════════════════════════════════

async def build_multi_agent_graph(
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    workflow = StateGraph(MultiAgentState)
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("agent_executor", _agent_executor_node)
    workflow.add_node("aggregator", _aggregator_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", _route_after_supervisor,
                                   {"agent_executor": "agent_executor", END: END})
    workflow.add_edge("agent_executor", "aggregator")
    workflow.add_edge("aggregator", END)
    return workflow.compile(checkpointer=checkpointer)


# ══════════════════════════════════════════════════════════════
#  MultiAgentGraph 流式封装
# ══════════════════════════════════════════════════════════════

class MultiAgentGraph:

    def __init__(self, graph: CompiledStateGraph) -> None:
        self._graph = graph

    @classmethod
    async def create(
        cls,
        checkpointer: BaseCheckpointSaver,
    ) -> MultiAgentGraph:
        graph = await build_multi_agent_graph(checkpointer)
        return cls(graph)

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        thread_id: str = "default",
        history: list[AnyMessage] | None = None,
        conversation_summary: str = "",
    ) -> AsyncGenerator[dict[str, Any], None]:
        tracer = get_tracer()

        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}
        input_messages: list[AnyMessage] = []
        if history:
            snapshot = self._graph.get_state(config)
            if not snapshot.values.get("messages"):
                input_messages.extend(history)
        input_messages.append(HumanMessage(content=user_input))
        input_state: dict[str, Any] = {
            "session_id": session_id,
            "messages": input_messages,
            "conversation_summary": conversation_summary,
        }

        try:
            with tracer.start_trace(
                session_id=session_id,
                query=user_input,
            ) as trace:
                # 推送 supervisor 启动
                yield {"type": "tool_start", "name": "supervisor",
                       "input": {"query": user_input}}

                # 同步执行整个图（不依赖 astream_events 捕获内部 token）
                final_state = await self._graph.ainvoke(input_state, config)
                trace.total_tokens += sum(
                    int(result.get("tokens_used", 0))
                    for result in final_state.get("agent_results", [])
                )
                trace.total_tool_calls += sum(
                    int(result.get("tool_calls_count", 0))
                    for result in final_state.get("agent_results", [])
                )

                # 推送 supervisor 完成
                yield {"type": "tool_end", "name": "supervisor",
                       "output": json.dumps({
                           "intent": final_state.get("intent", ""),
                           "subtasks_count": len(final_state.get("subtasks", [])),
                           "direct_reply": bool(final_state.get("direct_reply")),
                       }, ensure_ascii=False)[:2048]}

                # 推送各 Agent 的执行状态
                for r in final_state.get("agent_results", []):
                    agent_name = r.get("agent_name", "unknown")
                    yield {"type": "tool_start", "name": agent_name,
                           "input": {"agent": agent_name}}
                    yield {"type": "tool_end", "name": agent_name,
                           "output": json.dumps({
                               "success": r.get("success"),
                               "output_preview": (r.get("output", "") or "")[:200],
                               "elapsed_ms": r.get("elapsed_ms", 0),
                           }, ensure_ascii=False)[:2048]}

                # 完整结果已生成，仅分块传输，避免人为延长响应时间。
                final_output = final_state.get("final_output", "")
                if final_output:
                    chunk_size = 64
                    for i in range(0, len(final_output), chunk_size):
                        yield {"type": "token",
                               "content": final_output[i:i + chunk_size]}
                        await asyncio.sleep(0)

                yield {"type": "done", "session_id": session_id}

        except Exception as exc:
            logger.error(f"Multi-Agent 流式执行出错: {exc}", exc_info=True)
            yield {"type": "error", "message": f"处理请求时出错：{exc}"}
