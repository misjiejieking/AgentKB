"""Multi-Agent LangGraph 图——将 Supervisor→Agent→聚合→Reflection 编码为 StateGraph。

图结构:
  START → supervisor_node (意图+分解)
           ├─ chat/direct_reply → END
           └─ subtasks → agent_executor_node (loop)
                          └─ all done → aggregator_node (Reflection+聚合) → END

流式策略:
  不用 astream_events 捕获节点内 LLM token（因为 Specialist Agent 内部创建
  的 LLM 实例与 LangGraph callback chain 不连通）。改用 ainvoke 拿到完整
  final_output 后分块推送模拟流式。
"""

from __future__ import annotations

import asyncio
import json
from typing import AsyncGenerator, Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState, StateGraph, END
from loguru import logger

from agentkb.observability.tracer import get_tracer


# ══════════════════════════════════════════════════════════════
#  节点定义
# ══════════════════════════════════════════════════════════════

async def _supervisor_node(state: dict) -> dict:
    """Supervisor 节点——意图分析 + 任务分解 + Agent 路由。"""
    messages = state.get("messages", [])
    if not messages:
        return {"intent": "chat", "direct_reply": "你好！有什么可以帮助你的吗？"}

    last_msg = messages[-1]
    query = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

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

    result = {
        "intent": decomposition.intent,
        "reasoning": decomposition.reasoning,
        "direct_reply": decomposition.direct_reply or "",
        "subtasks": subtasks_dict,
        "current_subtask_index": 0,
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


async def _agent_executor_node(state: dict) -> dict:
    """Agent 执行节点——执行当前 subtask，推结果到 agent_results。"""
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)

    if idx >= len(subtasks):
        return {"final_output": "所有子任务已完成"}

    st = subtasks[idx]
    agent_name = st.get("assigned_agent", "")
    task_desc = st.get("description", "")

    messages = state.get("messages", [])
    history = [
        f"{'用户' if isinstance(m, HumanMessage) else '助手'}: {m.content[:256]}"
        for m in messages
        if hasattr(m, "content") and m.content
    ]

    completed = {
        str(r.get("id", i)): {"output": r.get("output", "")}
        for i, r in enumerate(state.get("agent_results", []))
    }

    from agentkb.agents.registry import get_agent_registry
    from agentkb.agents.base import AgentResult

    registry = get_agent_registry()
    agent = registry.get(agent_name)
    if agent is None:
        candidates = registry.find_by_intent(agent_name)
        agent = candidates[0] if candidates else None

    if agent is None:
        logger.warning(f"Agent '{agent_name}' 未注册")
        result = AgentResult(agent_name=agent_name, success=False,
                             error=f"Agent '{agent_name}' 未注册")
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
        "id": st.get("id"), "agent_name": agent_name,
        "success": result.success, "output": result.output,
        "data": result.data, "error": result.error,
        "elapsed_ms": result.elapsed_ms, "tokens_used": result.tokens_used,
    })

    logger.info(
        f"Agent '{agent_name}' subtask {idx + 1}/{len(subtasks)}: "
        f"success={result.success}, elapsed={result.elapsed_ms:.0f}ms"
    )

    return {"current_subtask_index": idx + 1, "agent_results": agent_results}


async def _aggregator_node(state: dict) -> dict:
    """聚合节点——Reflection 自检 + 生成最终回复。"""
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
        final_output = refined["revised_output"]
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

def _route_after_supervisor(state: dict) -> str:
    if state.get("direct_reply"):
        return END
    if state.get("subtasks"):
        return "agent_executor"
    return END


def _route_after_executor(state: dict) -> str:
    subtasks = state.get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    if subtasks and idx < len(subtasks):
        return "agent_executor"
    return "aggregator"


# ══════════════════════════════════════════════════════════════
#  图构建
# ══════════════════════════════════════════════════════════════

async def build_multi_agent_graph() -> StateGraph:
    workflow = StateGraph(MessagesState)
    workflow.add_node("supervisor", _supervisor_node)
    workflow.add_node("agent_executor", _agent_executor_node)
    workflow.add_node("aggregator", _aggregator_node)
    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges("supervisor", _route_after_supervisor,
                                   {"agent_executor": "agent_executor", END: END})
    workflow.add_conditional_edges("agent_executor", _route_after_executor,
                                   {"agent_executor": "agent_executor", "aggregator": "aggregator"})
    workflow.add_edge("aggregator", END)
    return workflow.compile(checkpointer=MemorySaver())


# ══════════════════════════════════════════════════════════════
#  MultiAgentGraph 流式封装
# ══════════════════════════════════════════════════════════════

class MultiAgentGraph:

    def __init__(self, graph=None) -> None:
        self._graph = graph

    @classmethod
    async def create(cls) -> MultiAgentGraph:
        graph = await build_multi_agent_graph()
        return cls(graph)

    async def stream(
        self,
        user_input: str,
        session_id: str = "default",
        thread_id: str = "default",
        history: list | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        if self._graph is None:
            self._graph = await build_multi_agent_graph()

        tracer = get_tracer()

        input_messages = list(history) if history else []
        input_messages.append(HumanMessage(content=user_input))
        input_state = {"session_id": session_id, "messages": input_messages}
        config = {"configurable": {"thread_id": thread_id}}

        try:
            with tracer.start_trace(session_id=session_id, query=user_input):
                # 推送 supervisor 启动
                yield {"type": "tool_start", "name": "supervisor",
                       "input": {"query": user_input}}

                # 同步执行整个图（不依赖 astream_events 捕获内部 token）
                final_state = await self._graph.ainvoke(input_state, config)

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

                # 推送最终回复——分块模拟流式
                final_output = final_state.get("final_output", "")
                if final_output:
                    chunk_size = 3
                    for i in range(0, len(final_output), chunk_size):
                        yield {"type": "token",
                               "content": final_output[i:i + chunk_size]}
                        await asyncio.sleep(0.015)

                yield {"type": "done", "session_id": session_id}

        except Exception as exc:
            logger.error(f"Multi-Agent 流式执行出错: {exc}", exc_info=True)
            yield {"type": "error", "message": f"处理请求时出错：{exc}"}
