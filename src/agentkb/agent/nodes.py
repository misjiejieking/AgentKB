"""LangGraph 节点实现：agent_node（LLM 决策）和 tools_node（工具执行）。"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agentkb.agent.prompts import SYSTEM_PROMPT
from agentkb.agent.state import AgentState
from agentkb.llm.factory import get_chat_model
from agentkb.tools.registry import ToolRegistry


async def agent_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Agent 决策节点——调用绑定了工具的 LLM，返回 AI 消息（可能含 tool_calls）。"""
    llm = get_chat_model(streaming=True)
    registry = ToolRegistry()
    tools = registry.get_langchain_tools()
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    session_id = state.get("session_id", "default")
    system = SystemMessage(content=SYSTEM_PROMPT.format(session_id=session_id))

    messages = list(state.get("messages", []))
    # 确保首条消息为系统提示词（仅当头部缺失时插入）
    if not messages or not isinstance(messages[0], SystemMessage):
        invoke_messages = [system] + messages
    else:
        invoke_messages = messages

    logger.debug(f"agent_node: 用 {len(invoke_messages)} 条消息调用 LLM")
    response: BaseMessage = await llm_with_tools.ainvoke(invoke_messages)

    return {"messages": [response]}


async def tools_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """工具执行节点——遍历最后一条 AI 消息的 tool_calls，逐一执行并收集结果。"""
    messages = state.get("messages", [])
    if not messages:
        return {}

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    registry = ToolRegistry()
    tool_messages: list[ToolMessage] = []
    tool_calls_log: list[dict] = []
    tool_results_log: list[dict] = []

    for tc in last_msg.tool_calls:
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = tc.get("id", "")

        tool_calls_log.append({"id": tool_id, "name": tool_name, "args": tool_args})

        logger.info(f"执行工具: {tool_name}({tool_args})")
        result = await registry.execute(tool_name, **tool_args)

        tool_results_log.append({
            "name": tool_name,
            "success": result.success,
            "result": str(result.data)[:2048] if result.success else "",
            "elapsed_ms": result.elapsed_ms,
            "error": result.error,
        })

        tool_messages.append(ToolMessage(
            content=result.to_json(),
            tool_call_id=tool_id,
            name=tool_name,
        ))

    return {
        "messages": tool_messages,
        "tool_calls": state.get("tool_calls", []) + tool_calls_log,
        "tool_results": state.get("tool_results", []) + tool_results_log,
    }
