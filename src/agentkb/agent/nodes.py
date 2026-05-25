"""LangGraph 节点实现：agent_node（LLM 决策）和 tools_node（工具执行）。"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from loguru import logger

from agentkb.agent.prompts import SYSTEM_PROMPT, FALLBACK_MESSAGE
from agentkb.agent.router import IntentRouter, Intent
from agentkb.agent.state import AgentState
from agentkb.llm.factory import get_chat_model
from agentkb.tools.registry import ToolRegistry


async def agent_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Agent 决策节点——先路由意图，再决定是否绑工具。"""
    llm = get_chat_model(streaming=True)
    registry = ToolRegistry()
    tools = registry.get_langchain_tools()

    session_id = state.get("session_id", "default")
    system = SystemMessage(content=SYSTEM_PROMPT.format(session_id=session_id))

    messages = list(state.get("messages", []))
    if not messages or not isinstance(messages[0], SystemMessage):
        invoke_messages = [system] + messages
    else:
        invoke_messages = messages

    # 同步对话历史到 knowledge_search 模块用于查询重写
    from agentkb.tools.knowledge_search import update_chat_history
    update_chat_history(messages)

    # 意图路由——仅首轮决策时使用（后续轮次已有 tool_calls 上下文）
    is_first_round = not any(
        isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
        for m in messages
    )

    from agentkb.config.settings import Settings
    cfg = Settings.load()

    if is_first_round and len(messages) <= 1:
        # 获取用户最新消息
        user_msg = ""
        for m in reversed(messages):
            if hasattr(m, "content") and not isinstance(m, SystemMessage):
                user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break

        router_llm = get_chat_model(streaming=False)  # 路由用小模型，不流式
        router = IntentRouter(llm_client=router_llm)
        intent = await router.classify(user_msg)

        if intent == Intent.CHAT:
            logger.info(f"路由: chat（无需工具）")
            try:
                response: BaseMessage = await asyncio.wait_for(
                    llm.ainvoke(invoke_messages),  # 不绑工具
                    timeout=cfg.llm_request_timeout,
                )
            except asyncio.TimeoutError:
                logger.error("LLM 调用超时")
                return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}
            content = getattr(response, "content", "")
            if isinstance(content, str) and not content.strip():
                return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}
            return {"messages": [response]}

    # 非 chat 意图或后续轮次：正常绑工具调用
    llm_with_tools = llm.bind_tools(tools) if tools else llm

    try:
        response: BaseMessage = await asyncio.wait_for(
            llm_with_tools.ainvoke(invoke_messages),
            timeout=cfg.llm_request_timeout,
        )
    except asyncio.TimeoutError:
        logger.error("LLM 调用超时（{}秒）", cfg.llm_request_timeout)
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}
    except Exception as exc:
        logger.error(f"LLM 调用异常: {exc}")
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}

    # 空内容兜底
    content = getattr(response, "content", "")
    if isinstance(content, str) and not content.strip() and not getattr(response, "tool_calls", None):
        return {"messages": [AIMessage(content=FALLBACK_MESSAGE)]}

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
