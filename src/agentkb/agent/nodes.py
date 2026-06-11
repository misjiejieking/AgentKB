"""LangGraph 节点实现：agent_node（LLM 决策）和 tools_node（工具执行）。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt
from loguru import logger

from agentkb.agent.prompts import SYSTEM_PROMPT, FALLBACK_MESSAGE
from agentkb.agent.router import IntentRouter, Intent
from agentkb.agent.state import AgentState
from agentkb.config.settings import Settings
from agentkb.llm.factory import get_chat_model, get_router_chat_model
from agentkb.memory.context import select_conversation_context
from agentkb.tools.base import ToolResult
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
    cfg = Settings.load()
    invoke_messages = [
        system,
        *select_conversation_context(
            messages,
            state.get("conversation_summary", ""),
            cfg.memory_working_max_turns,
        ),
    ]

    # 每个用户轮次都重新路由；工具返回后的 Agent 续跑不重复分类。
    is_new_user_turn = bool(messages) and not isinstance(
        messages[-1],
        ToolMessage,
    )

    if is_new_user_turn:
        # 获取用户最新消息
        user_msg = ""
        for m in reversed(messages):
            if hasattr(m, "content") and not isinstance(m, SystemMessage):
                user_msg = m.content if isinstance(m.content, str) else str(m.content)
                break

        router_llm = get_router_chat_model(streaming=False)  # 路由用小模型，不流式
        router = IntentRouter(llm_client=router_llm)
        intent = await router.classify(user_msg)

        if intent == Intent.CHAT:
            logger.info("路由: chat（无需工具）")
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
        response = await asyncio.wait_for(
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
    session_id = state.get("session_id", "default")

    last_msg = messages[-1]
    if not isinstance(last_msg, AIMessage) or not last_msg.tool_calls:
        return {}

    registry = ToolRegistry()
    tool_messages: list[ToolMessage] = []
    tool_calls_log: list[dict] = []
    tool_results_log: list[dict] = []

    for call_index, tc in enumerate(last_msg.tool_calls):
        tool_name = tc.get("name", "")
        tool_args = tc.get("args", {})
        tool_id = str(tc.get("id") or "")
        if not tool_id:
            payload = json.dumps(
                {"index": call_index, "name": tool_name, "args": tool_args},
                ensure_ascii=False,
                sort_keys=True,
            )
            tool_id = f"call_{hashlib.sha256(payload.encode()).hexdigest()[:16]}"
            tc["id"] = tool_id
        tool = registry.get(tool_name)

        tool_calls_log.append({"id": tool_id, "name": tool_name, "args": tool_args})

        logger.info(f"执行工具: {tool_name}({tool_args})")
        from agentkb.tools.knowledge_search import chat_history_context
        from agentkb.tools.personal_memory import personal_memory_context

        approval_id = ""
        if tool is not None and tool.requires_confirmation:
            from agentkb.storage.pg_database import get_db

            thread_id = str(config["configurable"]["thread_id"])
            approval_id = hashlib.sha256(
                f"{thread_id}:{tool_id}".encode()
            ).hexdigest()[:24]
            await asyncio.to_thread(
                get_db().create_tool_approval,
                approval_id=approval_id,
                run_id=state.get("run_id", ""),
                session_id=session_id,
                thread_id=thread_id,
                tool_call_id=tool_id,
                tool_name=tool_name,
                arguments=tool_args,
            )
            decision = interrupt(
                {
                    "type": "tool_approval",
                    "approval_id": approval_id,
                    "tool_name": tool_name,
                    "arguments": tool_args,
                    "message": tool.confirmation_message,
                }
            )
            approved = (
                bool(decision.get("approved"))
                if isinstance(decision, dict)
                else bool(decision)
            )
            if not approved:
                result = ToolResult(
                    tool_name=tool_name,
                    success=False,
                    error="用户拒绝执行该高风险工具",
                )
            else:
                with (
                    chat_history_context(messages),
                    personal_memory_context(session_id),
                ):
                    result = await registry.execute(tool_name, **tool_args)
                await asyncio.to_thread(
                    get_db().complete_tool_approval,
                    approval_id,
                    success=result.success,
                    result=result.data,
                    error=result.error or "",
                )
        else:
            with chat_history_context(messages), personal_memory_context(session_id):
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
        "tool_calls": tool_calls_log,
        "tool_results": tool_results_log,
    }
