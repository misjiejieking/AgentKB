"""数据库驱动的可配置 Specialist Agent。"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from agentkb.agents.base import AgentResult, SpecialistAgent
from agentkb.llm.factory import get_chat_model, get_chat_model_for
from agentkb.tools.registry import ToolRegistry


class ConfigurableAgent(SpecialistAgent):
    """按持久化定义执行的受限 Agent。"""

    def __init__(self, definition: dict[str, Any]) -> None:
        self.definition = definition

    @property
    def name(self) -> str:
        return str(self.definition["name"])

    @property
    def description(self) -> str:
        return str(self.definition["description"])

    @property
    def intents(self) -> list[str]:
        return list(self.definition["intents"])

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        started_at = time.perf_counter()
        context = context or {}
        tool_registry = ToolRegistry()
        allowed_tools = self._allowed_tools(tool_registry)
        model_name = self.definition.get("model_name")
        llm = (
            get_chat_model_for(model_name, streaming=False)
            if model_name
            else get_chat_model(streaming=False)
        )
        model = llm.bind_tools(
            [tool.to_langchain_tool() for tool in allowed_tools]
        ) if allowed_tools else llm

        completed = context.get("completed_subtasks", {})
        prior_outputs = "\n\n".join(
            str(item.get("output", ""))[:2000]
            for item in completed.values()
            if item.get("output")
        )
        system_prompt = (
            f"你是“{self.definition['display_name']}”。\n"
            f"{self.definition['instructions']}\n\n"
            "必须直接解决任务；工具结果只是证据，不得编造未返回的信息。"
            "如工具失败，明确说明失败，不要伪造成功。"
        )
        user_prompt = task
        if prior_outputs:
            user_prompt += f"\n\n前置子任务结果：\n{prior_outputs}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        tool_calls_count = 0
        try:
            for _ in range(4):
                response = await model.ainvoke(messages)
                messages.append(response)
                calls = getattr(response, "tool_calls", None) or []
                if not calls:
                    output = _message_text(response.content)
                    if not output:
                        raise ValueError("Agent 未返回有效文本")
                    return AgentResult(
                        agent_name=self.name,
                        success=True,
                        output=output,
                        elapsed_ms=(time.perf_counter() - started_at) * 1000,
                        tool_calls_count=tool_calls_count,
                        tokens_used=_estimate_tokens(messages),
                    )

                for call in calls:
                    tool_name = str(call.get("name", ""))
                    if tool_name not in self.definition["allowed_tools"]:
                        raise PermissionError(f"Agent 无权调用工具: {tool_name}")
                    result = await tool_registry.execute(
                        tool_name,
                        **dict(call.get("args", {})),
                    )
                    tool_calls_count += 1
                    messages.append(ToolMessage(
                        content=result.to_json(),
                        tool_call_id=str(call.get("id", "")),
                        name=tool_name,
                    ))
            raise RuntimeError("Agent 工具调用轮次超过限制")
        except Exception as exc:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(exc),
                output=f"{self.definition['display_name']}执行失败：{exc}",
                elapsed_ms=(time.perf_counter() - started_at) * 1000,
                tool_calls_count=tool_calls_count,
            )

    def _allowed_tools(self, registry: ToolRegistry):
        tools = []
        for name in self.definition["allowed_tools"]:
            tool = registry.get(name)
            if tool is None:
                raise ValueError(f"配置的工具未注册: {name}")
            if tool.requires_confirmation:
                raise PermissionError(f"自定义 Agent 不允许使用高风险工具: {name}")
            tools.append(tool)
        return tools


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ).strip()
    return ""


def _estimate_tokens(messages: list[Any]) -> int:
    return sum(len(str(message.content)) for message in messages) // 2
