from __future__ import annotations

import asyncio

from langchain_core.messages import HumanMessage

from agentkb.tools.knowledge_search import chat_history_context, get_chat_history


async def test_chat_history_context_is_isolated_between_tasks():
    async def read_history(content: str, delay: float) -> list[str]:
        with chat_history_context([HumanMessage(content=content)]):
            await asyncio.sleep(delay)
            return get_chat_history()

    first, second = await asyncio.gather(
        read_history("会话一", 0.02),
        read_history("会话二", 0),
    )

    assert first == ["会话一"]
    assert second == ["会话二"]
