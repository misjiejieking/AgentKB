"""LangGraph 单 Agent 状态定义。"""

from __future__ import annotations

import operator
from typing import Annotated, Any

from langgraph.graph import MessagesState


class AgentState(MessagesState, total=False):
    """单 Agent 图的完整状态。

    `messages` 使用 LangGraph 内置 reducer，其余日志字段使用列表拼接 reducer，
    保证工具循环中每个节点只返回本轮增量。
    """

    session_id: str
    run_id: str
    conversation_summary: str
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]
