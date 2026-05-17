"""LangGraph Agent 全局状态定义。"""

from __future__ import annotations


class AgentState(dict):
    """基于 dict 的 Agent 状态，避免 TypedDict 在不同 langgraph 版本中的兼容问题。

    messages 键由 langgraph 的 add_messages reducer 自动追加合并。

    键:
        messages: 消息历史（Annotated[list, add_messages]）
        session_id: 当前会话标识
        retrieved_docs: 本轮知识库检索结果
        tool_calls: 本轮工具调用记录
        tool_results: 本轮工具执行结果及耗时
    """
    pass
