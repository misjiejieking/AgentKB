from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agentkb.memory.context import SessionSummaryService, select_conversation_context


def test_select_context_keeps_complete_recent_turn_and_summary():
    messages = [
        HumanMessage(content="第一轮"),
        AIMessage(
            content="",
            tool_calls=[{"id": "call-1", "name": "search", "args": {}}],
        ),
        ToolMessage(content="结果", tool_call_id="call-1"),
        AIMessage(content="第一轮回答"),
        HumanMessage(content="第二轮"),
        AIMessage(content="第二轮回答"),
    ]

    context = select_conversation_context(messages, "更早摘要", max_turns=1)

    assert [message.content for message in context] == [
        "较早会话摘要：更早摘要",
        "第二轮",
        "第二轮回答",
    ]


class FakeSummaryDatabase:
    def __init__(self) -> None:
        self.rows = [
            {"role": "human", "content": "第一问", "sequence": 1},
            {"role": "ai", "content": "第一答", "sequence": 2},
            {"role": "human", "content": "第二问", "sequence": 3},
            {"role": "ai", "content": "第二答", "sequence": 4},
            {"role": "human", "content": "第三问", "sequence": 5},
            {"role": "ai", "content": "第三答", "sequence": 6},
        ]
        self.saved: tuple[str, str, int] | None = None

    def get_messages(self, session_id):
        return self.rows

    def get_session_summary(self, session_id):
        return None

    def upsert_session_summary(self, session_id, summary, covered_sequence):
        self.saved = (session_id, summary, covered_sequence)


class FakeSummaryModel:
    async def ainvoke(self, prompt):
        assert "第一问" in prompt
        assert "第二问" not in prompt
        return AIMessage(content="用户完成了第一轮问答。")


async def test_summary_service_only_covers_messages_outside_window():
    db = FakeSummaryDatabase()

    updated = await SessionSummaryService(db).refresh(
        "session-1",
        max_turns=2,
        llm=FakeSummaryModel(),
    )

    assert updated is True
    assert db.saved == ("session-1", "用户完成了第一轮问答。", 2)
