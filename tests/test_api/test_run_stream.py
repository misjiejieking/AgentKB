from __future__ import annotations

import json

from agentkb.api.routes import RunEventSink, _run_stream_response


class FakeRunDatabase:
    def __init__(self, events: list[dict] | None = None) -> None:
        self.events = events or []
        self.appended: list[dict] = []

    def get_run_events(self, run_id: str, after_event_id: int, limit: int = 500):
        return [
            event
            for event in self.events
            if event["event_id"] > after_event_id
        ][:limit]

    def get_run(self, run_id: str):
        return {"id": run_id, "status": "completed"}

    def append_run_event(self, run_id: str, event: dict):
        self.appended.append(event)
        return len(self.appended)


async def _read_response(response) -> list[str]:
    chunks = []
    async for chunk in response.body_iterator:
        chunks.append(chunk)
    return chunks


async def test_run_stream_replays_persisted_events():
    db = FakeRunDatabase([
        {
            "event_id": 1,
            "payload": {"type": "message_id", "message_id": "message-1"},
        },
        {
            "event_id": 2,
            "payload": {"type": "token", "content": "回答"},
        },
    ])

    chunks = await _read_response(_run_stream_response(db, "run-1"))

    assert [chunk.splitlines()[0] for chunk in chunks] == ["id: 1", "id: 2"]
    assert json.loads(chunks[1].split("data: ", 1)[1]) == {
        "type": "token",
        "content": "回答",
    }


async def test_run_stream_resumes_after_last_event_id():
    db = FakeRunDatabase([
        {"event_id": 1, "payload": {"type": "token", "content": "第一段"}},
        {"event_id": 2, "payload": {"type": "token", "content": "第二段"}},
    ])

    chunks = await _read_response(
        _run_stream_response(db, "run-1", last_event_id=1)
    )

    assert len(chunks) == 1
    assert chunks[0].startswith("id: 2\n")


async def test_run_event_sink_flushes_tokens_before_terminal_event():
    db = FakeRunDatabase()
    sink = RunEventSink(db, "run-1")

    await sink.emit({"type": "token", "content": "第一段"})
    await sink.emit({"type": "token", "content": "第二段"})
    await sink.emit({"type": "done"})

    assert db.appended == [
        {"type": "token", "content": "第一段"},
        {"type": "token", "content": "第二段"},
        {"type": "done"},
    ]
