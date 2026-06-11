from __future__ import annotations

from agentkb.memory.long_term import LongTermMemory


class FakeEmbedder:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed_query(self, text):
        self.queries.append(text)
        return [0.1, 0.2]


class FakeMemoryDatabase:
    def __init__(self) -> None:
        self.saved = None
        self.deleted = ""

    def add_long_term_memory(self, **kwargs):
        self.saved = kwargs
        return kwargs["memory_id"]

    def search_long_term_memories(self, embedding, top_k):
        assert embedding == [0.1, 0.2]
        assert top_k == 3
        return [
            {
                "id": "memory-1",
                "content": "用户偏好简洁回答",
                "category": "preference",
                "importance": 0.9,
                "score": 0.87654,
            }
        ]

    def delete_long_term_memory(self, memory_id):
        self.deleted = memory_id
        return True


def test_long_term_memory_persists_embedding_and_searches_database():
    db = FakeMemoryDatabase()
    embedder = FakeEmbedder()
    memory = LongTermMemory(db=db, embedder=embedder)

    memory_id = memory.save(
        "  用户偏好简洁回答  ",
        category="preference",
        importance=0.9,
        source_session="session-1",
    )
    results = memory.search("回答偏好", top_k=3)

    assert db.saved["content"] == "用户偏好简洁回答"
    assert db.saved["embedding"] == [0.1, 0.2]
    assert memory_id == db.saved["memory_id"]
    assert results[0]["score"] == 0.8765
    assert embedder.queries == ["用户偏好简洁回答", "回答偏好"]


def test_long_term_memory_forget_deletes_persisted_record():
    db = FakeMemoryDatabase()
    memory = LongTermMemory(db=db, embedder=FakeEmbedder())

    assert memory.forget("memory-1") is True
    assert db.deleted == "memory-1"
