"""跨会话长期记忆服务。"""

from __future__ import annotations

from typing import Any

from agentkb.storage.models import new_id

MEMORY_CATEGORIES = {"fact", "preference", "experience", "insight", "general"}


class LongTermMemory:
    """使用 PostgreSQL 与 pgvector 持久化和检索用户记忆。"""

    def __init__(self, db=None, embedder=None) -> None:
        self._db = db
        self._embedder = embedder

    @property
    def db(self):
        if self._db is None:
            from agentkb.storage.pg_database import get_db

            self._db = get_db()
        return self._db

    @property
    def embedder(self):
        if self._embedder is None:
            from agentkb.knowledge.embedder import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def save(
        self,
        content: str,
        category: str = "general",
        importance: float = 0.8,
        source_session: str = "",
    ) -> str:
        """向量化后持久化一条明确的跨会话记忆。"""
        normalized = " ".join(content.split())
        if not normalized:
            raise ValueError("记忆内容不能为空")
        if category not in MEMORY_CATEGORIES:
            raise ValueError(f"不支持的记忆类别: {category}")
        if not 0 <= importance <= 1:
            raise ValueError("记忆重要性必须在 0 到 1 之间")

        normalized = normalized[:1024]
        embedding = self.embedder.embed_query(normalized)
        return self.db.add_long_term_memory(
            memory_id=new_id(),
            content=normalized,
            category=category,
            importance=importance,
            source_session=source_session,
            embedding=embedding,
        )

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """按语义相似度检索长期记忆。"""
        normalized = " ".join(query.split())
        if not normalized:
            raise ValueError("检索问题不能为空")
        if not 1 <= top_k <= 10:
            raise ValueError("top_k 必须在 1 到 10 之间")

        embedding = self.embedder.embed_query(normalized)
        rows = self.db.search_long_term_memories(embedding, top_k)
        return [
            {
                **row,
                "score": round(float(row["score"]), 4),
                "importance": float(row["importance"]),
            }
            for row in rows
        ]

    def forget(self, memory_id: str) -> bool:
        """从持久化存储删除一条长期记忆。"""
        return self.db.delete_long_term_memory(memory_id)
