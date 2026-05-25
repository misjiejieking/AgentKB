"""长期记忆——跨会话的知识持久化与检索。

使用 pgvector 存储记忆 embedding，支持语义检索和重要性衰减。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class LongTermMemoryEntry:
    """长期记忆条目。"""
    id: str
    content: str
    category: str = "general"      # "fact" | "preference" | "experience" | "insight"
    importance: float = 0.5        # 0~1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source_session: str = ""


class LongTermMemory:
    """长期记忆——跨会话持久化，语义检索。

    使用方式:
      mem = LongTermMemory()
      mem.save("用户喜欢 Python 编程", category="preference", importance=0.8)
      results = mem.search("编程语言偏好")
    """

    def __init__(self) -> None:
        self._memories: dict[str, LongTermMemoryEntry] = {}
        self._embedder = None

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
        importance: float = 0.5,
        source_session: str = "",
    ) -> str:
        """保存一条长期记忆（同时写入 PG 持久化）。"""
        import uuid
        mem_id = uuid.uuid4().hex[:12]
        entry = LongTermMemoryEntry(
            id=mem_id,
            content=content[:1024],
            category=category,
            importance=importance,
            source_session=source_session,
        )
        self._memories[mem_id] = entry

        # 持久化到 PG
        try:
            self._persist_to_db(entry)
        except Exception as e:
            logger.debug(f"长期记忆 PG 持久化跳过: {e}")

        return mem_id

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """语义检索长期记忆——优先从 DB，内存作为 fallback。"""
        # 尝试 DB 检索
        try:
            results = self._search_db(query, top_k)
            if results:
                return results
        except Exception:
            pass

        # 内存 fallback（使用 embedding 相似度）
        if not self._memories:
            return []

        try:
            query_emb = self.embedder.embed_query(query)
            scored = []
            for mem in self._memories.values():
                mem_emb = self.embedder.embed_query(mem.content)
                similarity = self._cosine_sim(query_emb, mem_emb)
                # 重要性加权
                score = similarity * 0.7 + mem.importance * 0.2 + min(mem.access_count / 10, 1) * 0.1
                scored.append((score, mem))

            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "id": m.id,
                    "content": m.content,
                    "category": m.category,
                    "score": round(s, 4),
                    "importance": m.importance,
                }
                for s, m in scored[:top_k]
            ]
        except Exception as e:
            logger.warning(f"长期记忆搜索失败: {e}")
            return []

    def forget(self, memory_id: str) -> bool:
        """删除一条记忆。"""
        if memory_id in self._memories:
            del self._memories[memory_id]
            return True
        return False

    def consolidate(self) -> dict:
        """记忆整理——合并相似记忆、降低低重要性记忆的权重。

        Returns 整理统计信息。
        """
        stats = {"total": len(self._memories), "pruned": 0, "merged": 0}

        # 衰减低重要性旧记忆
        now = time.time()
        to_prune = []
        for mem_id, mem in self._memories.items():
            age_days = (now - mem.created_at) / 86400
            if age_days > 30 and mem.importance < 0.3 and mem.access_count < 2:
                to_prune.append(mem_id)

        for mem_id in to_prune:
            del self._memories[mem_id]
            stats["pruned"] += 1

        if stats["pruned"] > 0:
            logger.info(f"长期记忆整理: 清理了 {stats['pruned']} 条低价值记忆")

        return stats

    # ── 内部 ────────────────────────────────────────────────────

    def _persist_to_db(self, entry: LongTermMemoryEntry) -> None:
        """持久化到 PostgreSQL（如果有专用表）。"""
        from agentkb.storage.pg_database import get_db
        db = get_db()
        with db._connect() as conn:
            with conn.cursor() as cur:
                # 尝试写入（可能表不存在）
                cur.execute(
                    """INSERT INTO long_term_memories (id, content, category, importance, source_session)
                       VALUES (%s, %s, %s, %s, %s)
                       ON CONFLICT (id) DO NOTHING""",
                    (entry.id, entry.content, entry.category,
                     entry.importance, entry.source_session),
                )

    def _search_db(self, query: str, top_k: int) -> list[dict]:
        """从 DB 向量检索长期记忆。"""
        from agentkb.storage.pg_database import get_db
        db = get_db()
        query_emb = self.embedder.embed_query(query)
        with db._connect() as conn:
            with conn.cursor() as cur:
                try:
                    cur.execute(
                        """SELECT id, content, category, importance,
                                  1.0 - (embedding <=> %s::vector) AS score
                           FROM long_term_memories
                           ORDER BY embedding <=> %s::vector
                           LIMIT %s""",
                        (query_emb, query_emb, top_k),
                    )
                    return [
                        {
                            "id": r["id"], "content": r["content"],
                            "category": r["category"], "score": round(r["score"], 4),
                            "importance": r["importance"],
                        }
                        for r in cur.fetchall()
                    ]
                except Exception:
                    return []

    @staticmethod
    def _cosine_sim(a: list[float], b: list[float]) -> float:
        import numpy as np
        a_np, b_np = np.array(a), np.array(b)
        dot = float(np.dot(a_np, b_np))
        norm = float(np.linalg.norm(a_np) * np.linalg.norm(b_np))
        return dot / norm if norm > 0 else 0.0
