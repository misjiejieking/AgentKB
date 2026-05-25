"""检索结果语义缓存——基于 query embedding 余弦相似度的 LRU 缓存。"""

from __future__ import annotations

import numpy as np
from loguru import logger


class QueryCache:
    """LRU 语义缓存：对 query embedding 做余弦相似度匹配，命中则跳过检索。

    使用方式:
        cache = QueryCache(max_size=1000, similarity_threshold=0.95)
        results = cache.get(embedding)  # None 表示未命中
        cache.set(embedding, results)
        cache.invalidate()  # 知识库更新后清空
    """

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95) -> None:
        self._max_size = max_size
        self._threshold = similarity_threshold
        self._embeddings: list[list[float]] = []
        self._results: list[list[dict]] = []

    def get(self, query_embedding: list[float]) -> list[dict] | None:
        """匹配缓存——余弦相似度超过阈值则返回缓存结果，否则返回 None。"""
        if not self._embeddings:
            return None

        q = np.array(query_embedding, dtype=np.float32)
        embeddings_array = np.array(self._embeddings, dtype=np.float32)

        # 批量计算余弦相似度
        norms = np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(q) + 1e-10
        similarities = np.dot(embeddings_array, q) / norms

        max_idx = int(np.argmax(similarities))
        max_sim = float(similarities[max_idx])

        if max_sim >= self._threshold:
            # LRU: 命中项移到末尾
            hit = self._results[max_idx]
            self._embeddings.pop(max_idx)
            self._results.pop(max_idx)
            self._embeddings.append(query_embedding)
            self._results.append(hit)
            logger.debug(f"缓存命中（sim={max_sim:.3f}）")
            return hit

        return None

    def set(self, query_embedding: list[float], results: list[dict]) -> None:
        """存入缓存。"""
        if len(self._embeddings) >= self._max_size:
            # 淘汰最早条目
            self._embeddings.pop(0)
            self._results.pop(0)

        self._embeddings.append(query_embedding)
        self._results.append(results)

    def invalidate(self) -> None:
        """清空全部缓存——知识库文件变更时调用。"""
        count = len(self._embeddings)
        self._embeddings.clear()
        self._results.clear()
        logger.info(f"检索缓存已失效（清除 {count} 条）")

    def __len__(self) -> int:
        return len(self._embeddings)


# 模块级单例
_cache: QueryCache | None = None


def get_cache() -> QueryCache:
    global _cache
    if _cache is None:
        _cache = QueryCache()
    return _cache
