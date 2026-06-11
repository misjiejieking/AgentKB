"""混合检索——Dense (pgvector) + BM25 (tsvector) 并行搜索 + RRF 融合。"""

from __future__ import annotations

from loguru import logger

from agentkb.config.settings import Settings
from agentkb.knowledge.embedder import get_embedder
from agentkb.storage.pg_database import get_db


class HybridRetriever:
    """Dense + BM25 混合检索，RRF 加权融合结果。"""

    def __init__(self) -> None:
        self._db = get_db()
        self._embedder = get_embedder()

    def retrieve(self, query: str) -> list[dict]:
        """执行混合检索：dense + bm25 → RRF 融合 → 返回候选集，失败时逐级降级。"""
        cfg = Settings.load()

        # 1. 生成查询向量
        query_vector = self._embedder.embed_query(query)

        # 2. 分别检索，单侧失败不崩溃
        dense_results = []
        bm25_results = []
        degraded = False
        degrade_reason = ""

        try:
            dense_results = self._db.search_dense(query_vector, limit=cfg.retrieval_candidate_k)
        except Exception as e:
            logger.warning(f"Dense 检索失败，降级: {e}")
            degraded = True
            degrade_reason = "dense 检索异常"

        try:
            bm25_results = self._db.search_bm25(query, limit=cfg.retrieval_candidate_k)
        except Exception as e:
            logger.warning(f"BM25 检索失败，降级: {e}")
            degraded = True
            degrade_reason = degrade_reason or "bm25 检索异常"

        # 3. 三级降级融合
        if dense_results and bm25_results:
            merged = self._rrf_fusion(
                dense_results,
                bm25_results,
                dense_weight=cfg.retrieval_dense_weight,
                bm25_weight=cfg.retrieval_bm25_weight,
                k=cfg.retrieval_rrf_k,
            )
        elif dense_results:
            merged = dense_results  # 降级到 pure dense
            degraded = True
            degrade_reason = degrade_reason or "降级到 dense-only"
            logger.info("BM25 不可用，降级到 pure dense")
        elif bm25_results:
            merged = bm25_results  # 降级到 pure bm25
            degraded = True
            degrade_reason = degrade_reason or "降级到 bm25-only"
            logger.info("Dense 不可用，降级到 pure bm25")
        else:
            return []  # 全部失败

        # 合并降级信息到每一条结果
        if degraded:
            for item in merged:
                item["degraded"] = True
                item["degrade_reason"] = degrade_reason

        # 4. 按分数排序，返回 top candidate_k
        def sort_key(item: dict) -> float:
            return float(item.get("rrf_score", item.get("score", 0)))

        merged.sort(key=sort_key, reverse=True)
        return merged[:cfg.retrieval_candidate_k]

    @staticmethod
    def _rrf_fusion(
        dense_list: list[dict],
        bm25_list: list[dict],
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        k: int = 60,
    ) -> list[dict]:
        """Reciprocal Rank Fusion — 合并两个排序列表。"""
        id_to_item: dict[str, dict] = {}
        id_to_score: dict[str, float] = {}

        # Dense 结果
        for rank, item in enumerate(dense_list):
            item_id = item["id"]
            id_to_item[item_id] = dict(item)
            id_to_score[item_id] = dense_weight / (k + rank + 1)

        # BM25 结果
        for rank, item in enumerate(bm25_list):
            item_id = item["id"]
            if item_id in id_to_item:
                # 已出现，保留更高的原始 score 的那份信息
                existing = id_to_item[item_id]
                if item.get("score", 0) > existing.get("score", 0):
                    id_to_item[item_id] = dict(item)
            else:
                id_to_item[item_id] = dict(item)
            id_to_score[item_id] = id_to_score.get(item_id, 0) + bm25_weight / (k + rank + 1)

        # 合并分数到 item
        for item_id, item in id_to_item.items():
            item["rrf_score"] = round(id_to_score.get(item_id, 0), 6)

        return list(id_to_item.values())


# 模块级单例
_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
