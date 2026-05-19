"""BGE-Reranker 交叉编码器——对候选文档精排，取 top-K。"""

from __future__ import annotations

from loguru import logger
from sentence_transformers import CrossEncoder

from agentkb.utils.exceptions import KnowledgeBaseError


class RerankerService:
    """BGE-Reranker v2-m3 交叉编码器，直接对 (query, doc) 对打分。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
    ) -> None:
        logger.info(f"正在加载 Reranker 模型: {model_name}（设备={device}）")
        try:
            self._model = CrossEncoder(model_name, device=device, trust_remote_code=True)
        except Exception as e:
            raise KnowledgeBaseError(f"无法加载 Reranker 模型 '{model_name}': {e}") from e
        logger.info("Reranker 模型就绪")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        """对候选集精排，返回 top_k 个结果（含 rerank_score）。"""
        if not documents:
            return []

        # 构造 (query, content) 对
        pairs = [(query, doc.get("content", "") or "") for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=len(pairs) > 20)

        scored = []
        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = round(float(score), 4)
            scored.append(doc_copy)

        scored.sort(key=lambda d: d["rerank_score"], reverse=True)
        return scored[:top_k]


# 模块级单例
_reranker: RerankerService | None = None


def get_reranker(
    model_name: str = "BAAI/bge-reranker-v2-m3",
    device: str = "cpu",
) -> RerankerService:
    global _reranker
    if _reranker is None:
        _reranker = RerankerService(model_name=model_name, device=device)
    return _reranker
