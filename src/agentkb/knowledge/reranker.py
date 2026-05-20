"""Reranker 精排——支持本地 CrossEncoder 和阿里百炼 DashScope API。"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

import httpx
from loguru import logger

from agentkb.config.settings import Settings
from agentkb.utils.exceptions import KnowledgeBaseError


class RerankerService(ABC):
    """Reranker 抽象基类——对候选文档精排，取 top-K。"""

    @abstractmethod
    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        ...


class BailianReranker(RerankerService):
    """阿里百炼 DashScope Reranker API。"""

    def __init__(
        self,
        model_name: str = "gte-rerank",
        base_url: str = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank",
        api_key: str = "",
        timeout: int = 30,
    ) -> None:
        self._model = model_name
        self._url = base_url
        self._key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self._timeout = timeout
        if not self._key:
            logger.warning("DASHSCOPE_API_KEY 未设置，Reranker API 可能无法调用")
        else:
            logger.info(f"Reranker API 就绪: {model_name}")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        contents = [doc.get("content", "") or "" for doc in documents]
        payload = {
            "model": self._model,
            "input": {
                "query": query,
                "documents": contents,
            },
            "parameters": {"top_n": top_k, "return_documents": False},
        }
        headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

        try:
            resp = httpx.post(self._url, json=payload, headers=headers, timeout=self._timeout)
            resp.raise_for_status()
            body = resp.json()
        except httpx.HTTPStatusError as e:
            raise KnowledgeBaseError(f"Reranker API 请求失败 ({e.response.status_code}): {e.response.text}") from e
        except httpx.RequestError as e:
            raise KnowledgeBaseError(f"Reranker API 网络错误: {e}") from e

        results = body.get("output", {}).get("results", [])
        scored = []
        for item in results:
            idx = item.get("index", 0)
            if idx < len(documents):
                doc_copy = dict(documents[idx])
                doc_copy["rerank_score"] = round(float(item.get("relevance_score", 0)), 4)
                scored.append(doc_copy)

        scored.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        return scored[:top_k]


class LocalReranker(RerankerService):
    """本地 BGE-Reranker CrossEncoder（需 sentence-transformers）。"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder

        logger.info(f"正在加载 Reranker 模型: {model_name}（设备={device}）")
        try:
            self._model = CrossEncoder(model_name, device=device, trust_remote_code=True, local_files_only=True)
        except Exception as e:
            raise KnowledgeBaseError(f"无法加载 Reranker 模型 '{model_name}': {e}") from e
        logger.info("Reranker 模型就绪")

    def rerank(self, query: str, documents: list[dict], top_k: int = 5) -> list[dict]:
        if not documents:
            return []

        pairs = [(query, doc.get("content", "") or "") for doc in documents]
        scores = self._model.predict(pairs, show_progress_bar=len(pairs) > 20)

        scored = []
        for doc, score in zip(documents, scores):
            doc_copy = dict(doc)
            doc_copy["rerank_score"] = round(float(score), 4)
            scored.append(doc_copy)

        scored.sort(key=lambda d: d.get("rerank_score", 0), reverse=True)
        return scored[:top_k]


# 模块级单例
_reranker: RerankerService | None = None


def get_reranker() -> RerankerService:
    """根据配置创建或返回 Reranker 单例。"""
    global _reranker
    if _reranker is None:
        cfg = Settings.load()
        provider = cfg.reranker_provider
        if provider == "bailian":
            _reranker = BailianReranker(
                model_name=cfg.reranker_model_name,
                base_url=cfg.reranker_base_url,
                api_key=cfg.reranker_api_key,
                timeout=cfg.reranker_timeout,
            )
        elif provider == "local":
            _reranker = LocalReranker(
                model_name=cfg.reranker_model_name,
                device=cfg.embedding_device,
            )
        else:
            raise KnowledgeBaseError(f"不支持的 Reranker provider: {provider}")
    return _reranker
