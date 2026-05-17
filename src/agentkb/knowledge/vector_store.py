"""Qdrant vector store wrapper — local file mode."""

from __future__ import annotations

import uuid
from pathlib import Path

from loguru import logger
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from agentkb.utils.exceptions import KnowledgeBaseError


class VectorStore:
    """Qdrant local vector store for knowledge chunks."""

    def __init__(
        self,
        path: str = "data/vectors",
        collection_name: str = "knowledge",
        vector_size: int = 1024,
    ) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._collection = collection_name
        self._vector_size = vector_size

        self._client = QdrantClient(path=str(self._path))
        self._ensure_collection()
        logger.info(f"VectorStore ready: path={self._path}, collection={collection_name}")

    def _ensure_collection(self) -> None:
        """Create collection if it doesn't exist."""
        exists = self._client.collection_exists(self._collection)
        if not exists:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(
                    size=self._vector_size,
                    distance=qmodels.Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {self._collection}")

    # ── CRUD ─────────────────────────────────────────────────

    def upsert(
        self,
        points: list[dict],
    ) -> None:
        """Insert or update points. Each dict: {id, vector, payload}."""
        qpoints = []
        for p in points:
            qpoints.append(
                qmodels.PointStruct(
                    id=p.get("id", uuid.uuid4().hex),
                    vector=p["vector"],
                    payload=p.get("payload", {}),
                )
            )
        try:
            self._client.upsert(
                collection_name=self._collection,
                points=qpoints,
                wait=True,
            )
            logger.debug(f"Upserted {len(qpoints)} points")
        except Exception as e:
            raise KnowledgeBaseError(f"Vector upsert failed: {e}") from e

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
        score_threshold: float = 0.3,
    ) -> list[qmodels.ScoredPoint]:
        """Semantic search in the collection."""
        try:
            results = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as e:
            raise KnowledgeBaseError(f"Vector search failed: {e}") from e
        return results

    def delete_by_file_id(self, file_id: str) -> int:
        """Delete all points belonging to a specific file_id. Returns count."""
        try:
            result = self._client.delete(
                collection_name=self._collection,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="file_id",
                                match=qmodels.MatchValue(value=file_id),
                            )
                        ]
                    )
                ),
            )
            count = result.status.completed if result.status else 0
            logger.info(f"Deleted {count} points for file_id={file_id}")
            return count
        except Exception as e:
            raise KnowledgeBaseError(f"Vector delete failed: {e}") from e

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count


# Module-level singleton
_vs: VectorStore | None = None


def get_vector_store(
    path: str = "data/vectors",
    collection_name: str = "knowledge",
    vector_size: int = 1024,
) -> VectorStore:
    global _vs
    if _vs is None:
        _vs = VectorStore(path=path, collection_name=collection_name, vector_size=vector_size)
    return _vs
