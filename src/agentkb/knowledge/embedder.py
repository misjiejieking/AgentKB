"""BGE-M3 向量化服务，基于 sentence-transformers，支持 GPU 自动检测 + FP16 推理。"""

from __future__ import annotations

import os

from loguru import logger

# 本地模型可能只有 PyTorch 权重；禁止 Transformers 后台联网创建转换 PR。
os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

from sentence_transformers import SentenceTransformer

from agentkb.utils.exceptions import EmbeddingError


def _auto_device() -> str:
    """自动检测最优推理设备：CUDA > MPS > CPU。"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class EmbedderService:
    """BGE-M3 向量编码器封装（1024 维），支持批量编码与查询编码。"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None:
        if device == "auto":
            device = _auto_device()
        logger.info(f"正在加载向量化模型: {model_name}（设备={device}）")
        try:
            self._model = SentenceTransformer(
                model_name, device=device, trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            raise EmbeddingError(f"无法加载向量模型 '{model_name}': {e}") from e

        # CUDA 设备启用 FP16 推理以降低显存占用、提升速度
        if device == "cuda":
            try:
                self._model.half()
                logger.info("已启用 FP16 半精度推理")
            except Exception as e:
                logger.warning(f"FP16 启用失败，回退到 FP32: {e}")

        self._normalize = normalize
        self._batch_size = batch_size
        self._dimension = self._model.get_embedding_dimension()
        logger.info(f"向量模型就绪 — 维度={self._dimension}, 批次={batch_size}")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码文档文本，返回向量列表。"""
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=self._batch_size,
                normalize_embeddings=self._normalize,
                show_progress_bar=len(texts) > 100,
            )
        except Exception as e:
            raise EmbeddingError(f"文档向量化失败: {e}") from e
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        """编码单条查询文本，返回向量。"""
        try:
            embedding = self._model.encode(
                [query],
                normalize_embeddings=self._normalize,
            )
        except Exception as e:
            raise EmbeddingError(f"查询向量化失败: {e}") from e
        return embedding[0].tolist()


# 模块级单例：向量模型加载较重，全局复用
_embedder: EmbedderService | None = None


def get_embedder(
    model_name: str = "BAAI/bge-m3",
    device: str = "cpu",
    normalize: bool = True,
    batch_size: int = 32,
) -> EmbedderService:
    """获取或创建 EmbedderService 单例。"""
    global _embedder
    if _embedder is None:
        _embedder = EmbedderService(
            model_name=model_name,
            device=device,
            normalize=normalize,
            batch_size=batch_size,
        )
    return _embedder
