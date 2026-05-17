"""文本切块服务，使用递归字符分割器。"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from loguru import logger


class TextSplitter:
    """将文档切分为带重叠的语义块，供向量化使用。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        """切分文档，保留原始 metadata 并附加 chunk_index。"""
        chunks = self._splitter.split_documents(documents)
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
        logger.debug(f"切分完成: {len(documents)} 个文档 → {len(chunks)} 个块")
        return chunks
