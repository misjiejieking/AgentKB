"""多策略分块——滑动窗口、语义分块、父子分块，含自动策略选择。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from agentkb.config.settings import Settings


class ChunkingStrategy(Enum):
    NONE = auto()
    SLIDING_WINDOW = auto()
    SEMANTIC = auto()
    PARENT_CHILD = auto()


@dataclass
class DocumentFeatures:
    """文档特征向量，用于自动策略选择。"""
    total_chars: int = 0
    avg_paragraph_length: float = 0.0
    paragraph_length_std: float = 0.0
    header_density: float = 0.0
    short_line_ratio: float = 0.0
    section_count: int = 0
    paragraph_count: int = 0


# ══════════════════════════════════════════════════════════════
#  BaseChunker
# ══════════════════════════════════════════════════════════════

class BaseChunker(ABC):
    @abstractmethod
    def split(self, documents: list[Document]) -> list[Document]:
        ...


# ══════════════════════════════════════════════════════════════
#  滑动窗口分块
# ══════════════════════════════════════════════════════════════

class SlidingWindowChunker(BaseChunker):
    """基于 RecursiveCharacterTextSplitter 的滑动窗口分块。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "！", "？", " ", ""],
        )

    def split(self, documents: list[Document]) -> list[Document]:
        chunks = self._splitter.split_documents(documents)
        for i, c in enumerate(chunks):
            c.metadata["chunk_index"] = i
            c.metadata["strategy"] = "sliding_window"
        logger.debug(f"滑动窗口: {len(documents)} doc → {len(chunks)} chunks")
        return chunks


# ══════════════════════════════════════════════════════════════
#  语义分块
# ══════════════════════════════════════════════════════════════

class SemanticChunker(BaseChunker):
    """基于句子 embedding 相似度骤降点切分的语义分块。"""

    def __init__(self, embedder, threshold: float = 0.5) -> None:
        self._embedder = embedder
        self._threshold = threshold

    def split(self, documents: list[Document]) -> list[Document]:
        all_sentences: list[tuple[str, str, int]] = []  # (text, source, doc_offset)
        for doc_idx, doc in enumerate(documents):
            source = doc.metadata.get("source", "unknown")
            sentences = self._split_sentences(doc.page_content)
            for s in sentences:
                s = s.strip()
                if s:
                    all_sentences.append((s, source, doc_idx))

        if len(all_sentences) <= 1:
            # 句子太少，退回滑动窗口
            logger.debug("句子过少，退回到滑动窗口")
            return SlidingWindowChunker().split(documents)

        # 批量编码句子
        texts = [s[0] for s in all_sentences]
        embeddings = self._embedder.embed_documents(texts)
        embeddings_np = np.array(embeddings)

        # 计算相邻句子相似度
        similarities = []
        for i in range(1, len(embeddings_np)):
            sim = float(np.dot(embeddings_np[i - 1], embeddings_np[i]) /
                        (np.linalg.norm(embeddings_np[i - 1]) * np.linalg.norm(embeddings_np[i]) + 1e-8))
            similarities.append(sim)

        # 找到切分点：相似度低于阈值的边界
        cut_points = [0]
        for i, sim in enumerate(similarities):
            if sim < self._threshold:
                cut_points.append(i + 1)
        cut_points.append(len(all_sentences))

        # 按切分点合并句子为 chunk
        chunks = []
        chunk_idx = 0
        for cp_start, cp_end in zip(cut_points[:-1], cut_points[1:]):
            if cp_start >= cp_end:
                continue
            chunk_text = " ".join(all_sentences[i][0] for i in range(cp_start, cp_end))
            source = all_sentences[cp_start][1]
            chunks.append(Document(
                page_content=chunk_text,
                metadata={"source": source, "chunk_index": chunk_idx, "strategy": "semantic"},
            ))
            chunk_idx += 1

        logger.debug(f"语义分块: {len(all_sentences)} 句话 → {len(chunks)} chunks")
        return chunks

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """简单句子切分（中文句号/问号/感叹号 + 英文标点 + 换行）。"""
        import re
        raw = re.split(r'(?<=[。！？.!?\n])\s*', text)
        return [s.strip() for s in raw if s.strip()]


# ══════════════════════════════════════════════════════════════
#  父子分块
# ══════════════════════════════════════════════════════════════

class ParentChildChunker(BaseChunker):
    """父块（大上下文）含多子块（检索命中单元）；检索命中子块 → 返回父块 content。"""

    def __init__(self, parent_size: int = 1024, child_size: int = 256) -> None:
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size, chunk_overlap=min(100, parent_size // 10),
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size, chunk_overlap=min(50, child_size // 5),
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )
        self._parent_size = parent_size
        self._child_size = child_size

    def split(self, documents: list[Document]) -> list[Document]:
        parent_chunks = self._parent_splitter.split_documents(documents)

        result = []
        for p_idx, parent in enumerate(parent_chunks):
            parent_content = parent.page_content
            parent_source = parent.metadata.get("source", "unknown")

            # 父块内部切子块
            child_docs = self._child_splitter.split_documents([parent])
            for c_idx, child in enumerate(child_docs):
                child.metadata["parent_content"] = parent_content
                child.metadata["chunk_index"] = len(result)
                child.metadata["parent_index"] = p_idx
                child.metadata["child_index"] = c_idx
                child.metadata["source"] = parent_source
                child.metadata["strategy"] = "parent_child"
                result.append(child)

        logger.debug(f"父子分块: {len(parent_chunks)} 父块 → {len(result)} 子块 "
                     f"(parent={self._parent_size}, child={self._child_size})")
        return result


# ══════════════════════════════════════════════════════════════
#  策略自动选择
# ══════════════════════════════════════════════════════════════

class ChunkingStrategySelector:
    """根据文档特征自动选出最佳分块策略。"""

    @staticmethod
    def extract_features(documents: list[Document]) -> DocumentFeatures:
        import re

        full_text = "\n\n".join(d.page_content for d in documents)
        total = len(full_text)
        paragraphs = [p.strip() for p in full_text.split("\n\n") if p.strip()]
        para_count = len(paragraphs)
        para_lengths = [len(p) for p in paragraphs]

        # 标题密度：Markdown 标题行 / 总行数
        lines = full_text.split("\n")
        line_count = len(lines) if lines else 1
        header_count = len(re.findall(r"^#{1,6}\s", full_text, re.MULTILINE))
        header_density = header_count / line_count

        # 短行占比
        short_count = sum(1 for line in lines if len(line.strip()) < 60 and line.strip())
        short_line_ratio = short_count / line_count

        # 章节数（h1/h2）
        section_count = len(re.findall(r"^#{1,2}\s[^#]", full_text, re.MULTILINE))

        avg_len = float(np.mean(para_lengths)) if para_lengths else 0.0
        std_len = float(np.std(para_lengths)) if para_lengths else 0.0

        return DocumentFeatures(
            total_chars=total,
            avg_paragraph_length=avg_len,
            paragraph_length_std=std_len,
            header_density=header_density,
            short_line_ratio=short_line_ratio,
            section_count=section_count,
            paragraph_count=para_count,
        )

    @staticmethod
    def select(features: DocumentFeatures) -> ChunkingStrategy:
        total = features.total_chars

        # 极短文 — 不分块
        if total < 2000:
            logger.info(f"策略选择: none（文档仅 {total} 字符）")
            return ChunkingStrategy.NONE

        # 结构清晰的文档 — 语义分块
        if features.header_density > 0.05 and features.section_count >= 3:
            logger.info(f"策略选择: semantic（header_density={features.header_density:.3f}, "
                        f"sections={features.section_count}）")
            return ChunkingStrategy.SEMANTIC

        # 段落长短差异大 — 父子分块（有长上下文段落）
        if features.avg_paragraph_length > 300 and features.paragraph_length_std > 200:
            logger.info(f"策略选择: parent_child（avg_para={features.avg_paragraph_length:.0f}, "
                        f"std={features.paragraph_length_std:.0f}）")
            return ChunkingStrategy.PARENT_CHILD

        # 短行居多（对话/FAQ/列表） — 语义分块
        if features.short_line_ratio > 0.4:
            logger.info(f"策略选择: semantic（short_line_ratio={features.short_line_ratio:.2f}）")
            return ChunkingStrategy.SEMANTIC

        # 默认 — 滑动窗口
        logger.info(f"策略选择: sliding_window（默认）")
        return ChunkingStrategy.SLIDING_WINDOW


# ══════════════════════════════════════════════════════════════
#  TextSplitter — 统一分块入口
# ══════════════════════════════════════════════════════════════

class TextSplitter:
    """统一分块入口：自动分析文档特征 → 选择策略 → 执行分块。"""

    def __init__(self, embedder=None) -> None:
        cfg = Settings.load()
        self._selector = ChunkingStrategySelector()
        self._embedder = embedder
        self._chunkers = {
            ChunkingStrategy.SLIDING_WINDOW: SlidingWindowChunker(
                chunk_size=cfg.chunking_sliding_size,
                chunk_overlap=cfg.chunking_sliding_overlap,
            ),
            ChunkingStrategy.SEMANTIC: SemanticChunker(
                embedder=embedder if embedder else None,
                threshold=cfg.chunking_semantic_threshold,
            ),
            ChunkingStrategy.PARENT_CHILD: ParentChildChunker(
                parent_size=cfg.chunking_parent_size,
                child_size=cfg.chunking_child_size,
            ),
        }

    def split(self, documents: list[Document]) -> list[Document]:
        features = ChunkingStrategySelector.extract_features(documents)
        strategy = ChunkingStrategySelector.select(features)

        if strategy == ChunkingStrategy.NONE:
            for i, d in enumerate(documents):
                d.metadata["chunk_index"] = i
                d.metadata["strategy"] = "none"
            return documents

        chunker = self._chunkers[strategy]
        if chunker is None and strategy == ChunkingStrategy.SEMANTIC:
            # embedder 不可用时退回滑动窗口
            logger.warning("embedder 不可用，语义分块退回滑动窗口")
            chunker = self._chunkers[ChunkingStrategy.SLIDING_WINDOW]

        return chunker.split(documents)
