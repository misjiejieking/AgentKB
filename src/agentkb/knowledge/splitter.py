"""文本分块兼容层——重新导出 chunker 模块的统一入口。"""

from __future__ import annotations

from agentkb.knowledge.chunker import TextSplitter, ChunkingStrategy

__all__ = ["TextSplitter", "ChunkingStrategy"]
