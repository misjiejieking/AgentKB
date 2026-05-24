"""工具：search_knowledge_base — 混合检索 + 重排序。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentkb.config.settings import Settings
from agentkb.tools.base import BaseTool, ToolResult
from agentkb.knowledge.retriever import get_retriever


class KnowledgeSearchInput(BaseModel):
    """知识库检索的输入参数。"""
    query: str = Field(description="搜索查询词，尽量使用关键词")


class KnowledgeSearchTool(BaseTool):
    """混合检索（dense + BM25 → RRF 融合 → Reranker 精排）本地知识库。"""

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "检索本地知识库中的文档内容。"
            "适合回答需要从用户上传的文件中查找信息的问题，"
            "例如公司制度、学习笔记、文档内容等。"
            "输入应为简练的搜索关键词或问题。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return KnowledgeSearchInput

    async def _execute(self, query: str) -> ToolResult:
        cfg = Settings.load()
        retriever = get_retriever()

        # 1. 混合检索 → 候选集
        candidates = retriever.retrieve(query)

        if not candidates:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"query": query, "results": [], "hint": "知识库中没有找到相关内容"},
            )

        # 2. 按 RRF 分数排序（reranker 暂不使用：百炼免费额度已用完，本地 CPU 推理太慢）
        candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        ranked = candidates[:cfg.retrieval_final_k]

        # 3. 格式化输出：优先使用 parent_content 保证完整上下文
        results = []
        for r in ranked:
            content = r.get("parent_content") or r.get("content", "")
            chunk_meta = r.get("chunk_metadata")
            if isinstance(chunk_meta, str):
                import json
                try:
                    chunk_meta = json.loads(chunk_meta)
                except json.JSONDecodeError:
                    chunk_meta = {}

            filename = (chunk_meta or {}).get("filename", "") or r.get("file_id", "unknown")
            results.append({
                "content": content,
                "filename": filename,
                "score": r.get("rerank_score", 0),
            })

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": query, "results": results, "total": len(results)},
        )
