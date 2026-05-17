"""工具：search_knowledge_base — 本地知识库语义检索。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentkb.tools.base import BaseTool, ToolResult
from agentkb.knowledge.embedder import get_embedder
from agentkb.knowledge.vector_store import get_vector_store


class KnowledgeSearchInput(BaseModel):
    """知识库检索的输入参数。"""
    query: str = Field(description="搜索查询词，尽量使用关键词")


class KnowledgeSearchTool(BaseTool):
    """检索用户本地知识库中的文档内容。"""

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
        embedder = get_embedder()
        vector_store = get_vector_store()

        query_vector = embedder.embed_query(query)
        hits = vector_store.search(query_vector, limit=5, score_threshold=0.3)

        if not hits:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [],
                    "hint": "知识库中没有找到相关内容",
                },
            )

        results = []
        for hit in hits:
            results.append({
                "content": hit.payload.get("content", "")[:1024],
                "filename": hit.payload.get("filename", "unknown"),
                "score": round(hit.score, 4),
            })

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": query, "results": results, "total": len(results)},
        )
