"""知识图谱相邻关系查询工具。"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel, Field

from agentkb.storage.pg_database import get_db
from agentkb.tools.base import BaseTool, ToolResult


class KnowledgeGraphQueryInput(BaseModel):
    query: str = Field(description="要查询的实体名称、关系词或关联问题")


class KnowledgeGraphQueryTool(BaseTool):
    @property
    def name(self) -> str:
        return "query_knowledge_graph"

    @property
    def description(self) -> str:
        return (
            "查询用户知识库中的实体关系图谱。"
            "适合回答人物、组织、系统、项目和概念之间的关系、依赖、归属或组成问题。"
            "图谱结果包含原始文件和证据，不能用于实时互联网信息。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return KnowledgeGraphQueryInput

    async def _execute(
        self,
        query: str = "",
        **kwargs: Any,
    ) -> ToolResult:
        graph = await asyncio.to_thread(
            get_db().search_knowledge_graph,
            query,
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "query": query,
                **graph,
                "relation_count": len(graph["edges"]),
            },
        )
