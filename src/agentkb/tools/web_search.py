"""工具：search_web — DuckDuckGo 联网搜索。"""

from __future__ import annotations

from pydantic import BaseModel, Field
from loguru import logger

from agentkb.tools.base import BaseTool, ToolResult


class WebSearchInput(BaseModel):
    """联网搜索的输入参数。"""
    query: str = Field(description="搜索关键词")


class WebSearchTool(BaseTool):
    """通过 DuckDuckGo 进行联网搜索，无需 API Key。"""

    def __init__(self, max_results: int = 5, timeout: int = 10) -> None:
        self._max_results = max_results
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "search_web"

    @property
    def description(self) -> str:
        return (
            "联网搜索互联网获取最新信息。"
            "适合回答需要实时信息、新闻、或本地知识库没有的内容时使用。"
            "输入应为简练的搜索关键词。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return WebSearchInput

    async def _execute(self, query: str) -> ToolResult:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="duckduckgo_search 包未安装",
            )

        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self._max_results))
        except Exception as e:
            logger.error(f"DuckDuckGo 搜索失败: {e}")
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"搜索请求失败: {e}",
            )

        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"query": query, "results": [], "hint": "未找到相关网页"},
            )

        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", "")[:512],
                "url": r.get("href", ""),
            })

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": query, "results": formatted, "total": len(formatted)},
        )
