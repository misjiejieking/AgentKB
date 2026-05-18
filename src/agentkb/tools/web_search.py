"""工具：search_web — 联网搜索，优先 duckduckgo_search，失败时回退到 DuckDuckGo Lite HTML。"""

from __future__ import annotations

import re
from urllib.parse import quote_plus

import httpx
from pydantic import BaseModel, Field
from loguru import logger

from agentkb.tools.base import BaseTool, ToolResult


class WebSearchInput(BaseModel):
    """联网搜索的输入参数。"""
    query: str = Field(description="搜索关键词")


class WebSearchTool(BaseTool):
    """联网搜索：优先 duckduckgo_search 库，失败时回退到 DuckDuckGo Lite 页面解析。"""

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
        # 方案一：尝试 duckduckgo_search 库
        results = await self._search_with_lib(query)
        if results is not None:
            return self._format_results(query, results)

        # 方案二：回退到 DuckDuckGo Lite HTML 解析
        logger.info(f"duckduckgo_search 库失败，回退到 DDG Lite HTML")
        results = await self._search_with_lite(query)
        if results is not None:
            return self._format_results(query, results)

        return ToolResult(
            tool_name=self.name,
            success=False,
            error="联网搜索暂时不可用，请稍后重试",
        )

    # ── 方案一：duckduckgo_search 库 ────────────────────────────

    async def _search_with_lib(self, query: str) -> list[dict] | None:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            return None

        try:
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=self._max_results))
            if not raw:
                return None
            return [
                {"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")}
                for r in raw
            ]
        except Exception as e:
            logger.warning(f"duckduckgo_search 库异常: {e}")
            return None

    # ── 方案二：DuckDuckGo Lite HTML 解析 ────────────────────────

    async def _search_with_lite(self, query: str) -> list[dict] | None:
        """直接请求 DuckDuckGo Lite 页面并解析结果。"""
        url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        try:
            async with httpx.AsyncClient(timeout=self._timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers=headers)
                if resp.status_code != 200:
                    logger.warning(f"DDG Lite 返回 {resp.status_code}")
                    return None
                return self._parse_lite_html(resp.text)
        except Exception as e:
            logger.warning(f"DDG Lite 请求失败: {e}")
            return None

    @staticmethod
    def _parse_lite_html(html: str) -> list[dict]:
        """解析 DuckDuckGo Lite 的 HTML 结果。"""
        results = []
        # Lite 页面结构：<a rel="nofollow" href="..."> 标题 </a> 后面跟 <span class="link-text">url</span> 和描述文本
        # 用正则匹配每个结果块
        pattern = re.compile(
            r'<a[^>]*rel="nofollow"[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
            r'.*?<span class="link-text">[^<]*</span>\s*(.*?)(?=<(?:a\s+rel="nofollow"|br\s*/?\s*>\s*<br|$))',
            re.DOTALL,
        )
        for m in pattern.finditer(html):
            url = m.group(1)
            title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
            snippet = re.sub(r"<[^>]+>", "", m.group(3)).strip()
            snippet = re.sub(r"\s+", " ", snippet)
            if title and url.startswith("http"):
                results.append({"title": title, "snippet": snippet[:512], "url": url})
                if len(results) >= 5:
                    break

        # 如果上面没匹配到，用更宽松的正则
        if not results:
            link_pattern = re.compile(
                r'<a[^>]*href="(https?://[^"]*)"[^>]*>(.+?)</a>',
                re.DOTALL,
            )
            matches = link_pattern.findall(html)
            for url, title_raw in matches[:5]:
                title = re.sub(r"<[^>]+>", "", title_raw).strip()
                if title and "duckduckgo" not in url:
                    results.append({"title": title, "snippet": "", "url": url})

        return results

    # ── 格式化输出 ───────────────────────────────────────────────

    def _format_results(self, query: str, results: list[dict]) -> ToolResult:
        if not results:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={"query": query, "results": [], "hint": "未找到相关网页"},
            )

        formatted = []
        for r in results[:self._max_results]:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", "")[:512],
                "url": r.get("url", ""),
            })

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={"query": query, "results": formatted, "total": len(formatted)},
        )
