"""工具：browse_web——网页浏览（获取网页完整内容并提取正文）。

用于需要深度阅读网页内容的场景，比 search_web（返回摘要）更深入。
"""

from __future__ import annotations

import re

import httpx
from pydantic import BaseModel, Field
from loguru import logger

from agentkb.tools.base import BaseTool, ToolResult


class BrowseInput(BaseModel):
    url: str = Field(description="要浏览的网页 URL")
    extract_mode: str = Field(default="text", description="提取模式: text（纯文本）或 html（原始HTML）")


class WebBrowserTool(BaseTool):
    """网页浏览工具——获取网页全文并提取正文。"""

    def __init__(self, timeout: int = 15) -> None:
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "browse_web"

    @property
    def description(self) -> str:
        return (
            "浏览指定网页并提取正文内容。"
            "适合需要深度阅读网页文章的完整内容时使用。"
            "输入为网页 URL。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return BrowseInput

    async def _execute(self, url: str, extract_mode: str = "text") -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

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
                    return ToolResult(
                        tool_name=self.name, success=False,
                        error=f"HTTP {resp.status_code}: 无法访问该网页",
                    )

                html = resp.text

                if extract_mode == "html":
                    return ToolResult(
                        tool_name=self.name, success=True,
                        data={"url": url, "html": html[:8192], "length": len(html)},
                    )

                # 提取正文
                text = self._extract_text(html)
                return ToolResult(
                    tool_name=self.name, success=True,
                    data={
                        "url": url,
                        "title": self._extract_title(html),
                        "text": text[:4096],
                        "text_length": len(text),
                    },
                )

        except httpx.TimeoutException:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"请求超时（{self._timeout}秒）",
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"浏览失败: {e}",
            )

    @staticmethod
    def _extract_title(html: str) -> str:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip()[:200] if m else ""

    @staticmethod
    def _extract_text(html: str) -> str:
        """简单 HTML → 纯文本提取。"""
        # 移除 script/style
        html = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", "", html, flags=re.DOTALL | re.IGNORECASE)
        # 移除标签
        text = re.sub(r"<[^>]+>", " ", html)
        # 合并空白
        text = re.sub(r"&nbsp;", " ", text)
        text = re.sub(r"\s+", " ", text)
        # 解码实体
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&quot;", '"').replace("&#39;", "'")
        return text.strip()
