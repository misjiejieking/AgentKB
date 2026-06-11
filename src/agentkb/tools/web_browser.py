"""工具：browse_web——网页浏览（获取网页完整内容并提取正文）。

用于需要深度阅读网页内容的场景，比 search_web（返回摘要）更深入。
"""

from __future__ import annotations

import asyncio
import ipaddress
import re
import socket
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
from pydantic import BaseModel, Field

from agentkb.tools.base import BaseTool, ToolResult


class BrowseInput(BaseModel):
    url: str = Field(description="要浏览的网页 URL")
    extract_mode: str = Field(default="text", description="提取模式: text（纯文本）或 html（原始HTML）")


class WebBrowserTool(BaseTool):
    """网页浏览工具——获取网页全文并提取正文。"""

    _MAX_REDIRECTS = 5
    _MAX_RESPONSE_BYTES = 2 * 1024 * 1024

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

    async def _execute(
        self,
        url: str = "",
        extract_mode: str = "text",
        **kwargs: Any,
    ) -> ToolResult:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

        try:
            async with httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
            ) as client:
                current_url = url
                html = ""
                for _ in range(self._MAX_REDIRECTS + 1):
                    await self._validate_public_url(current_url)
                    async with client.stream(
                        "GET",
                        current_url,
                        headers=headers,
                    ) as resp:
                        if resp.is_redirect:
                            location = resp.headers.get("location")
                            if not location:
                                return ToolResult(
                                    tool_name=self.name,
                                    success=False,
                                    error="网页重定向缺少 Location",
                                )
                            current_url = urljoin(current_url, location)
                            continue

                        if resp.status_code != 200:
                            return ToolResult(
                                tool_name=self.name,
                                success=False,
                                error=f"HTTP {resp.status_code}: 无法访问该网页",
                            )

                        content = bytearray()
                        async for chunk in resp.aiter_bytes():
                            content.extend(chunk)
                            if len(content) > self._MAX_RESPONSE_BYTES:
                                return ToolResult(
                                    tool_name=self.name,
                                    success=False,
                                    error="网页响应体超过 2MB 限制",
                                )
                        encoding = resp.encoding or "utf-8"
                        html = bytes(content).decode(encoding, errors="replace")
                        break
                else:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        error=f"网页重定向超过 {self._MAX_REDIRECTS} 次",
                    )

                if not html:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        error="网页响应内容为空",
                    )

                if extract_mode == "html":
                    return ToolResult(
                        tool_name=self.name, success=True,
                        data={
                            "url": current_url,
                            "html": html[:8192],
                            "length": len(html),
                        },
                    )

                # 提取正文
                text = self._extract_text(html)
                return ToolResult(
                    tool_name=self.name, success=True,
                    data={
                        "url": current_url,
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
    async def _validate_public_url(url: str) -> None:
        """拒绝凭据、私网、环回、链路本地和保留地址。"""
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("仅允许访问有效的 HTTP/HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("URL 不允许包含用户名或密码")
        if parsed.port not in {None, 80, 443}:
            raise ValueError("仅允许访问 80 或 443 端口")
        if parsed.hostname.lower() == "localhost":
            raise ValueError("不允许访问本机或私有网络地址")

        try:
            addresses = [ipaddress.ip_address(parsed.hostname)]
        except ValueError:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            records = await asyncio.to_thread(
                socket.getaddrinfo,
                parsed.hostname,
                port,
                type=socket.SOCK_STREAM,
            )
            addresses = list(
                {
                    ipaddress.ip_address(record[4][0])
                    for record in records
                }
            )

        if not addresses or any(not address.is_global for address in addresses):
            raise ValueError("不允许访问本机或私有网络地址")

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
