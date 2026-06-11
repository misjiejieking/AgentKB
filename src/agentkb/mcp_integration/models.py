"""MCP 配置模型与安全策略。"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, model_validator

_SERVER_NAME = re.compile(r"^[a-z][a-z0-9_]{2,39}$")


class MCPServerCreate(BaseModel):
    name: str = Field(min_length=3, max_length=40)
    transport: Literal["stdio", "streamable_http"]
    command: str | None = Field(default=None, min_length=1)
    args: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    confirmation_policy: Literal["always", "writes", "never"] = "writes"

    @model_validator(mode="after")
    def validate_transport(self) -> MCPServerCreate:
        if not _SERVER_NAME.fullmatch(self.name):
            raise ValueError("服务标识仅允许小写字母、数字和下划线，且必须以字母开头")
        if self.transport == "stdio":
            if not self.command or self.url:
                raise ValueError("stdio 服务必须配置 command，且不能配置 url")
        elif not self.url or self.command:
            raise ValueError("streamable_http 服务必须配置 url，且不能配置 command")
        if self.url and not self.url.startswith(("http://", "https://")):
            raise ValueError("MCP URL 必须使用 http 或 https")
        return self


class MCPToolStatusRequest(BaseModel):
    enabled: bool


def requires_confirmation(policy: str, annotations: dict) -> bool:
    """未明确声明只读的工具按写操作处理。"""
    if policy == "always":
        return True
    if policy == "never":
        return False
    return annotations.get("readOnlyHint") is not True
