"""工具基类：统一错误处理、计时、LangChain 适配。"""

from __future__ import annotations

import asyncio
import json
import time
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import BaseModel
from loguru import logger


class ToolResult:
    """标准化的工具执行结果。"""

    def __init__(
        self,
        tool_name: str,
        success: bool,
        data: Any = None,
        error: str | None = None,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.tool_name = tool_name
        self.success = success
        self.data = data
        self.error = error
        self.elapsed_ms = elapsed_ms

    def to_json(self) -> str:
        """将结果序列化为 JSON 字符串，供 ToolMessage 使用。"""
        return json.dumps(
            {
                "tool": self.tool_name,
                "success": self.success,
                "data": self.data if self.success else None,
                "error": self.error if not self.success else None,
                "elapsed_ms": round(self.elapsed_ms, 1),
            },
            ensure_ascii=False,
            indent=2,
        )


class BaseTool(ABC):
    """工具抽象基类——子类必须实现 name、description、_execute。"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    def args_schema(self) -> type[BaseModel] | None:
        """可选：Pydantic 模型，定义工具参数 schema。"""
        return None

    @abstractmethod
    async def _execute(self, **kwargs) -> ToolResult:
        """子类实现具体的工具逻辑。"""

    async def execute(self, **kwargs) -> ToolResult:
        """统一执行入口：计时 + 异常捕获，始终返回 ToolResult（不抛出异常）。"""
        logger.info(f"工具 [{self.name}] 被调用，参数: {kwargs}")
        start = time.perf_counter()
        try:
            result = await self._execute(**kwargs)
            result.elapsed_ms = (time.perf_counter() - start) * 1000
            result.tool_name = self.name
        except Exception as exc:
            elapsed = (time.perf_counter() - start) * 1000
            logger.error(f"工具 [{self.name}] 执行失败: {exc}")
            result = ToolResult(
                tool_name=self.name,
                success=False,
                error=str(exc),
                elapsed_ms=elapsed,
            )
        logger.info(
            f"工具 [{self.name}] 完成 — 成功={result.success}, "
            f"耗时={result.elapsed_ms:.0f}ms"
        )
        return result

    def to_langchain_tool(self) -> StructuredTool:
        """转换为 LangChain StructuredTool，用于 ChatModel.bind_tools()。"""

        async def _coro(**kwargs: Any) -> str:
            result = await self.execute(**kwargs)
            return result.to_json()

        def _sync(**kwargs: Any) -> str:
            return asyncio.run(_coro(**kwargs))

        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            func=_sync,
            coroutine=_coro,
            args_schema=self.args_schema,
        )
