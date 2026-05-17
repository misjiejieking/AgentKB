"""ToolRegistry 工具注册表——单例模式，管理所有可用工具。"""

from __future__ import annotations

from langchain_core.tools import StructuredTool
from loguru import logger

from agentkb.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """工具注册表单例。先 register() 注册工具，再 get_langchain_tools() 获取列表。"""

    _instance: ToolRegistry | None = None
    _tools: dict[str, BaseTool]

    def __new__(cls) -> ToolRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._tools = {}
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """重置单例（测试用途）。"""
        cls._instance = None

    def register(self, tool: BaseTool) -> None:
        """注册一个工具，以 tool.name 为键。"""
        self._tools[tool.name] = tool
        logger.debug(f"已注册工具: {tool.name}")

    def unregister(self, name: str) -> None:
        """移除指定名称的工具。"""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """根据名称获取工具实例。"""
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """返回所有已注册工具实例的列表。"""
        return list(self._tools.values())

    def get_langchain_tools(self) -> list[StructuredTool]:
        """将所有已注册工具转为 LangChain StructuredTool 列表。"""
        return [t.to_langchain_tool() for t in self._tools.values()]

    async def execute(self, name: str, **kwargs) -> ToolResult:
        """按名称执行一个已注册的工具。"""
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                error=f"工具 '{name}' 未注册",
            )
        return await tool.execute(**kwargs)
