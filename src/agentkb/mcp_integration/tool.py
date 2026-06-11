"""将远程 MCP 工具适配为 AgentKB 工具。"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from langchain_core.tools import StructuredTool

from agentkb.tools.base import BaseTool, ToolResult

if TYPE_CHECKING:
    from agentkb.mcp_integration.manager import MCPManager


class MCPTool(BaseTool):
    def __init__(
        self,
        *,
        manager: MCPManager,
        server_id: str,
        server_name: str,
        remote_name: str,
        local_name: str,
        description: str,
        input_schema: dict[str, Any],
        requires_confirmation: bool,
    ) -> None:
        self.manager = manager
        self.server_id = server_id
        self.server_name = server_name
        self.remote_name = remote_name
        self._name = local_name
        self._description = description
        self.input_schema = input_schema
        self._requires_confirmation = requires_confirmation

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    @property
    def confirmation_message(self) -> str:
        return (
            f"MCP 服务 {self.server_name} 的工具 {self.remote_name} "
            "可能修改外部状态，必须由你确认。"
        )

    async def _execute(self, **kwargs: Any) -> ToolResult:
        return await self.manager.call_tool(
            self.server_id,
            self.remote_name,
            kwargs,
        )

    def to_langchain_tool(self) -> StructuredTool:
        async def _coro(**kwargs: Any) -> str:
            return (await self.execute(**kwargs)).to_json()

        return StructuredTool.from_function(
            name=self.name,
            description=self.description,
            coroutine=_coro,
            args_schema=self.input_schema,
        )
