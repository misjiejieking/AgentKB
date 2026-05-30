"""MCP Client——支持 stdio 传输的本地 MCP Server，自动注册工具到 ToolRegistry。"""

from __future__ import annotations

import asyncio
import json
import subprocess
from typing import Any

from loguru import logger

from agentkb.tools.base import BaseTool, ToolResult
from agentkb.tools.registry import ToolRegistry


class MCPTool(BaseTool):
    """MCP Server 提供的工具代理——执行时转发到 MCP Server。"""

    def __init__(self, mcp_name: str, tool_name: str, description: str, input_schema: dict,
                 mcp_instance: MCPClientBridge) -> None:
        self._mcp_name = mcp_name
        self._tool_name = tool_name
        self._description = description
        self._input_schema = input_schema
        self._mcp = mcp_instance

    @property
    def name(self) -> str:
        return f"mcp_{self._mcp_name}_{self._tool_name}"[:64]

    @property
    def description(self) -> str:
        return f"[MCP:{self._mcp_name}] {self._description}"

    @property
    def args_schema(self) -> Any:
        return None  # MCP 工具参数动态适配

    async def _execute(self, **kwargs) -> ToolResult:
        try:
            result = await self._mcp.call_tool(self._tool_name, kwargs)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data=result,
            )
        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error=f"MCP 工具执行失败: {e}",
            )


class MCPClientBridge:
    """单个 MCP Server 的 stdio 桥接——管理进程生命周期与 JSON-RPC 通信。"""

    def __init__(self, name: str, command: str, args: list[str] | None = None) -> None:
        self._name = name
        self._command = command
        self._args = args or []
        self._process: subprocess.Popen | None = None
        self._request_id = 0

    async def start(self) -> None:
        """启动 MCP Server 进程并完成初始化握手。"""
        try:
            self._process = subprocess.Popen(
                [self._command] + self._args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            # 初始化握手
            init_result = await self._send_request("initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "AgentKB", "version": "0.2.0"},
            })
            logger.info(f"MCP Server '{self._name}' 初始化成功: {init_result}")
        except Exception as e:
            logger.warning(f"MCP Server '{self._name}' 启动失败: {e}")
            self._process = None

    async def stop(self) -> None:
        """停止 MCP Server 进程。"""
        if self._process:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

    async def _send_request(self, method: str, params: dict | None = None) -> dict:
        """发送 JSON-RPC 请求并返回结果——阻塞 I/O 通过线程池执行。"""
        if not self._process or self._process.poll() is not None:
            raise RuntimeError(f"MCP Server '{self._name}' 未运行")

        self._request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": params or {},
        }

        # 同步 I/O 丢到线程池，避免阻塞事件循环
        import asyncio as _asyncio
        loop = _asyncio.get_running_loop()

        def _sync_io() -> dict:
            self._process.stdin.write(json.dumps(request) + "\n")
            self._process.stdin.flush()
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError(f"MCP Server '{self._name}' 无响应")
            response = json.loads(line)
            if "error" in response:
                raise RuntimeError(f"MCP 错误: {response['error']}")
            return response.get("result", {})

        return await loop.run_in_executor(None, _sync_io)

    async def list_tools(self) -> list[dict]:
        """获取 MCP Server 提供的工具列表。"""
        result = await self._send_request("tools/list")
        return result.get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """调用 MCP Server 的指定工具。"""
        return await self._send_request("tools/call", {
            "name": tool_name,
            "arguments": arguments,
        })


class MCPManager:
    """管理多个 MCP Server 的生命周期与工具注册。"""

    def __init__(self) -> None:
        self._bridges: list[MCPClientBridge] = []

    def add_server(self, name: str, command: str, args: list[str] | None = None) -> None:
        self._bridges.append(MCPClientBridge(name=name, command=command, args=args))

    async def start_all(self) -> None:
        """启动所有 MCP Server 并将工具注册到 ToolRegistry。"""
        registry = ToolRegistry()
        for bridge in self._bridges:
            try:
                await bridge.start()
                if bridge._process:
                    tools = await bridge.list_tools()
                    for tool_info in tools:
                        mcp_tool = MCPTool(
                            mcp_name=bridge._name,
                            tool_name=tool_info.get("name", "unknown"),
                            description=tool_info.get("description", ""),
                            input_schema=tool_info.get("inputSchema", {}),
                            mcp_instance=bridge,
                        )
                        registry.register(mcp_tool)
                    logger.info(f"已从 MCP '{bridge._name}' 注册 {len(tools)} 个工具")
            except Exception as e:
                logger.warning(f"MCP Server '{bridge._name}' 初始化跳过: {e}")

    async def stop_all(self) -> None:
        for bridge in self._bridges:
            await bridge.stop()
        self._bridges.clear()


# 模块级单例
_mcp_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager:
    global _mcp_manager
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    return _mcp_manager
