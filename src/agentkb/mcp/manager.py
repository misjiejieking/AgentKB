"""MCP 服务发现、状态管理和工具调用。"""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncIterator

import httpx
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from agentkb.mcp.models import requires_confirmation
from agentkb.mcp.tool import MCPTool
from agentkb.storage.pg_database import Database, get_db
from agentkb.tools.base import ToolResult
from agentkb.tools.registry import ToolRegistry

_ENV_REFERENCE = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)\}$")


class MCPManager:
    def __init__(
        self,
        db: Database | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.db = db or get_db()
        self.registry = registry or ToolRegistry()
        self._locks: dict[str, asyncio.Lock] = {}

    async def start(self) -> None:
        servers = await asyncio.to_thread(self.db.list_mcp_servers, enabled_only=True)
        for server in servers:
            try:
                await self.connect(str(server["id"]))
            except Exception as exc:
                logger.error(f"MCP 服务 {server['name']} 启动连接失败: {exc}")

    async def stop(self) -> None:
        for server in await asyncio.to_thread(self.db.list_mcp_servers):
            self._unregister_server_tools(str(server["id"]))

    async def connect(self, server_id: str) -> list[dict[str, Any]]:
        async with self._lock(server_id):
            server = await self._get_server(server_id)
            await asyncio.to_thread(
                self.db.set_mcp_server_connection,
                server_id,
                connection_status="connecting",
                enabled=True,
            )
            try:
                remote_tools = await self._discover(server)
                snapshots = [
                    self._tool_snapshot(server, tool.model_dump(mode="json"))
                    for tool in remote_tools
                ]
                rows = await asyncio.to_thread(
                    self.db.replace_mcp_tools,
                    server_id,
                    snapshots,
                )
                self._register_tools(server, rows)
                await asyncio.to_thread(
                    self.db.set_mcp_server_connection,
                    server_id,
                    connection_status="connected",
                    enabled=True,
                )
                return rows
            except Exception as exc:
                self._unregister_server_tools(server_id)
                await asyncio.to_thread(
                    self.db.set_mcp_server_connection,
                    server_id,
                    connection_status="error",
                    error=str(exc),
                )
                raise

    async def disconnect(self, server_id: str) -> None:
        await self._get_server(server_id)
        self._unregister_server_tools(server_id)
        await asyncio.to_thread(
            self.db.set_mcp_server_connection,
            server_id,
            connection_status="disconnected",
            enabled=False,
        )

    async def call_tool(
        self,
        server_id: str,
        remote_name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        server = await self._get_server(server_id)
        async with self._session(server) as session:
            result = await session.call_tool(remote_name, arguments=arguments)
        data: Any = result.structuredContent
        if data is None:
            data = [
                block.model_dump(mode="json", by_alias=True)
                for block in result.content
            ]
        return ToolResult(
            tool_name=remote_name,
            success=not result.isError,
            data=data if not result.isError else None,
            error=_error_text(result.content) if result.isError else None,
        )

    def public_servers(self) -> list[dict[str, Any]]:
        servers = self.db.list_mcp_servers()
        tools = self.db.list_mcp_tools()
        by_server: dict[str, list[dict[str, Any]]] = {}
        for tool in tools:
            by_server.setdefault(str(tool["server_id"]), []).append(tool)
        return [
            {
                **{
                    key: value
                    for key, value in server.items()
                    if key not in {"env", "headers"}
                },
                "env_keys": sorted(server["env"]),
                "header_keys": sorted(server["headers"]),
                "tools": by_server.get(str(server["id"]), []),
            }
            for server in servers
        ]

    async def delete(self, server_id: str) -> bool:
        self._unregister_server_tools(server_id)
        return bool(await asyncio.to_thread(self.db.delete_mcp_server, server_id))

    async def set_tool_enabled(
        self,
        server_id: str,
        remote_name: str,
        enabled: bool,
    ) -> dict[str, Any] | None:
        row = await asyncio.to_thread(
            self.db.set_mcp_tool_enabled,
            server_id,
            remote_name,
            enabled,
        )
        server = await self._get_server(server_id)
        rows = await asyncio.to_thread(self.db.list_mcp_tools, server_id)
        self._register_tools(server, rows)
        return row

    async def _discover(self, server: dict[str, Any]) -> list[Any]:
        async with self._session(server) as session:
            page = await session.list_tools()
            tools = list(page.tools)
            cursor = page.nextCursor
            while cursor:
                page = await session.list_tools(cursor=cursor)
                tools.extend(page.tools)
                cursor = page.nextCursor
            return tools

    @asynccontextmanager
    async def _session(
        self,
        server: dict[str, Any],
    ) -> AsyncIterator[ClientSession]:
        if server["transport"] == "stdio":
            params = StdioServerParameters(
                command=server["command"],
                args=list(server["args"]),
                env={**os.environ, **_resolve_templates(server["env"])},
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=60),
                ) as session:
                    await session.initialize()
                    yield session
            return

        headers = _resolve_templates(server["headers"])
        async with httpx.AsyncClient(headers=headers, timeout=60) as client:
            async with streamable_http_client(
                server["url"],
                http_client=client,
            ) as streams:
                async with ClientSession(
                    streams[0],
                    streams[1],
                    read_timeout_seconds=timedelta(seconds=60),
                ) as session:
                    await session.initialize()
                    yield session

    async def _get_server(self, server_id: str) -> dict[str, Any]:
        server = await asyncio.to_thread(self.db.get_mcp_server, server_id)
        if server is None:
            raise ValueError("MCP 服务不存在")
        return server

    def _register_tools(
        self,
        server: dict[str, Any],
        rows: list[dict[str, Any]],
    ) -> None:
        self._unregister_server_tools(str(server["id"]))
        for row in rows:
            if not row["enabled"]:
                continue
            self.registry.register(
                MCPTool(
                    manager=self,
                    server_id=str(server["id"]),
                    server_name=str(server["name"]),
                    remote_name=str(row["remote_name"]),
                    local_name=str(row["local_name"]),
                    description=str(row["description"]),
                    input_schema=dict(row["input_schema"]),
                    requires_confirmation=bool(row["requires_confirmation"]),
                )
            )

    def _unregister_server_tools(self, server_id: str) -> None:
        for tool in self.registry.list_tools():
            if isinstance(tool, MCPTool) and tool.server_id == server_id:
                self.registry.unregister(tool.name)

    def _tool_snapshot(
        self,
        server: dict[str, Any],
        tool: dict[str, Any],
    ) -> dict[str, Any]:
        annotations = tool.get("annotations") or {}
        remote_name = str(tool["name"])
        return {
            "remote_name": remote_name,
            "local_name": _local_tool_name(str(server["name"]), remote_name),
            "description": str(tool.get("description") or remote_name),
            "input_schema": tool["inputSchema"],
            "annotations": annotations,
            "requires_confirmation": requires_confirmation(
                str(server["confirmation_policy"]),
                annotations,
            ),
        }

    def _lock(self, server_id: str) -> asyncio.Lock:
        return self._locks.setdefault(server_id, asyncio.Lock())


def _resolve_templates(values: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, value in values.items():
        match = _ENV_REFERENCE.fullmatch(value)
        if not match:
            resolved[key] = value
            continue
        env_name = match.group(1)
        if env_name not in os.environ:
            raise ValueError(f"环境变量 {env_name} 未配置")
        resolved[key] = os.environ[env_name]
    return resolved


def _local_tool_name(server_name: str, remote_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]", "_", remote_name).strip("_").lower()
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        raise ValueError(f"MCP 工具名称无有效字符: {remote_name}")
    return f"mcp__{server_name}__{normalized}"[:128]


def _error_text(content: list[Any]) -> str:
    texts = [
        str(getattr(block, "text", ""))
        for block in content
        if getattr(block, "text", "")
    ]
    return "\n".join(texts) or json.dumps(
        [block.model_dump(mode="json", by_alias=True) for block in content],
        ensure_ascii=False,
    )


_manager: MCPManager | None = None


def get_mcp_manager() -> MCPManager:
    global _manager
    if _manager is None:
        _manager = MCPManager()
    return _manager
