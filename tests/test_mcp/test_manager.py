from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from agentkb.mcp_integration.manager import (
    MCPManager,
    _local_tool_name,
    _resolve_templates,
)
from agentkb.mcp_integration.models import MCPServerCreate, requires_confirmation
from agentkb.tools.registry import ToolRegistry


class FakeDatabase:
    def __init__(self, server: dict[str, Any]) -> None:
        self.server = server
        self.tools: list[dict[str, Any]] = []

    def get_mcp_server(self, server_id: str):
        return self.server if server_id == self.server["id"] else None

    def list_mcp_servers(self, *, enabled_only: bool = False):
        if enabled_only and self.server["status"] != "enabled":
            return []
        return [self.server]

    def set_mcp_server_connection(
        self,
        server_id: str,
        *,
        connection_status: str,
        enabled: bool | None = None,
        error: str = "",
    ):
        self.server["connection_status"] = connection_status
        self.server["last_error"] = error
        if enabled is not None:
            self.server["status"] = "enabled" if enabled else "disabled"
        return self.server

    def replace_mcp_tools(self, server_id: str, tools: list[dict[str, Any]]):
        existing_enabled = {
            tool["remote_name"]: tool["enabled"]
            for tool in self.tools
        }
        self.tools = [
            {
                "server_id": server_id,
                "enabled": existing_enabled.get(tool["remote_name"], True),
                **tool,
            }
            for tool in tools
        ]
        return self.tools

    def list_mcp_tools(self, server_id: str | None = None):
        return [
            tool for tool in self.tools
            if server_id is None or tool["server_id"] == server_id
        ]

    def set_mcp_tool_enabled(
        self,
        server_id: str,
        remote_name: str,
        enabled: bool,
    ):
        for tool in self.tools:
            if tool["server_id"] == server_id and tool["remote_name"] == remote_name:
                tool["enabled"] = enabled
                return tool
        return None


@pytest.fixture
def stdio_server() -> dict[str, Any]:
    script = Path(__file__).parents[1] / "fixtures" / "mcp_stdio_server.py"
    return {
        "id": "server-1",
        "name": "test_server",
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(script)],
        "url": None,
        "env": {},
        "headers": {},
        "confirmation_policy": "writes",
        "status": "disabled",
        "connection_status": "disconnected",
        "last_error": "",
    }


@pytest.mark.asyncio
async def test_stdio_discovery_registration_and_call(stdio_server):
    ToolRegistry.reset()
    database = FakeDatabase(stdio_server)
    manager = MCPManager(db=database, registry=ToolRegistry())

    tools = await manager.connect("server-1")

    assert {tool["remote_name"] for tool in tools} == {"echo", "replace_value"}
    assert database.server["connection_status"] == "connected"
    echo = ToolRegistry().get("mcp__test_server__echo")
    write_tool = ToolRegistry().get("mcp__test_server__replace_value")
    assert echo is not None and not echo.requires_confirmation
    assert write_tool is not None and write_tool.requires_confirmation

    result = await echo.execute(text="hello")
    assert result.success
    assert result.data == {"text": "hello"}

    await manager.set_tool_enabled("server-1", "echo", False)
    assert ToolRegistry().get("mcp__test_server__echo") is None
    assert ToolRegistry().get("mcp__test_server__replace_value") is not None
    ToolRegistry.reset()


def test_server_validation_and_confirmation_policy():
    server = MCPServerCreate(
        name="remote_tools",
        transport="streamable_http",
        url="https://example.com/mcp",
    )
    assert server.command is None
    assert requires_confirmation("writes", {"readOnlyHint": True}) is False
    assert requires_confirmation("writes", {}) is True
    assert requires_confirmation("always", {"readOnlyHint": True}) is True
    assert requires_confirmation("never", {}) is False


def test_environment_templates_are_explicit(monkeypatch):
    monkeypatch.setenv("MCP_TEST_TOKEN", "secret")
    assert _resolve_templates({"TOKEN": "${MCP_TEST_TOKEN}", "MODE": "local"}) == {
        "TOKEN": "secret",
        "MODE": "local",
    }
    with pytest.raises(ValueError, match="MISSING_TOKEN"):
        _resolve_templates({"TOKEN": "${MISSING_TOKEN}"})


def test_local_tool_name_is_namespaced_and_normalized():
    assert _local_tool_name("github", "Create-Issue") == "mcp__github__create_issue"
