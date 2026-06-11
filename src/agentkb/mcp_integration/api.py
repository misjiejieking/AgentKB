"""MCP 服务配置与工具管理 API。"""

from __future__ import annotations

import asyncio
import uuid

from fastapi import APIRouter, HTTPException
from psycopg2 import IntegrityError  # type: ignore[import-untyped]

from agentkb.mcp_integration.manager import get_mcp_manager
from agentkb.mcp_integration.models import MCPServerCreate, MCPToolStatusRequest
from agentkb.storage.pg_database import get_db

router = APIRouter(prefix="/mcp", tags=["mcp"])


@router.get("/servers")
async def list_servers():
    servers = await asyncio.to_thread(get_mcp_manager().public_servers)
    return {"servers": servers}


@router.post("/servers", status_code=201)
async def create_server(request: MCPServerCreate):
    payload = {
        "id": uuid.uuid4().hex,
        **request.model_dump(),
    }
    try:
        row = await asyncio.to_thread(get_db().create_mcp_server, payload)
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="MCP 服务标识已存在") from exc
    return {"server": _public_server(row)}


@router.post("/servers/{server_id}/connect")
async def connect_server(server_id: str):
    try:
        tools = await get_mcp_manager().connect(server_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MCP 连接失败: {exc}") from exc
    return {"connected": True, "tools": tools}


@router.post("/servers/{server_id}/disconnect")
async def disconnect_server(server_id: str):
    try:
        await get_mcp_manager().disconnect(server_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"disconnected": True}


@router.post("/servers/{server_id}/refresh")
async def refresh_server(server_id: str):
    return await connect_server(server_id)


@router.patch("/servers/{server_id}/tools/{remote_name}")
async def set_tool_status(
    server_id: str,
    remote_name: str,
    request: MCPToolStatusRequest,
):
    try:
        row = await get_mcp_manager().set_tool_enabled(
            server_id,
            remote_name,
            request.enabled,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if row is None:
        raise HTTPException(status_code=404, detail="MCP 工具不存在")
    return {"tool": row}


@router.delete("/servers/{server_id}")
async def delete_server(server_id: str):
    if not await get_mcp_manager().delete(server_id):
        raise HTTPException(status_code=404, detail="MCP 服务不存在")
    return {"deleted": True, "id": server_id}


def _public_server(row: dict) -> dict:
    return {
        key: value
        for key, value in row.items()
        if key not in {"env", "headers"}
    } | {
        "env_keys": sorted(row["env"]),
        "header_keys": sorted(row["headers"]),
        "tools": [],
    }
