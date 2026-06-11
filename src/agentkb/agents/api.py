"""自定义 Agent HTTP API。"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, HTTPException
from psycopg2 import IntegrityError  # type: ignore[import-untyped]
from pydantic import BaseModel, Field

from agentkb.agents.custom_service import AgentDraft, CustomAgentService

router = APIRouter(prefix="/agents", tags=["agents"])


class AgentDraftRequest(BaseModel):
    request: str = Field(min_length=10, max_length=2000)


class AgentStatusRequest(BaseModel):
    status: Literal["active", "disabled"]


@router.post("/draft")
async def draft_agent(req: AgentDraftRequest):
    """根据自然语言生成待用户确认的 Agent 草案。"""
    try:
        draft = await CustomAgentService().draft(req.request)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "draft": draft.model_dump(),
        "requires_confirmation": True,
    }


@router.post("")
async def create_agent(draft: AgentDraft):
    """确认草案并创建 Agent。"""
    try:
        row = await asyncio.to_thread(CustomAgentService().create, draft)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Agent 名称已存在") from exc
    return {"agent": _public_agent(row)}


@router.get("")
async def list_agents():
    """列出自定义 Agent 和可分配工具。"""
    service = CustomAgentService()
    rows = await asyncio.to_thread(service.db.list_custom_agents)
    tools = [
        {
            "name": tool.name,
            "description": tool.description,
        }
        for tool in service.safe_tools().values()
    ]
    return {
        "agents": [_public_agent(row) for row in rows],
        "tools": tools,
    }


@router.patch("/{agent_id}/status")
async def set_agent_status(agent_id: str, req: AgentStatusRequest):
    """启用或停用自定义 Agent。"""
    row = await asyncio.to_thread(
        CustomAgentService().set_status,
        agent_id,
        req.status,
    )
    if not row:
        raise HTTPException(status_code=404, detail="自定义 Agent 不存在")
    return {"agent": _public_agent(row)}


@router.delete("/{agent_id}")
async def delete_agent(agent_id: str):
    """删除自定义 Agent。"""
    row = await asyncio.to_thread(CustomAgentService().delete, agent_id)
    if not row:
        raise HTTPException(status_code=404, detail="自定义 Agent 不存在")
    return {"deleted": True, "id": agent_id}


def _public_agent(row: dict) -> dict:
    return {
        "id": row["id"],
        "name": row["name"],
        "display_name": row["display_name"],
        "description": row["description"],
        "instructions": row["instructions"],
        "intents": row["intents"],
        "allowed_tools": row["allowed_tools"],
        "model_name": row["model_name"],
        "status": row["status"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
