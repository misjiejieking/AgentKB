"""SQLite 存储层的数据模型定义（Pydantic）。"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


def new_id() -> str:
    """生成 12 位短的唯一 ID，用作主键。"""
    return uuid.uuid4().hex[:12]


def now_iso() -> str:
    """返回当前时间的 ISO 格式字符串（精确到秒）。"""
    return datetime.now().isoformat(timespec="seconds")


class MessageRole(str, Enum):
    """消息角色枚举：用户 / 助手 / 工具。"""
    HUMAN = "human"
    AI = "ai"
    TOOL = "tool"


class Session(BaseModel):
    """会话记录。"""
    id: str = Field(default_factory=new_id)
    title: str = "New Chat"
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class Message(BaseModel):
    """聊天消息记录。tool_calls 和 tool_results 以 JSON 字符串存储。"""
    id: str = Field(default_factory=new_id)
    session_id: str
    role: MessageRole
    content: str
    tool_calls: str | None = None       # JSON 序列化的工具调用记录
    tool_results: str | None = None     # JSON 序列化的工具返回结果
    created_at: str = Field(default_factory=now_iso)


class KnowledgeFile(BaseModel):
    """知识库文件记录。status 为 'active' 或 'deleted'（软删除）。"""
    id: str = Field(default_factory=new_id)
    filename: str
    filepath: str
    file_size: int = 0
    chunk_count: int = 0
    status: str = "active"
    created_at: str = Field(default_factory=now_iso)
