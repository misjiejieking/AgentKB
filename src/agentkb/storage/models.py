"""存储层通用标识生成。"""

from __future__ import annotations

import uuid


def new_id() -> str:
    """生成 12 位短的唯一 ID，用作主键。"""
    return uuid.uuid4().hex[:12]
