"""自定义 Agent 草案生成、校验和运行时注册。"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

from agentkb.agents.configurable import ConfigurableAgent
from agentkb.agents.registry import get_agent_registry
from agentkb.storage.models import new_id
from agentkb.tools.registry import ToolRegistry

AGENT_DRAFT_PROMPT = """你是 Agent 配置设计器。根据用户描述生成一个专业、单一职责的 Agent 草案。

首要约束：
- Agent 的名称、描述、指令和意图必须直接对应用户描述，不得替换成其他能力。
- 草案阶段禁止分配任何工具，allowed_tools 必须是空数组。工具由用户在确认界面手工授权。

必须实现的用户原始描述：
{request}

输出严格 JSON：
{{
  "name": "英文小写 snake_case，3-40字符",
  "display_name": "中文展示名称",
  "description": "一句话说明适用任务，供调度器路由",
  "instructions": "明确职责、处理步骤、输出要求和不可编造约束",
  "intents": ["英文小写 snake_case 意图"],
  "allowed_tools": ["仅选择上方工具名称"],
  "model_name": null
}}

不要创建万能 Agent，不要改变用户要求的业务领域。"""

class AgentDraft(BaseModel):
    name: str = Field(
        min_length=3,
        max_length=40,
        description=(
            "Agent 唯一英文标识，必须概括用户要求的业务职责。"
            "例如需求评审使用 requirement_reviewer。"
        ),
    )
    display_name: str = Field(
        min_length=1,
        max_length=80,
        description="面向用户的中文名称，例如“需求评审 Agent”。",
    )
    description: str = Field(
        min_length=5,
        max_length=300,
        description="供调度器路由的一句话职责，必须复述用户要求的具体业务能力。",
    )
    instructions: str = Field(
        min_length=20,
        max_length=4000,
        description="具体执行步骤、输出结构、事实边界和禁止事项。",
    )
    intents: list[str] = Field(
        min_length=1,
        max_length=8,
        description=(
            "与用户业务职责对应的英文 snake_case 意图。"
            "例如需求评审使用 requirement_review。"
        ),
    )
    allowed_tools: list[str] = Field(
        default_factory=list,
        max_length=8,
        description="草案阶段必须为空数组。",
    )
    model_name: str | None = Field(
        default=None,
        max_length=120,
        description="必须为 null，使用系统默认模型。",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip().lower()
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", value):
            raise ValueError("name 必须是 3-40 位英文 snake_case")
        return value

    @field_validator("intents")
    @classmethod
    def validate_intents(cls, values: list[str]) -> list[str]:
        normalized = []
        for value in values:
            intent = value.strip().lower()
            if not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", intent):
                raise ValueError("intent 必须使用英文 snake_case")
            if intent not in normalized:
                normalized.append(intent)
        return normalized

    @field_validator("allowed_tools")
    @classmethod
    def deduplicate_tools(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))


class CustomAgentService:
    """管理自定义 Agent 的完整生命周期。"""

    def __init__(self, db=None, llm=None) -> None:
        if db is None:
            from agentkb.storage.pg_database import get_db
            db = get_db()
        self.db = db
        self.llm = llm

    async def draft(self, request: str) -> AgentDraft:
        request = request.strip()
        llm = self.llm
        if llm is None:
            from agentkb.llm.factory import get_chat_model
            llm = get_chat_model(streaming=False)
        retry_note = ""
        for _ in range(2):
            prompt = AGENT_DRAFT_PROMPT.format(
                request=request,
            )
            prompt += retry_note
            result = await llm.with_structured_output(AgentDraft).ainvoke(prompt)
            draft = (
                result
                if isinstance(result, AgentDraft)
                else AgentDraft.model_validate(result)
            )
            draft = draft.model_copy(update={"allowed_tools": []})
            self.validate_draft(draft)
            if _is_semantically_aligned(request, draft):
                return draft
            retry_note = (
                "\n\n上一版偏离了用户描述。新草案必须保留原始业务关键词和全部要求，"
                "禁止替换成知识库、文本分类或日志分析等其他职责。"
            )
        raise ValueError("本地模型生成的草案与用户描述不一致，请换一种更具体的描述")

    def create(self, draft: AgentDraft) -> dict[str, Any]:
        self.validate_draft(draft)
        row = self.db.create_custom_agent(
            agent_id=new_id(),
            **draft.model_dump(),
        )
        get_agent_registry().register(ConfigurableAgent(row))
        return row

    def set_status(self, agent_id: str, status: str) -> dict[str, Any] | None:
        row = self.db.set_custom_agent_status(agent_id, status)
        if not row:
            return None
        registry = get_agent_registry()
        if status == "active":
            registry.register(ConfigurableAgent(row))
        else:
            registry.unregister(row["name"])
        return row

    def delete(self, agent_id: str) -> dict[str, Any] | None:
        row = self.db.delete_custom_agent(agent_id)
        if row:
            get_agent_registry().unregister(row["name"])
        return row

    def load_active(self) -> int:
        rows = self.db.list_custom_agents(active_only=True)
        registry = get_agent_registry()
        for row in rows:
            registry.register(ConfigurableAgent(row))
        return len(rows)

    def validate_draft(self, draft: AgentDraft) -> None:
        registry = get_agent_registry()
        existing = registry.get(draft.name)
        if existing is not None:
            raise ValueError(f"Agent 名称已存在: {draft.name}")
        safe_tools = self.safe_tools()
        invalid = [
            name for name in draft.allowed_tools
            if name not in safe_tools
        ]
        if invalid:
            raise ValueError(f"工具不存在或需要人工确认: {', '.join(invalid)}")

    @staticmethod
    def safe_tools() -> dict[str, Any]:
        return {
            tool.name: tool
            for tool in ToolRegistry().list_tools()
            if not tool.requires_confirmation
        }


def _is_semantically_aligned(request: str, draft: AgentDraft) -> bool:
    """用业务关键词覆盖率拦截职责被替换的草案。"""
    request_terms = _business_terms(request)
    if not request_terms:
        return True
    draft_text = " ".join((
        draft.display_name,
        draft.description,
        draft.instructions,
    ))
    covered = request_terms & _business_terms(draft_text)
    return len(covered) / len(request_terms) >= 0.6


def _business_terms(text: str) -> set[str]:
    stop_terms = {
        "一个", "创建", "生成", "提供", "进行", "用于", "能够", "可以",
        "需要", "用户", "输出", "结果", "任务", "功能", "执行", "处理",
        "相关", "信息", "具体", "要求", "使用", "根据",
    }
    sequences = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    return {
        sequence[index:index + 2]
        for sequence in sequences
        for index in range(len(sequence) - 1)
        if sequence[index:index + 2] not in stop_terms
    }
