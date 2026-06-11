from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from agentkb.agents.configurable import ConfigurableAgent
from agentkb.agents.custom_service import AgentDraft, CustomAgentService
from agentkb.agents.registry import AgentRegistry
from agentkb.tools.base import BaseTool, ToolResult
from agentkb.tools.registry import ToolRegistry


class EmptyArgs(BaseModel):
    pass


class FakeTool(BaseTool):
    def __init__(self, name: str, requires_confirmation: bool = False) -> None:
        self._name = name
        self._requires_confirmation = requires_confirmation

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"{self._name} description"

    @property
    def args_schema(self):
        return EmptyArgs

    @property
    def requires_confirmation(self) -> bool:
        return self._requires_confirmation

    async def _execute(self, **kwargs) -> ToolResult:
        return ToolResult(self.name, True, {"ok": True})


class FakeDraftLlm:
    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, prompt):
        return {
            "name": "requirement_reviewer",
            "display_name": "需求评审 Agent",
            "description": "检查需求完整性并输出验收标准",
            "instructions": "逐项检查目标、范围、边界、风险和验收标准，不得补造缺失信息。",
            "intents": ["requirement_review"],
            "allowed_tools": [],
            "model_name": None,
        }


class FakeDatabase:
    def __init__(self) -> None:
        self.row = None

    def create_custom_agent(self, *, agent_id, **values):
        self.row = {
            "id": agent_id,
            "status": "active",
            "created_at": "now",
            "updated_at": "now",
            **values,
        }
        return self.row

    def set_custom_agent_status(self, agent_id, status):
        if not self.row or self.row["id"] != agent_id:
            return None
        self.row["status"] = status
        return dict(self.row)

    def delete_custom_agent(self, agent_id):
        if not self.row or self.row["id"] != agent_id:
            return None
        row = self.row
        self.row = None
        return row

    def list_custom_agents(self, *, active_only=False):
        if not self.row:
            return []
        if active_only and self.row["status"] != "active":
            return []
        return [dict(self.row)]


@pytest.fixture(autouse=True)
def reset_registries():
    AgentRegistry.reset()
    ToolRegistry.reset()
    yield
    AgentRegistry.reset()
    ToolRegistry.reset()


async def test_natural_language_draft_does_not_grant_tools():
    ToolRegistry().register(FakeTool("safe_search"))
    ToolRegistry().register(FakeTool("dangerous_write", True))

    draft = await CustomAgentService(
        db=FakeDatabase(),
        llm=FakeDraftLlm(),
    ).draft("创建一个需求评审 Agent，检查需求完整性并输出验收标准")

    assert draft.name == "requirement_reviewer"
    assert draft.allowed_tools == []


async def test_natural_language_draft_rejects_unrelated_business_domain():
    class UnrelatedLlm(FakeDraftLlm):
        async def ainvoke(self, prompt):
            return {
                "name": "knowledge_searcher",
                "display_name": "知识库搜索 Agent",
                "description": "从知识库检索相关文档",
                "instructions": "根据用户问题检索知识库并返回相关文档，不得编造内容。",
                "intents": ["knowledge_search"],
                "allowed_tools": [],
                "model_name": None,
            }

    with pytest.raises(ValueError, match="草案与用户描述不一致"):
        await CustomAgentService(
            db=FakeDatabase(),
            llm=UnrelatedLlm(),
        ).draft("创建一个需求评审 Agent，检查需求完整性并输出验收标准")


async def test_natural_language_draft_rejects_log_parser_for_review_request():
    class LogParserLlm(FakeDraftLlm):
        async def ainvoke(self, prompt):
            return {
                "name": "parse_log_file",
                "display_name": "解析日志文件",
                "description": "从日志文件中提取关键信息并格式化输出",
                "instructions": "读取日志内容，解析日期、时间、级别和消息并输出结构化数据。",
                "intents": ["parse_log"],
                "allowed_tools": [],
                "model_name": None,
            }

    with pytest.raises(ValueError, match="草案与用户描述不一致"):
        await CustomAgentService(
            db=FakeDatabase(),
            llm=LogParserLlm(),
        ).draft("创建一个需求评审 Agent，检查需求完整性，识别风险并输出验收标准")


def test_custom_agent_requires_confirmation_before_registration():
    ToolRegistry().register(FakeTool("safe_search"))
    db = FakeDatabase()
    service = CustomAgentService(db=db)
    draft = AgentDraft(
        name="requirement_reviewer",
        display_name="需求评审 Agent",
        description="检查需求完整性并输出验收标准",
        instructions="逐项检查目标、范围、边界、风险和验收标准，不得补造缺失信息。",
        intents=["requirement_review"],
        allowed_tools=["safe_search"],
    )

    assert AgentRegistry().get(draft.name) is None
    created = service.create(draft)
    assert AgentRegistry().get(draft.name) is not None

    service.set_status(created["id"], "disabled")
    assert AgentRegistry().get(draft.name) is None


def test_custom_agent_rejects_confirmation_tools():
    ToolRegistry().register(FakeTool("dangerous_write", True))
    draft = AgentDraft(
        name="unsafe_agent",
        display_name="不安全 Agent",
        description="尝试执行高风险写入操作",
        instructions="执行高风险操作，这个配置应当在创建阶段被拒绝。",
        intents=["unsafe_operation"],
        allowed_tools=["dangerous_write"],
    )

    with pytest.raises(ValueError, match="需要人工确认"):
        CustomAgentService(db=FakeDatabase()).create(draft)


async def test_configurable_agent_returns_model_output(monkeypatch):
    class FakeExecutionLlm:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages):
            return AIMessage(content="需求缺少明确的验收指标。")

    monkeypatch.setattr(
        "agentkb.agents.configurable.get_chat_model",
        lambda streaming=False: FakeExecutionLlm(),
    )
    agent = ConfigurableAgent({
        "name": "requirement_reviewer",
        "display_name": "需求评审 Agent",
        "description": "检查需求完整性",
        "instructions": "检查需求并指出缺失项，输出必须明确、简洁、可执行。",
        "intents": ["requirement_review"],
        "allowed_tools": [],
        "model_name": None,
    })

    result = await agent.execute("用户需要一个搜索功能")

    assert result.success is True
    assert result.output == "需求缺少明确的验收指标。"
