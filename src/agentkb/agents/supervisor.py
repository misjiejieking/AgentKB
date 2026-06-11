"""Supervisor Agent——任务分解、Agent 路由、结果聚合和质量控制。

Supervisor 是 Multi-Agent 系统的中枢：
  1. 意图分析：理解用户真正想要什么
  2. 任务分解：将复杂请求拆成子任务
  3. Agent 路由：选择合适的 Specialist Agent
  4. 结果聚合：将多个 Agent 的输出合并为连贯回答
  5. 质量控制：检查完整性，必要时触发 Reflection
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from loguru import logger

SUPERVISOR_PROMPT = """你是 AgentKB 的总调度员，负责理解用户意图并将任务指派给合适的 Specialist Agent。

## 可用 Agent
{agent_descriptions}

## 对话历史
{history}

## 用户最新消息
{query}

## 决策规则（严格遵守）

**规则 1：不需要 Agent 的情况 → intent=chat，在 direct_reply 中直接回复**
以下情况必须使用 intent="chat"，subtasks=[], direct_reply 写完整回复：
- 闲聊、打招呼（"你好"、"你是谁"）
- 询问助手能力（"你会做什么"、"你能干嘛"、"你有什么功能"）
- 常识性问题（"今天星期几"、"1+1等于几"）
- 不需要工具就能回答的简单问题

**规则 2：需要 Agent 的情况 → intent 选对应类型，subtasks 填任务列表**
以下情况必须创建 subtasks，direct_reply=null：
- 查本地文档 → intent="knowledge_search", assigned_agent="knowledge_agent"
- 写文章/报告/脚本 → intent="content_creation", assigned_agent="content_creator"
- 管理任务/项目 → intent="task_management", assigned_agent="task_manager"
- 学习辅导/教程 → intent="learning", assigned_agent="learning_tutor"
- 社媒内容（小红书/抖音等） → intent="social_content", assigned_agent="social_writer"
- 保存或查询跨会话记忆 → intent="personal_memory", assigned_agent="memory_agent"

除上述内置场景外，如“可用 Agent”中存在职责明确匹配的自定义 Agent，
必须使用其 name 作为 assigned_agent，并使用其 intents 中最匹配的一项作为 intent。

复杂请求应拆成多个子任务。无依赖的子任务可以并行；后续子任务通过 dependencies
引用前置子任务 id。每个子任务必须包含 id、description、assigned_agent、dependencies。

## 输出格式（严格 JSON，不要包含 markdown 标记，assigned_agent 不要用字符串 "null"）
{{
  "intent": "knowledge_search",
  "reasoning": "判断依据",
  "direct_reply": null,
  "subtasks": [
    {{
      "id": 1,
      "description": "明确、可执行的子任务",
      "assigned_agent": "knowledge_agent",
      "dependencies": []
    }}
  ]
}}"""


@dataclass
class TaskDecomposition:
    """Supervisor 对用户请求的分析结果。"""
    intent: str
    reasoning: str = ""
    direct_reply: str | None = None       # 简单闲聊直接回复
    subtasks: list[SubTask] = field(default_factory=list)

    @property
    def needs_agent(self) -> bool:
        return len(self.subtasks) > 0 and self.direct_reply is None


@dataclass
class SubTask:
    """单个子任务定义。"""
    id: int
    description: str
    assigned_agent: str = ""
    dependencies: list[int] = field(default_factory=list)  # 依赖的 subtask id


class SupervisorAgent:
    """任务调度器——分析意图、分解任务、路由 Agent。"""

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    async def analyze(self, query: str, history: list[str] | None = None) -> TaskDecomposition:
        """分析用户请求并生成任务分解。"""
        from agentkb.agents.registry import get_agent_registry

        # 快速路由（避免不必要的 LLM 调用）
        fast = self._fast_route(query)
        if fast:
            return fast

        registry = get_agent_registry()
        agent_descriptions = registry.get_agent_descriptions()

        history_text = "\n".join(history[-6:]) if history else "（无历史）"

        prompt = SUPERVISOR_PROMPT.format(
            agent_descriptions=agent_descriptions,
            history=history_text,
            query=query,
        )

        llm = self._llm
        if llm is None:
            from agentkb.llm.factory import get_chat_model
            llm = get_chat_model(streaming=True)

        try:
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            content = self._extract_json(content)
            data = json.loads(content)

            subtasks = []
            used_ids: set[int] = set()
            for index, st in enumerate(data.get("subtasks", []), 1):
                task_id = st.get("id")
                if (
                    not isinstance(task_id, int)
                    or task_id <= 0
                    or task_id in used_ids
                ):
                    task_id = index
                    while task_id in used_ids:
                        task_id += 1
                used_ids.add(task_id)
                subtasks.append(SubTask(
                    id=task_id,
                    description=st.get("description", ""),
                    assigned_agent=st.get("assigned_agent", ""),
                    dependencies=[
                        dependency
                        for dependency in st.get("dependencies", [])
                        if isinstance(dependency, int)
                    ],
                ))

            return TaskDecomposition(
                intent=data.get("intent", "knowledge_search"),
                reasoning=data.get("reasoning", ""),
                direct_reply=data.get("direct_reply"),
                subtasks=subtasks,
            )
        except Exception as e:
            logger.warning(f"Supervisor 分析失败: {e}，降级到知识检索")
            return TaskDecomposition(
                intent="knowledge_search",
                reasoning="Supervisor 分析异常，降级处理",
                subtasks=[SubTask(
                    id=1, description=query, assigned_agent="knowledge_agent",
                )],
            )

    @staticmethod
    def _fast_route(query: str) -> TaskDecomposition | None:
        """快速关键词路由——命中明确场景直接返回，不需要 LLM。"""
        q = query.strip().lower()

        memory_kw = [
            "记住",
            "以后记得",
            "你还记得",
            "我的偏好",
            "我之前说过",
            "忘记",
            "remember",
            "forget",
        ]
        if any(keyword in q for keyword in memory_kw):
            return TaskDecomposition(
                intent="personal_memory",
                subtasks=[
                    SubTask(
                        id=1,
                        description=query,
                        assigned_agent="memory_agent",
                    )
                ],
            )

        # 纯闲聊和能力询问——不调用 LLM 直接回复
        chat_kw = [
            "你好", "hi", "hello", "嘿", "早啊", "晚安", "再见", "谢谢", "谢了", "thx", "thanks",
            "你会做什么", "你能干嘛", "你有啥用", "你有什么功能", "你是谁", "你是啥",
            "介绍一下自己", "自我介绍",
        ]
        if any(q == kw for kw in chat_kw):
            return TaskDecomposition(intent="chat", direct_reply=(
                "你好！我是 AgentKB，一个运行在本地的个人知识助手。\n\n"
                "我可以帮你：\n"
                "- 📚 **搜索本地知识库** ——上传文档后，用自然语言查找内容\n"
                "- 🌐 **联网搜索** ——获取实时信息\n"
                "- ✍️ **内容创作** ——写文章、脚本、文案、简历、报告\n"
                "- 📋 **任务管理** ——拆解任务、制定计划\n"
                "- 🎓 **学习辅导** ——生成个性化学习路径\n"
                "- 📱 **社媒内容** ——写小红书笔记、抖音脚本、公众号文章\n\n"
                "你可以先说「帮我写一篇小红书笔记」试试！"
            ))

        # 创作类
        creation_kw = ["写", "生成", "创作", "撰写", "帮我写", "起个标题", "文案", "脚本",
                       "简历", "报告", "文章", "短视频", "小红书", "朋友圈"]
        if any(kw in q for kw in creation_kw):
            if any(kw in q for kw in ["小红书", "redbook"]):
                return TaskDecomposition(
                    intent="social_content",
                    subtasks=[SubTask(id=1, description=q, assigned_agent="social_writer")],
                )
            return TaskDecomposition(
                intent="content_creation",
                subtasks=[SubTask(id=1, description=q, assigned_agent="content_creator")],
            )

        # 任务类
        task_kw = ["待办", "提醒", "日程", "计划", "任务", "todo", "项目管理", "分解", "kanban"]
        if any(kw in q for kw in task_kw):
            return TaskDecomposition(
                intent="task_management",
                subtasks=[SubTask(id=1, description=q, assigned_agent="task_manager")],
            )

        # 学习类
        learn_kw = ["学习", "教我", "解释", "什么是", "怎么学", "学习路径", "教程", "入门"]
        if any(kw in q for kw in learn_kw):
            return TaskDecomposition(
                intent="learning",
                subtasks=[SubTask(id=1, description=q, assigned_agent="learning_tutor")],
            )

        # 知识库查询类
        kb_kw = ["知识库", "有哪些文件", "上传的文件", "文件列表", "我的文件",
                 "文档内容", "制度", "规定", "政策", "手册", "笔记",
                 "考勤", "年假", "请假", "报销", "流程", "公司",
                 "文件里", "查一下", "帮我查", "搜索", "检索"]
        if any(kw in q for kw in kb_kw):
            return TaskDecomposition(
                intent="knowledge_search",
                subtasks=[SubTask(id=1, description=q, assigned_agent="knowledge_agent")],
            )

        # 联网搜索类
        web_kw = ["天气", "股价", "新闻", "最新", "今天", "本周", "实时", "热搜"]
        if any(kw in q for kw in web_kw):
            return TaskDecomposition(
                intent="web_search",
                subtasks=[SubTask(id=1, description=q, assigned_agent="knowledge_agent")],
            )

        return None  # 需要 LLM 分析

    @staticmethod
    def _extract_json(text: str) -> str:
        """从 LLM 输出中提取 JSON（容错 markdown 代码块）。"""
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]
        return text.strip()

    async def aggregate(
        self, results: list[dict], query: str
    ) -> str:
        """将多个 Agent 的结果聚合为最终回复。"""
        if len(results) == 1 and results[0].get("success"):
            return results[0].get("output", "")

        # 多 Agent 结果聚合
        parts = []
        for r in results:
            agent = r.get("agent_name", "unknown")
            output = r.get("output", "")
            if output:
                parts.append(f"【{agent}】\n{output}")
        return "\n\n---\n\n".join(parts)
