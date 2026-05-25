"""TaskManager——任务与工作流管理 Specialist Agent。

支持: 创建待办、自动拆解任务、项目跟踪、进度报告。
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentkb.agents.base import SpecialistAgent, AgentResult


TASK_DECOMPOSE_PROMPT = """你是项目管理专家。将用户的任务或目标拆解为可执行的子任务。

## 用户目标
{task}

## 拆解要求
- 子任务应该是具体的、可执行的操作步骤
- 每个子任务包含优先级（high/medium/low）和估计耗时
- 考虑子任务之间的依赖关系
- 总子任务数不超过 8 个

## 输出格式（JSON）
{{
  "project_name": "项目名称",
  "summary": "一句话概述",
  "subtasks": [
    {{
      "id": 1,
      "title": "子任务标题",
      "description": "详细描述",
      "priority": "high/medium/low",
      "estimated_hours": 2,
      "dependencies": []
    }}
  ],
  "total_estimated_hours": 20,
  "milestones": ["里程碑1", "里程碑2"]
}}"""


class TaskManagerAgent(SpecialistAgent):
    """任务与项目管理专家。"""

    @property
    def name(self) -> str:
        return "task_manager"

    @property
    def description(self) -> str:
        return "管理待办事项：自动拆解任务、创建项目计划、跟踪进度、生成摘要报告"

    @property
    def intents(self) -> list[str]:
        return ["task_management"]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        t0 = time.time()
        tokens_used = 0

        try:
            llm = self.llm

            # 拆解任务
            self.log("拆解任务")
            prompt = TASK_DECOMPOSE_PROMPT.format(task=task)
            resp = await llm.ainvoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            tokens_used = len(prompt) // 2 + len(content) // 2

            try:
                content_clean = content.strip()
                if content_clean.startswith("```"):
                    content_clean = content_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                plan = json.loads(content_clean)
            except json.JSONDecodeError:
                plan = {
                    "project_name": "任务分解",
                    "subtasks": [
                        {"id": i, "title": f"步骤 {i}", "priority": "medium",
                         "estimated_hours": 1, "dependencies": []}
                        for i in range(1, 4)
                    ],
                    "total_estimated_hours": 3,
                    "milestones": [],
                }

            # 格式化输出
            lines = [
                f"## {plan.get('project_name', '任务计划')}",
                "",
                plan.get("summary", ""),
                "",
                "### 子任务列表",
                "",
                "| # | 任务 | 优先级 | 预计耗时 | 依赖 |",
                "|---|------|--------|----------|------|",
            ]

            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
            for st in plan.get("subtasks", []):
                icon = priority_icon.get(st.get("priority", "medium"), "⚪")
                deps = ", ".join(str(d) for d in st.get("dependencies", [])) or "无"
                lines.append(
                    f"| {st['id']} | {st['title']} | {icon} {st.get('priority', 'medium')} "
                    f"| {st.get('estimated_hours', 1)}h | {deps} |"
                )

            lines.append("")
            lines.append(f"**总预计耗时**: {plan.get('total_estimated_hours', 0)} 小时")

            milestones = plan.get("milestones", [])
            if milestones:
                lines.append("\n### 关键里程碑")
                for i, m in enumerate(milestones, 1):
                    lines.append(f"{i}. {m}")

            lines.append("\n---\n*提示：你可以对着某个子任务说「帮我做第X步」，我会协助你完成。*")

            output = "\n".join(lines)

            return AgentResult(
                agent_name=self.name,
                success=True,
                output=output,
                data=plan,
                tokens_used=tokens_used,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                output="任务规划过程中遇到了问题。",
                elapsed_ms=(time.time() - t0) * 1000,
            )
