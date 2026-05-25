"""LearningTutor——个性化学习助手 Specialist Agent。

支持: 诊断知识水平、生成学习路径、定制学习材料、知识讲解。
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentkb.agents.base import SpecialistAgent, AgentResult


LEARNING_PATH_PROMPT = """你是资深教育顾问。根据用户的学习目标和当前水平，制定个性化学习路径。

## 用户学习目标
{task}

## 学习路径要求
- 根据目标拆解为 3~7 个阶段
- 每个阶段包含学习主题、推荐资源和实践练习
- 考虑递进关系：基础 → 进阶 → 实战
- 标注每个阶段的预估学习时间（小时）

## 输出格式（JSON）
{{
  "path_name": "学习路径名称",
  "target_level": "目标水平描述",
  "prerequisites": ["前置知识1"],
  "stages": [
    {{
      "stage": 1,
      "title": "阶段标题",
      "topics": ["主题1", "主题2"],
      "resources": ["推荐书籍/课程/文章"],
      "practice": "练习建议",
      "estimated_hours": 10,
      "checkpoint": "如何判断自己掌握了"
    }}
  ],
  "total_hours": 60,
  "tips": ["学习建议1", "学习建议2"]
}}"""


class LearningTutorAgent(SpecialistAgent):
    """个性化学习导师。"""

    @property
    def name(self) -> str:
        return "learning_tutor"

    @property
    def description(self) -> str:
        return "个性化学习指导：诊断知识水平、生成学习路径、定制学习材料、讲解知识概念"

    @property
    def intents(self) -> list[str]:
        return ["learning"]

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

            # 先检索本地知识库是否有相关学习资料
            ctx_text = ""
            try:
                from agentkb.knowledge.retriever import get_retriever
                retriever = get_retriever()
                candidates = retriever.retrieve(task)
                if candidates:
                    contents = [
                        c.get("parent_content") or c.get("content", "")
                        for c in candidates[:3]
                    ]
                    ctx_text = "## 知识库中已有的相关材料\n" + "\n\n".join(contents[:1500])
            except Exception:
                pass

            # 生成学习路径
            prompt = LEARNING_PATH_PROMPT.format(task=task)
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
                    "path_name": "学习路径",
                    "stages": [],
                    "total_hours": 0,
                    "tips": [],
                }

            # 格式化输出
            lines = [
                f"# {plan.get('path_name', '个性化学习路径')}",
                "",
                f"**目标**: {plan.get('target_level', task)}",
                "",
            ]

            prerequisites = plan.get("prerequisites", [])
            if prerequisites:
                lines.append("## 前置知识")
                for p in prerequisites:
                    lines.append(f"- {p}")
                lines.append("")

            lines.append("## 学习路线")
            for stage in plan.get("stages", []):
                lines.append(f"### 第 {stage.get('stage', '?')} 阶段: {stage.get('title', '')}")
                lines.append("")
                lines.append("**学习主题**:")
                for t in stage.get("topics", []):
                    lines.append(f"- {t}")
                lines.append("")
                resources = stage.get("resources", [])
                if resources:
                    lines.append("**推荐资源**:")
                    for r in resources:
                        lines.append(f"- {r}")
                    lines.append("")
                practice = stage.get("practice", "")
                if practice:
                    lines.append(f"**练习**: {practice}")
                lines.append("")
                checkpoint = stage.get("checkpoint", "")
                if checkpoint:
                    lines.append(f"**掌握标准**: {checkpoint}")
                lines.append(f"*预计 {stage.get('estimated_hours', '?')} 小时*")
                lines.append("")

            lines.append(f"## 总时长: ~{plan.get('total_hours', '?')} 小时")

            tips = plan.get("tips", [])
            if tips:
                lines.append("\n## 学习建议")
                for tip in tips:
                    lines.append(f"- {tip}")

            if ctx_text:
                lines.append(f"\n---\n\n{ctx_text}")

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
                output="学习路径规划遇到了问题。",
                elapsed_ms=(time.time() - t0) * 1000,
            )
