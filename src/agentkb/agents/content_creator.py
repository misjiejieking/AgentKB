"""ContentCreator——智能内容创作 Specialist Agent。

多 Agent 内部协作: 规划 → 写作 → 优化（三阶段 Pipeline）。
支持: 文章、短视频脚本、朋友圈文案、简历、报告。
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentkb.agents.base import SpecialistAgent, AgentResult


CONTENT_PLANNER_PROMPT = """你是内容规划师。根据用户需求，制定内容创作大纲。

## 用户需求
{task}

## 参考上下文
{context}

## 输出格式（JSON）
{{
  "content_type": "文章/脚本/文案/简历/报告",
  "title": "标题",
  "sections": [
    {{"heading": "章节标题", "key_points": ["要点1", "要点2"], "estimated_words": 100}}
  ],
  "tone": "正式/轻松/专业/幽默",
  "target_audience": "目标读者"
}}"""

CONTENT_WRITER_PROMPT = """你是专业写作者。根据大纲撰写内容。

## 大纲
{outline}

## 写作要求
- 风格: {tone}
- 目标读者: {audience}
- 如果有参考上下文，请基于事实写作

## 参考上下文
{context}

## 写作（直接输出内容，不要输出其他）"""

CONTENT_POLISH_PROMPT = """你是资深编辑。优化以下内容使其更流畅、更有吸引力。

## 原始内容
{content}

## 优化要求
- 修正语法和表达
- 增强可读性
- 保持原风格: {tone}
- 添加适当的过渡和强调

## 优化后内容（直接输出）"""


class ContentCreatorAgent(SpecialistAgent):
    """智能内容创作助手——规划 → 写作 → 优化。"""

    @property
    def name(self) -> str:
        return "content_creator"

    @property
    def description(self) -> str:
        return "生成各类内容：文章、短视频脚本、朋友圈文案、简历、工作报告、演讲稿等"

    @property
    def intents(self) -> list[str]:
        return ["content_creation", "hybrid"]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        t0 = time.time()
        context = context or {}
        total_tokens = 0
        steps = []

        try:
            llm = self.llm

            # 获取已完成子任务中的上下文（如知识检索结果）
            ctx_text = ""
            completed = context.get("completed_subtasks", {})
            for st_id, st_data in completed.items():
                ctx_text += st_data.get("output", "")[:1500] + "\n"

            # Phase 1: 规划
            self.log("Phase 1: 内容规划")
            plan_prompt = CONTENT_PLANNER_PROMPT.format(
                task=task,
                context=ctx_text[:2000] or "（无参考上下文）",
            )
            plan_resp = await llm.ainvoke(plan_prompt)
            plan_text = plan_resp.content if hasattr(plan_resp, "content") else str(plan_resp)
            total_tokens += len(plan_prompt) // 2 + len(plan_text) // 2

            try:
                plan_text_clean = plan_text.strip()
                if plan_text_clean.startswith("```"):
                    plan_text_clean = plan_text_clean.split("\n", 1)[-1].rsplit("```", 1)[0]
                plan = json.loads(plan_text_clean)
            except json.JSONDecodeError:
                plan = {
                    "content_type": "文章",
                    "title": "",
                    "sections": [{"heading": "正文", "key_points": [], "estimated_words": 500}],
                    "tone": "专业",
                    "target_audience": "通用",
                }
            steps.append({"phase": "planning", "plan": plan})

            # Phase 2: 写作
            self.log(f"Phase 2: 写作 ({plan.get('content_type', '文章')})")
            outline = "\n".join(
                f"## {s['heading']}\n- " + "\n- ".join(s.get("key_points", []))
                for s in plan.get("sections", [])
            )
            write_prompt = CONTENT_WRITER_PROMPT.format(
                outline=outline,
                tone=plan.get("tone", "专业"),
                audience=plan.get("target_audience", "通用"),
                context=ctx_text[:2000] or "（无参考上下文）",
            )
            write_resp = await llm.ainvoke(write_prompt)
            draft = write_resp.content if hasattr(write_resp, "content") else str(write_resp)
            total_tokens += len(write_prompt) // 2 + len(draft) // 2
            steps.append({"phase": "writing", "draft_length": len(draft)})

            # Phase 3: 优化
            self.log("Phase 3: 内容优化")
            polish_prompt = CONTENT_POLISH_PROMPT.format(
                content=draft,
                tone=plan.get("tone", "专业"),
            )
            polish_resp = await llm.ainvoke(polish_prompt)
            final = polish_resp.content if hasattr(polish_resp, "content") else str(polish_resp)
            total_tokens += len(polish_prompt) // 2 + len(final) // 2
            steps.append({"phase": "polishing"})

            # 添加标题
            title = plan.get("title", "")
            if title:
                final = f"# {title}\n\n{final}"

            return AgentResult(
                agent_name=self.name,
                success=True,
                output=final,
                data={
                    "content_type": plan.get("content_type", "文章"),
                    "word_count": len(final),
                    "phases": ["planning", "writing", "polishing"],
                },
                intermediate_steps=steps,
                tokens_used=total_tokens,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                output="内容创作过程中遇到了问题，请稍后重试。",
                elapsed_ms=(time.time() - t0) * 1000,
            )
