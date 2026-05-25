"""Reflection / Self-Critique 模块——让 Agent 自我审查输出质量并自动修正。

Reflection 流程:
  1. 检查输出是否回答了用户问题
  2. 检查是否有幻觉（与检索上下文不一致）
  3. 检查是否遗漏关键信息
  4. 必要时自动修订
"""

from __future__ import annotations

from typing import Any

from loguru import logger

REFLECTION_PROMPT = """你是质量审查员。审查以下 Agent 对用户问题的回答。

## 用户问题
{query}

## Agent 回答
{agent_output}

## 审查标准
1. **完整性**: 是否回答了用户的所有问题？
2. **准确性**: 是否有编造或幻觉内容？
3. **相关性**: 是否直接回应用户问题，没有跑题？
4. **可操作性**: 对用户是否有实际帮助价值？

## 输出格式（严格 JSON）
{{
  "needs_revision": true/false,
  "critique": "一句话指出问题",
  "revised_output": "如果 needs_revision=true，给出修订后的完整回答（否则 null）",
  "confidence": 0.0~1.0
}}"""


class ReflectionModule:
    """自检模块——审查 Agent 输出并提出改进。"""

    def __init__(self, enabled: bool = True, max_rounds: int = 2) -> None:
        self._enabled = enabled
        self._max_rounds = max_rounds

    async def critique(
        self,
        query: str,
        agent_results: list[Any],
        llm_client=None,
    ) -> dict[str, Any]:
        """审查 Agent 输出。

        Returns:
          {
            "needs_revision": bool,
            "critique": str,
            "revised_output": str | None,
            "confidence": float,
          }
        """
        if not self._enabled:
            return {"needs_revision": False}

        # 合并所有 Agent 输出
        outputs = []
        errors = []
        for r in agent_results:
            if hasattr(r, "output") and r.output:
                outputs.append(r.output)
            if hasattr(r, "error") and r.error:
                errors.append(r.error)

        combined = "\n\n---\n\n".join(outputs) if outputs else "（无输出）"

        # 明显的错误——不需要 LLM 判断
        if errors and not outputs:
            return {
                "needs_revision": True,
                "critique": f"所有 Agent 均失败: {errors}",
                "revised_output": "抱歉，处理您的请求时遇到了问题，请稍后重试。",
                "confidence": 0.0,
            }

        # 如果输出足够短且完整，跳过 Reflection 节省 token
        if len(combined) < 100 and len(outputs) == 1:
            return {"needs_revision": False}

        # LLM 审查
        llm = llm_client
        if llm is None:
            from agentkb.llm.factory import get_chat_model
            llm = get_chat_model(streaming=False)

        try:
            import json
            prompt = REFLECTION_PROMPT.format(query=query, agent_output=combined[:3000])
            response = await llm.ainvoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0]
            data = json.loads(content)

            logger.debug(
                f"Reflection: needs_revision={data.get('needs_revision')}, "
                f"confidence={data.get('confidence', 0)}"
            )
            return data
        except Exception as e:
            logger.warning(f"Reflection 审查失败: {e}")
            return {"needs_revision": False}
