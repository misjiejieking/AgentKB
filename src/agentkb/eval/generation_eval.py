"""生成质量评估——基于 LLM-as-judge 的 Faithfulness / Answer Relevance / Context Relevance。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


@dataclass
class GenerationMetrics:
    """单条答案的生成质量指标。"""
    faithfulness: float = 0.0  # 答案是否来自检索上下文 (0~1)
    answer_relevance: float = 0.0  # 答案是否回答了问题 (0~1)
    context_relevance: float = 0.0  # 检索上下文是否与问题相关 (0~1)


@dataclass
class GenerationEvalResult:
    """一次生成评估的聚合结果。"""
    metrics: list[GenerationMetrics] = field(default_factory=list)

    @property
    def avg_faithfulness(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.faithfulness for m in self.metrics) / len(self.metrics)

    @property
    def avg_answer_relevance(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.answer_relevance for m in self.metrics) / len(self.metrics)

    @property
    def avg_context_relevance(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(m.context_relevance for m in self.metrics) / len(self.metrics)

    def to_dict(self) -> dict:
        return {
            "avg_faithfulness": round(self.avg_faithfulness, 3),
            "avg_answer_relevance": round(self.avg_answer_relevance, 3),
            "avg_context_relevance": round(self.avg_context_relevance, 3),
            "total_evaluated": len(self.metrics),
            "per_sample": [
                {
                    "faithfulness": round(m.faithfulness, 3),
                    "answer_relevance": round(m.answer_relevance, 3),
                    "context_relevance": round(m.context_relevance, 3),
                }
                for m in self.metrics
            ],
        }


# ══════════════════════════════════════════════════════════════
#  LLM-as-Judge prompts
# ══════════════════════════════════════════════════════════════

FAITHFULNESS_PROMPT = """评估以下答案是否**完全基于**提供的检索上下文。如果答案包含上下文没有的信息（编造），给低分。

## 检索上下文
{contexts}

## 问题
{query}

## 答案
{answer}

## 评分标准
1 分：答案完全是编造的，与上下文无关
2 分：答案大部分是编造的，只用了极少上下文
3 分：答案混合了上下文信息和编造内容
4 分：答案基本来自上下文，有少量不准确
5 分：答案完全来自上下文，准确无误

只回复一个数字（1～5），不要解释。"""


ANSWER_RELEVANCE_PROMPT = """评估以下答案是否**直接回答了**问题。

## 问题
{query}

## 答案
{answer}

## 评分标准
1 分：答案完全偏题，与问题无关
2 分：答案只部分相关，遗漏了问题的核心
3 分：答案基本回答了问题，但不够完整
4 分：答案很好地回答了问题，有轻微不足
5 分：答案完美回答了问题，完整准确

只回复一个数字（1～5），不要解释。"""


CONTEXT_RELEVANCE_PROMPT = """评估检索上下文是否与问题相关。

## 问题
{query}

## 检索上下文
{contexts}

## 评分标准
1 分：上下文与问题完全无关
2 分：上下文只少量相关
3 分：上下文部分相关但不完整
4 分：上下文大部分相关
5 分：上下文高度相关，覆盖了回答所需信息

只回复一个数字（1～5），不要解释。"""


class GenerationEval:
    """生成质量评估器——LLM 打分归一化到 0~1。"""

    def __init__(self, llm_client=None) -> None:
        self._llm = llm_client

    async def evaluate_one(
        self,
        query: str,
        answer: str,
        contexts: list[str],
    ) -> GenerationMetrics:
        """评估单条答案的生成质量。"""
        contexts_text = "\n\n---\n\n".join(contexts[:5]) if contexts else "（无上下文）"

        metrics = GenerationMetrics()

        if self._llm is None:
            return metrics

        try:
            # Faithfulness
            faith_prompt = FAITHFULNESS_PROMPT.format(
                contexts=contexts_text[:3000],
                query=query,
                answer=answer[:2000],
            )
            faith_resp = await self._llm.ainvoke(faith_prompt)
            metrics.faithfulness = _parse_score(faith_resp) / 5.0

            # Answer Relevance
            rel_prompt = ANSWER_RELEVANCE_PROMPT.format(
                query=query,
                answer=answer[:2000],
            )
            rel_resp = await self._llm.ainvoke(rel_prompt)
            metrics.answer_relevance = _parse_score(rel_resp) / 5.0

            # Context Relevance
            ctx_prompt = CONTEXT_RELEVANCE_PROMPT.format(
                query=query,
                contexts=contexts_text[:3000],
            )
            ctx_resp = await self._llm.ainvoke(ctx_prompt)
            metrics.context_relevance = _parse_score(ctx_resp) / 5.0

        except Exception as e:
            logger.error(f"生成评估失败: {e}")

        return metrics

    async def evaluate_batch(
        self,
        items: list[dict],
    ) -> GenerationEvalResult:
        """批量评估。

        items: [{"query": str, "answer": str, "contexts": [str]}, ...]
        """
        result = GenerationEvalResult()
        for i, item in enumerate(items):
            m = await self.evaluate_one(
                query=item["query"],
                answer=item["answer"],
                contexts=item.get("contexts", []),
            )
            result.metrics.append(m)
            if (i + 1) % 5 == 0:
                logger.info(f"  生成评估进度: {i + 1}/{len(items)}")
        return result


def _parse_score(response) -> int:
    """从 LLM 回复中提取 1~5 的评分。"""
    content = response.content if hasattr(response, "content") else str(response)
    import re
    numbers = re.findall(r'[1-5]', content)
    if numbers:
        score = int(numbers[0])
        return max(1, min(5, score))
    return 3  # 默认中等分
