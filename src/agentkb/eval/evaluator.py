"""评估编排器——串联检索管线与指标计算，支持两次评估对比。

一次完整评估的执行流程:
  1. 加载测试集 (TestSet)
  2. 对每条 query 执行完整检索管线 (HybridRetriever → Reranker)
  3. 收集检索结果的 chunk_id 列表
  4. 调用 compute_metrics() 计算指标
  5. 生成 EvalResult + DiffReport

对比评估 (compare):
  将 baseline 和 current 的每个指标做 diff，标记哪些指标提升/退化。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

from agentkb.eval.metrics import EvalResult, compute_metrics
from agentkb.eval.testset import TestSet
from agentkb.eval.generation_eval import GenerationEval
from agentkb.config.settings import Settings


# ══════════════════════════════════════════════════════════════
#  差异报告
# ══════════════════════════════════════════════════════════════

@dataclass
class MetricDiff:
    """单个指标的差异项。"""
    name: str  # 指标名称，如 "Recall@5"
    baseline: float
    current: float
    delta: float  # current - baseline
    direction: str  # "up" | "down" | "stable"

    @property
    def is_improvement(self) -> bool:
        """正差异 = 提升（所有指标越大越好）。"""
        return self.delta > 0.001

    @property
    def is_regression(self) -> bool:
        """负差异 = 退化。"""
        return self.delta < -0.001


@dataclass
class DiffReport:
    """对比报告——基线 vs 当前的全面对比。

    解读:
      查看 .diffs 列表中 delta 为正（绿色，提升）和 delta 为负（红色，退化）的指标
      重点关注退化项，确认改动没有损害检索质量
    """
    baseline_name: str = ""
    current_name: str = ""
    diffs: list[MetricDiff] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "baseline_name": self.baseline_name,
            "current_name": self.current_name,
            "diffs": [
                {
                    "metric": d.name,
                    "baseline": round(d.baseline, 4),
                    "current": round(d.current, 4),
                    "delta": round(d.delta, 4),
                    "direction": d.direction,
                }
                for d in self.diffs
            ],
        }


# ══════════════════════════════════════════════════════════════
#  评估器
# ══════════════════════════════════════════════════════════════

class Evaluator:
    """检索评估编排器。

    使用示例:
        evaluator = Evaluator()
        result = evaluator.evaluate(testset)
        print(f"Recall@5:  {result.recall_at_k[5]:.3f}")
        print(f"Recall@20: {result.recall_at_k[20]:.3f}")
        print(f"MRR:        {result.mrr:.3f}")
    """

    def __init__(self, skip_reranker: bool = False) -> None:
        """Args:
            skip_reranker: True 时只用混合检索的 RRF 分数排序，跳过 reranker。
                           用于对比「reranker 有没有用」。
        """
        self._skip_reranker = skip_reranker

    def evaluate(self, testset: TestSet) -> EvalResult:
        """对测试集执行完整评估。

        对每条 query:
          1. HybridRetriever.retrieve(query) → 候选集
          2. RerankerService.rerank(query, candidates) → 精排（可选跳过）
          3. 收集排序后的 chunk_id 列表
        然后:
          4. compute_metrics() → EvalResult
        """
        from agentkb.knowledge.retriever import get_retriever
        from agentkb.knowledge.reranker import get_reranker

        cfg = Settings.load()
        retriever = get_retriever()
        reranker = None
        if not self._skip_reranker:
            try:
                reranker = get_reranker()
            except Exception as e:
                logger.warning(f"Reranker 初始化失败，跳过精排: {e}")

        queries = []
        relevant_ids_per_query = []
        retrieved_ids_per_query = []

        logger.info(f"开始评估: {len(testset.items)} 条查询（{'含' if reranker else '跳过'} reranker）")

        for idx, item in enumerate(testset.items):
            queries.append(item.query)
            relevant_ids_per_query.append(set(item.relevant_chunk_ids))

            try:
                # 1. 混合检索 → 候选集（全部 candidate_k 条，用于多级 Recall@K）
                candidates = retriever.retrieve(item.query)

                if candidates and reranker:
                    # 2. 精排 → 将 reranker 排序靠前的放前面，其余按 RRF 分数拼接
                    try:
                        ranked = reranker.rerank(item.query, candidates, top_k=cfg.retrieval_final_k)
                        ranked_ids = {r["id"] for r in ranked}
                        remaining = sorted(
                            [c for c in candidates if c["id"] not in ranked_ids],
                            key=lambda x: x.get("rrf_score", 0), reverse=True,
                        )
                        retrieved_ids = [r["id"] for r in ranked] + [c["id"] for c in remaining]
                    except Exception as e:
                        logger.warning(f"Reranker 失败，回退到 RRF 排序: {e}")
                        sorted_candidates = sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)
                        retrieved_ids = [c["id"] for c in sorted_candidates]
                elif candidates:
                    # 无 reranker，全部按 RRF score 排序
                    sorted_candidates = sorted(candidates, key=lambda x: x.get("rrf_score", 0), reverse=True)
                    retrieved_ids = [c["id"] for c in sorted_candidates]
                else:
                    retrieved_ids = []

            except Exception as e:
                logger.error(f"检索失败 [{item.query}]: {e}")
                retrieved_ids = []

            retrieved_ids_per_query.append(retrieved_ids)

            if (idx + 1) % 10 == 0:
                logger.info(f"  进度: {idx + 1}/{len(testset.items)}")

        result = compute_metrics(
            queries=queries,
            relevant_ids_per_query=relevant_ids_per_query,
            retrieved_ids_per_query=retrieved_ids_per_query,
            k_values=cfg.eval_retrieval_k_values,
        )

        logger.info(
            f"评估完成: "
            f"Recall@5={result.recall_at_k.get(5, 0):.3f}, "
            f"Recall@20={result.recall_at_k.get(20, 0):.3f}, "
            f"MRR={result.mrr:.3f}"
        )
        return result

    @staticmethod
    def diff(
        baseline: EvalResult,
        current: EvalResult,
        baseline_name: str = "baseline",
        current_name: str = "current",
    ) -> DiffReport:
        """对比两次评估结果，生成差异报告。

        返回 DiffReport，其中每个 MetricDiff 包含:
          - delta 正数 = 提升 (✅)
          - delta 负数 = 退化 (❌)
          - delta 接近 0 = 持平 (➡️)
        """
        diffs = []

        # Recall@K
        for k in baseline.k_values:
            b = baseline.recall_at_k.get(k, 0)
            c = current.recall_at_k.get(k, 0)
            diffs.append(_make_diff(f"Recall@{k}", b, c))

        # Precision@K
        for k in baseline.k_values:
            b = baseline.precision_at_k.get(k, 0)
            c = current.precision_at_k.get(k, 0)
            diffs.append(_make_diff(f"Precision@{k}", b, c))

        # MRR
        diffs.append(_make_diff("MRR", baseline.mrr, current.mrr))

        # NDCG@K
        for k in baseline.k_values:
            b = baseline.ndcg_at_k.get(k, 0)
            c = current.ndcg_at_k.get(k, 0)
            diffs.append(_make_diff(f"NDCG@{k}", b, c))

        return DiffReport(
            baseline_name=baseline_name,
            current_name=current_name,
            diffs=diffs,
        )


    async def evaluate_full(
        self,
        testset: TestSet,
        llm_client=None,
    ) -> dict:
        """完整评估：检索指标 + 生成质量。

        先跑检索评估，再对每条 query 用 Agent 生成答案，评估生成质量。
        """
        logger.info("开始完整评估（检索 + 生成质量）")

        # 1. 检索评估
        retrieval_result = self.evaluate(testset)

        # 2. 生成评估
        from agentkb.knowledge.retriever import get_retriever
        retriever = get_retriever()

        gen_eval = GenerationEval(llm_client=llm_client)
        eval_items = []

        for item in testset.items[:10]:  # 默认评估前 10 条，控制成本
            try:
                candidates = retriever.retrieve(item.query)
                contexts = [
                    c.get("parent_content") or c.get("content", "")[:1024]
                    for c in (candidates or [])[:5]
                ]

                # 用评估专用的 judge LLM 生成答案
                if llm_client:
                    rag_prompt = f"基于以下上下文回答问题：\n\n上下文：\n{chr(10).join(contexts[:3])}\n\n问题：{item.query}\n\n答案："
                    resp = await llm_client.ainvoke(rag_prompt)
                    answer = resp.content if hasattr(resp, "content") else str(resp)
                else:
                    answer = "（无 LLM 可用）"

                eval_items.append({
                    "query": item.query,
                    "answer": answer,
                    "contexts": contexts,
                })
            except Exception as e:
                logger.error(f"生成评估项构建失败 [{item.query}]: {e}")

        gen_result = await gen_eval.evaluate_batch(eval_items)

        return {
            "retrieval": {
                "recall_at_k": retrieval_result.recall_at_k,
                "mrr": retrieval_result.mrr,
                "ndcg_at_k": retrieval_result.ndcg_at_k,
            },
            "generation": gen_result.to_dict(),
        }


def _make_diff(name: str, baseline: float, current: float) -> MetricDiff:
    delta = current - baseline
    if delta > 0.001:
        direction = "up"
    elif delta < -0.001:
        direction = "down"
    else:
        direction = "stable"
    return MetricDiff(name=name, baseline=baseline, current=current, delta=delta, direction=direction)
