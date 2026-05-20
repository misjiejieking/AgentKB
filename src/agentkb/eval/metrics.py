"""检索评估指标——Recall@K、Precision@K、MRR、NDCG@K。

每个指标都有明确的中文注释，说明：
  - 该指标衡量什么（一句话）
  - 计算公式
  - 数值范围与如何解读
  - 适用场景（什么时候该关注这个指标）
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueryMetrics:
    """单条 query 的评估明细，用于定位差 query。"""
    query: str
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    relevant_count: int = 0  # 标注的相关 chunk 数量
    first_relevant_rank: int | None = None  # 第一个正确答案的排名（1-based），None 表示没命中


# ══════════════════════════════════════════════════════════════
#  单 query 级别计算
# ══════════════════════════════════════════════════════════════

def _compute_query_metrics(
    query: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k_values: list[int],
) -> QueryMetrics:
    """计算单条 query 的完整指标。"""
    # 找到所有相关 chunk 在检索结果中的排名（1-based）
    hit_ranks = []
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            hit_ranks.append(rank)

    first_rank = hit_ranks[0] if hit_ranks else None
    relevant_count = len(relevant_ids)

    # Recall@K: 正确答案有多少出现在 top-K 中
    # 公式: |retrieved_topK ∩ relevant| / |relevant|
    # 范围: 0~1，越高越好。K=5 时 0.8 意味着遗漏了 20% 的相关内容
    # 关注时机: 调检索参数后，Recall@5 不能降（查出正确答案的能力不能变差）
    recall = {}
    for k in k_values:
        hits_in_topk = sum(1 for r in hit_ranks if r <= k)
        recall[k] = hits_in_topk / relevant_count if relevant_count > 0 else 0.0

    # Precision@K: top-K 中有多少是真正相关的
    # 公式: |retrieved_topK ∩ relevant| / K
    # 范围: 0~1，越高越好。P@5=0.4 意味着翻 5 条结果平均只有 2 条有用
    # 关注时机: 与 Recall 配合看——Recall 高 Precision 低说明模型在"瞎蒙"
    precision = {}
    for k in k_values:
        hits_in_topk = sum(1 for r in hit_ranks if r <= k)
        precision[k] = hits_in_topk / k

    # Reciprocal Rank: 第一个正确答案排在第几位
    # 公式: 1 / rank_of_first_relevant
    # 范围: 0~1，RR=1.0 表示第一个结果就是对的，RR=0.1 表示翻了 10 条才找到
    # 关注时机: 衡量用户体验——用户只关心"第一个有用的结果在哪"
    rr = 1.0 / first_rank if first_rank else 0.0

    # NDCG@K: 排序质量的综合指标
    # 公式: DCG@K = Σ(relevance_i / log2(i+1)), NDCG = DCG / IDCG
    # 范围: 0~1，越高越好。比 MRR 更全面，对排名靠前的结果更敏感
    # 关注时机: 当你不仅关心"对的结果在不在"，还关心"对的结果排得够不够靠前"
    ndcg = {}
    for k in k_values:
        dcg = 0.0
        for i, cid in enumerate(retrieved_ids[:k], start=1):
            if cid in relevant_ids:
                dcg += 1.0 / math.log2(i + 1)
        # IDCG: 理想情况——所有相关 chunk 排在前面
        idcg = 0.0
        for i in range(1, min(relevant_count, k) + 1):
            idcg += 1.0 / math.log2(i + 1)
        ndcg[k] = dcg / idcg if idcg > 0 else 0.0

    return QueryMetrics(
        query=query,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        relevant_count=relevant_count,
        first_relevant_rank=first_rank,
    )


# ══════════════════════════════════════════════════════════════
#  聚合级评估结果
# ══════════════════════════════════════════════════════════════

@dataclass
class EvalResult:
    """一次评估的完整结果——包含聚合指标与每条 query 的明细。

    字段说明:
      - recall_at_k[k]     : 所有 query 的 Recall@k 平均值（宏观指标，衡量检索覆盖度）
      - precision_at_k[k]  : 所有 query 的 Precision@k 平均值（宏观指标，衡量检索准确度）
      - mrr                 : Mean Reciprocal Rank（宏观指标，衡量第一个正确答案的排名）
      - ndcg_at_k[k]        : 所有 query 的 NDCG@k 平均值（宏观指标，衡量排序质量）
      - per_query           : 每条 query 的详细指标（用于定位问题 query）

    解读策略:
      优先看 Recall@5 → 太低说明检索根本找不对东西
      再看 Recall@20 → 太低说明候选池太小，需要增大 candidate_k
      然后看 MRR → 太低说明第一个正确答案排太靠后，润色/重排没做好
      最后看 NDCG → 综合判断排序质量
    """
    k_values: list[int]
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    per_query: list[QueryMetrics] = field(default_factory=list)


def compute_metrics(
    queries: list[str],
    relevant_ids_per_query: list[set[str]],
    retrieved_ids_per_query: list[list[str]],
    k_values: list[int] | None = None,
) -> EvalResult:
    """计算检索评估的完整指标。

    Args:
        queries:              查询文本列表
        relevant_ids_per_query: 每条查询对应的正确答案 chunk_id 集合（标注集）
        retrieved_ids_per_query: 每条查询的系统检索结果 chunk_id 列表（按分数降序）
        k_values:              评估的 K 值列表，默认 [5, 10, 20]

    Returns:
        EvalResult 包含聚合平均值与每条 query 的详细指标

    使用示例:
        result = compute_metrics(
            queries=["公司年假有多少天"],
            relevant_ids_per_query=[{"chunk_3", "chunk_7"}],
            retrieved_ids_per_query=[["chunk_7", "chunk_1", "chunk_3", "chunk_5", "chunk_2"]],
        )
        print(f"Recall@5: {result.recall_at_k[5]:.3f}")
        print(f"MRR:       {result.mrr:.3f}")
    """
    if k_values is None:
        k_values = [5, 10, 20]

    assert len(queries) == len(relevant_ids_per_query) == len(retrieved_ids_per_query), \
        "queries、relevant_ids_per_query、retrieved_ids_per_query 长度必须一致"

    per_query = []
    for q, rel_ids, ret_ids in zip(queries, relevant_ids_per_query, retrieved_ids_per_query):
        per_query.append(_compute_query_metrics(q, ret_ids, rel_ids, k_values))

    n = len(per_query)
    result = EvalResult(k_values=k_values, per_query=per_query)

    # 聚合平均
    for k in k_values:
        result.recall_at_k[k] = sum(q.recall_at_k[k] for q in per_query) / n if n else 0.0
        result.precision_at_k[k] = sum(q.precision_at_k[k] for q in per_query) / n if n else 0.0
        result.ndcg_at_k[k] = sum(q.ndcg_at_k[k] for q in per_query) / n if n else 0.0

    result.mrr = sum(q.reciprocal_rank for q in per_query) / n if n else 0.0

    return result
