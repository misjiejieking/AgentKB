"""评估报告生成——带详细指标注释的 Markdown / JSON 报告。

每个指标在报告中都有明确的中文注释，说明:
  - 该指标衡量什么
  - 如何解读当前数值
  - 如果数值偏低应该关注什么问题
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from agentkb.eval.metrics import EvalResult
from agentkb.eval.evaluator import DiffReport


# ══════════════════════════════════════════════════════════════
#  指标注释表（中文）
# ══════════════════════════════════════════════════════════════

_METRIC_DESCRIPTIONS: dict[str, str] = {
    "Recall": (
        "召回率 —— 正确答案有多少被检索系统找到了？\n"
        "  公式: |检索结果 ∩ 标注答案| / |标注答案|\n"
        "  范围: 0~1，越高越好\n"
        "  解读:\n"
        "    Recall@5  < 0.5 → 多数正确答案连 top-5 都进不去，检索能力严重不足\n"
        "    Recall@5  0.6-0.8 → 中等水平，仍有较多遗漏\n"
        "    Recall@5  > 0.9 → 优秀，绝大多数正确答案在前 5 名\n"
        "    Recall@20 > 0.95 → 候选池大小足够\n"
        "  优化方向: 调整 embedding 模型、增加 BM25 权重、优化分块策略"
    ),
    "Precision": (
        "精确率 —— top-K 结果中有多少是真正相关的？\n"
        "  公式: |检索结果 ∩ 标注答案| / K\n"
        "  范围: 0~1，越高越好\n"
        "  解读:\n"
        "    与 Recall 配合看——Recall 高 Precision 低说明检索过于激进，放进来很多噪声\n"
        "    Precision@5 < 0.3 → 用户在 top-5 中大半是无关内容\n"
        "  优化方向: 提高 score threshold、加强 reranker、增加查询意图理解"
    ),
    "MRR": (
        "平均倒数排名 (Mean Reciprocal Rank) —— 第一个正确答案排在第几位？\n"
        "  公式: mean(1 / rank_of_first_relevant)\n"
        "  范围: 0~1，越高越好\n"
        "  解读:\n"
        "    MRR > 0.8  → 用户几乎不需要翻页就能看到答案（排名 1~2 位）\n"
        "    MRR 0.4-0.6 → 用户需要翻 2~3 条才找到有用结果\n"
        "    MRR < 0.3  → 用户体验差，第一个正确答案常常排在很后面\n"
        "  优化方向: 改进 reranker 精度、调整 RRF 权重、优化 query 理解"
    ),
    "NDCG": (
        "归一化折损累计增益 (NDCG) —— 排序质量的综合指标\n"
        "  公式: DCG / IDCG，对排名靠前的位置赋予更高权重\n"
        "  范围: 0~1，越高越好\n"
        "  解读:\n"
        "    比 MRR 更全面——MRR 只看第一个，NDCG 看全部相关结果的排序\n"
        "    NDCG@10 > 0.7 → 排序质量较好\n"
        "    NDCG@10 < 0.4 → 即使找到了正确答案，排序顺序不好\n"
        "  优化方向: reranker 微调、RRF 融合策略、multi-stage 排序"
    ),
}

_DIFF_NOTE = (
    "差异解读:\n"
    "  delta > 0  (✅ 提升): 改动带来了正向效果\n"
    "  delta < 0  (❌ 退化): 改动损害了检索质量，需排查\n"
    "  delta ≈ 0  (➡️ 持平): 改动对该指标无明显影响"
)


# ══════════════════════════════════════════════════════════════
#  Markdown 报告
# ══════════════════════════════════════════════════════════════

def render_result_markdown(result: EvalResult, title: str = "检索评估报告") -> str:
    """生成带指标注释的单次评估 Markdown 报告。

    报告结构:
      1. 宏观指标汇总（含每个指标的中文解释）
      2. 按 K 值拆分的 Recall/Precision/NDCG
      3. 每个 K 值的解读建议
      4. 差 query 列表（Recall@5=0 的 query）
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        f"# {title}",
        f"生成时间: {now}",
        f"评估查询数: {len(result.per_query)}",
        f"评估 K 值: {result.k_values}",
        "",
        "---",
        "",
        "## 一、指标详解（含中文注释）",
        "",
    ]

    # Recall
    lines.append(f"### 1. Recall@K — {_METRIC_DESCRIPTIONS['Recall']}")
    lines.append("")
    for k in result.k_values:
        v = result.recall_at_k.get(k, 0)
        lines.append(f"- **Recall@{k}**: `{v:.4f}`")
    lines.append("")

    # Precision
    lines.append(f"### 2. Precision@K — {_METRIC_DESCRIPTIONS['Precision']}")
    lines.append("")
    for k in result.k_values:
        v = result.precision_at_k.get(k, 0)
        lines.append(f"- **Precision@{k}**: `{v:.4f}`")
    lines.append("")

    # MRR
    lines.append(f"### 3. MRR — {_METRIC_DESCRIPTIONS['MRR']}")
    lines.append("")
    lines.append(f"- **MRR**: `{result.mrr:.4f}`")
    lines.append("")

    # NDCG
    lines.append(f"### 4. NDCG@K — {_METRIC_DESCRIPTIONS['NDCG']}")
    lines.append("")
    for k in result.k_values:
        v = result.ndcg_at_k.get(k, 0)
        lines.append(f"- **NDCG@{k}**: `{v:.4f}`")
    lines.append("")

    # 汇总表
    lines.extend([
        "---",
        "",
        "## 二、指标汇总表",
        "",
        "| 指标 | 数值 | 评估 |",
        "|---|---|---|",
    ])
    for k in result.k_values:
        r = result.recall_at_k.get(k, 0)
        status = _status_icon(r, 0.9, 0.6)
        lines.append(f"| Recall@{k} | `{r:.4f}` | {status} |")
    for k in result.k_values:
        p = result.precision_at_k.get(k, 0)
        status = _status_icon(p, 0.7, 0.4)
        lines.append(f"| Precision@{k} | `{p:.4f}` | {status} |")
    mrr_status = _status_icon(result.mrr, 0.8, 0.4)
    lines.append(f"| MRR | `{result.mrr:.4f}` | {mrr_status} |")
    for k in result.k_values:
        n = result.ndcg_at_k.get(k, 0)
        status = _status_icon(n, 0.7, 0.4)
        lines.append(f"| NDCG@{k} | `{n:.4f}` | {status} |")
    lines.append("")

    # 差 query
    lines.extend([
        "---",
        "",
        "## 三、低质量查询（Recall@5 = 0）",
        "",
        "以下查询在前 5 个结果中没有命中任何正确答案，需重点关注:",
        "",
    ])
    bad_queries = [q for q in result.per_query if q.recall_at_k.get(5, 0) == 0]
    if bad_queries:
        for i, qm in enumerate(bad_queries, 1):
            rank_str = f"最优排名: {qm.first_relevant_rank}" if qm.first_relevant_rank else "无命中"
            lines.append(f"{i}. **{qm.query}** — {rank_str}")
    else:
        lines.append("> 所有查询 Recall@5 > 0，检索覆盖良好 ✅")
    lines.append("")

    return "\n".join(lines)


def render_diff_markdown(diff: DiffReport) -> str:
    """生成对比差异 Markdown 报告。

    报告结构:
      1. 提升的指标（绿色 ✅）
      2. 退化的指标（红色 ❌）
      3. 持平的指标
      4. 综合建议
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "# 评估对比报告",
        f"生成时间: {now}",
        f"基线: {diff.baseline_name}",
        f"对比: {diff.current_name}",
        "",
        _DIFF_NOTE,
        "",
        "---",
        "",
    ]

    improved = [d for d in diff.diffs if d.is_improvement]
    regressed = [d for d in diff.diffs if d.is_regression]
    stable = [d for d in diff.diffs if not d.is_improvement and not d.is_regression]

    # 提升
    lines.append("## ✅ 提升")
    if improved:
        lines.append("")
        lines.append("| 指标 | 基线 | 当前 | Δ |")
        lines.append("|---|---|---|---|")
        for d in improved:
            lines.append(f"| {d.name} | `{d.baseline:.4f}` | `{d.current:.4f}` | `+{d.delta:.4f}` |")
    else:
        lines.append("> 无显著提升")
    lines.append("")

    # 退化
    lines.append("## ❌ 退化")
    if regressed:
        lines.append("")
        lines.append("| 指标 | 基线 | 当前 | Δ |")
        lines.append("|---|---|---|---|")
        for d in regressed:
            lines.append(f"| {d.name} | `{d.baseline:.4f}` | `{d.current:.4f}` | `{d.delta:.4f}` |")
        lines.append("")
        lines.append("> ⚠️ 退化项需排查根因——可能是分块策略变化、权重调整、或模型参数改动导致")
    else:
        lines.append("> 无显著退化 ✅")
    lines.append("")

    # 持平
    lines.append("## ➡️ 持平")
    if stable:
        lines.append("")
        lines.append(", ".join(d.name for d in stable))
    else:
        lines.append("> 所有指标均有变化")
    lines.append("")

    return "\n".join(lines)


def save_json_report(result: EvalResult, path: str | Path, diff: DiffReport | None = None) -> None:
    """保存 JSON 格式的评估结果。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            "recall_at_k": {str(k): round(v, 4) for k, v in result.recall_at_k.items()},
            "precision_at_k": {str(k): round(v, 4) for k, v in result.precision_at_k.items()},
            "mrr": round(result.mrr, 4),
            "ndcg_at_k": {str(k): round(v, 4) for k, v in result.ndcg_at_k.items()},
        },
        "per_query": [
            {
                "query": qm.query,
                "recall_at_k": {str(k): round(v, 4) for k, v in qm.recall_at_k.items()},
                "reciprocal_rank": round(qm.reciprocal_rank, 4),
                "first_relevant_rank": qm.first_relevant_rank,
                "relevant_count": qm.relevant_count,
            }
            for qm in result.per_query
        ],
    }

    if diff:
        data["diff"] = diff.to_dict()

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"JSON 报告已保存: {path}")


# ══════════════════════════════════════════════════════════════
#  辅助
# ══════════════════════════════════════════════════════════════

def _status_icon(value: float, good: float, warn: float) -> str:
    if value >= good:
        return "✅ 优秀"
    elif value >= warn:
        return "⚠️ 一般"
    return "❌ 需改善"
