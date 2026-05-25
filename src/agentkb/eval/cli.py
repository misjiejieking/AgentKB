"""评估框架命令行入口。

使用方式:
  # 生成银标测试集（从知识库采样 chunk，LLM 生成问题）
  python -m agentkb.eval generate --sample-size 50

  # 跑评估（使用默认测试集）
  python -m agentkb.eval run

  # 跑评估但跳过 reranker（对比 reranker 效果）
  python -m agentkb.eval run --skip-reranker --output no_rerank.json

  # 对比两次评估
  python -m agentkb.eval compare --baseline baseline.json --current after.json

  # 生成报告
  python -m agentkb.eval report --input eval_result.json --format md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from agentkb.eval.reporter import (
    render_result_markdown,
    render_diff_markdown,
    save_json_report,
)
from agentkb.config.settings import Settings


def _ensure_dirs() -> None:
    for d in ["data/eval", "data/eval/reports"]:
        Path(d).mkdir(parents=True, exist_ok=True)


def cmd_generate(args) -> None:
    """生成银标测试集。"""
    logger.info("正在生成银标测试集……")

    from agentkb.storage.pg_database import get_db
    from agentkb.eval.testset import TestSet

    cfg = Settings.load()
    db = get_db()

    testset = TestSet.generate(
        db=db,
        sample_size=args.sample_size or cfg.eval_generation_sample_size,
        questions_per_chunk=cfg.eval_questions_per_chunk,
        seed=args.seed,
    )

    output = args.output or cfg.eval_testset_path
    testset.save(output)

    # 验证
    check = testset.validate()
    if check["valid"]:
        logger.info("测试集验证通过 ✅")
    else:
        logger.warning(f"测试集验证发现问题:\n" + "\n".join(f"  - {i}" for i in check["issues"]))


def cmd_run(args) -> None:
    """执行评估。"""
    logger.info("正在执行评估……")

    from agentkb.storage.pg_database import get_db
    from agentkb.knowledge.embedder import get_embedder
    from agentkb.eval.testset import TestSet
    from agentkb.eval.evaluator import Evaluator

    cfg = Settings.load()
    _ensure_dirs()

    # 加载测试集
    testset_path = args.testset or cfg.eval_testset_path
    testset = TestSet.load(testset_path)

    # 预热组件
    get_db()
    get_embedder()

    evaluator = Evaluator(skip_reranker=args.skip_reranker)
    result = evaluator.evaluate(testset)

    # 保存结果
    output = Path(args.output or "data/eval/latest_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json_report(result, output)

    # 打印摘要
    print()
    print("=" * 60)
    print("  评估摘要")
    print("=" * 60)
    for k in result.k_values:
        r = result.recall_at_k.get(k, 0)
        status = "✅" if r >= 0.9 else ("⚠️" if r >= 0.6 else "❌")
        print(f"  Recall@{k:<3}:  {r:.4f}  {status}  (正确答案出现在 top-{k} 中的比例)")
    print(f"  MRR:       {result.mrr:.4f}           (第一个正确答案的平均排名倒数)")
    for k in result.k_values:
        n = result.ndcg_at_k.get(k, 0)
        print(f"  NDCG@{k:<3}:  {n:.4f}           (归一化折损累计增益)")
    print("=" * 60)
    print()

    # 生成 Markdown 报告
    report_path = output.with_suffix(".md")
    md = render_result_markdown(result, title="AgentKB 检索评估报告")
    report_path.write_text(md, encoding="utf-8")
    logger.info(f"Markdown 报告: {report_path}")

    # diff-baseline 对比模式：Recall@5 退化超过 2% 则 exit 1
    if args.diff_baseline:
        import json
        from agentkb.eval.metrics import EvalResult
        baseline_path = Path(args.diff_baseline)
        if not baseline_path.exists():
            logger.error(f"基线文件不存在: {baseline_path}")
            return
        baseline_data = json.loads(baseline_path.read_text(encoding="utf-8"))
        bm = baseline_data.get("metrics", {})
        baseline = EvalResult(
            k_values=[int(k) for k in bm["recall_at_k"].keys()],
            recall_at_k={int(k): v for k, v in bm["recall_at_k"].items()},
            precision_at_k={int(k): v for k, v in bm.get("precision_at_k", {}).items()},
            mrr=bm.get("mrr", 0),
            ndcg_at_k={int(k): v for k, v in bm.get("ndcg_at_k", {}).items()},
        )

        from agentkb.eval.evaluator import Evaluator
        diff = Evaluator.diff(baseline, result, args.diff_baseline, "current")

        recall5_diff = next((d for d in diff.diffs if d.name == "Recall@5"), None)
        if recall5_diff and recall5_diff.delta < -0.02:
            logger.error(
                f"Recall@5 退化 {abs(recall5_diff.delta):.3f} 超过阈值 0.02，CI 失败。"
                f" 基线: {recall5_diff.baseline:.4f} → 当前: {recall5_diff.current:.4f}"
            )
            import sys
            sys.exit(1)

        md = render_diff_markdown(diff)
        diff_path = output.with_suffix(".diff.md")
        diff_path.write_text(md, encoding="utf-8")
        logger.info(f"对比报告: {diff_path}")
        logger.info("Recall@5 退化在阈值内，通过 ✅")


def cmd_compare(args) -> None:
    """对比两次评估结果。"""
    import json
    from agentkb.eval.metrics import EvalResult

    baseline_data = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    current_data = json.loads(Path(args.current).read_text(encoding="utf-8"))

    # 从 JSON 重建 EvalResult（仅聚合指标）
    baseline_metrics = baseline_data.get("metrics", {})
    current_metrics = current_data.get("metrics", {})

    baseline = EvalResult(
        k_values=[int(k) for k in baseline_metrics["recall_at_k"].keys()],
        recall_at_k={int(k): v for k, v in baseline_metrics["recall_at_k"].items()},
        precision_at_k={int(k): v for k, v in baseline_metrics.get("precision_at_k", {}).items()},
        mrr=baseline_metrics.get("mrr", 0),
        ndcg_at_k={int(k): v for k, v in baseline_metrics.get("ndcg_at_k", {}).items()},
    )
    current = EvalResult(
        k_values=[int(k) for k in current_metrics["recall_at_k"].keys()],
        recall_at_k={int(k): v for k, v in current_metrics["recall_at_k"].items()},
        precision_at_k={int(k): v for k, v in current_metrics.get("precision_at_k", {}).items()},
        mrr=current_metrics.get("mrr", 0),
        ndcg_at_k={int(k): v for k, v in current_metrics.get("ndcg_at_k", {}).items()},
    )

    from agentkb.eval.evaluator import Evaluator
    diff = Evaluator.diff(baseline, current, args.baseline, args.current)

    # 打印对比
    print()
    print("=" * 70)
    print(f"  对比: {args.baseline} → {args.current}")
    print("=" * 70)
    print(f"  {'指标':<16} {'基线':>8} {'当前':>8} {'Δ':>8}  评估")
    print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8}  {'-'*10}")
    for d in diff.diffs:
        icon = "✅" if d.delta > 0.001 else ("❌" if d.delta < -0.001 else "➡️")
        print(f"  {d.name:<16} {d.baseline:>8.4f} {d.current:>8.4f} {d.delta:>+8.4f}  {icon}")
    print("=" * 70)
    print()

    # 保存
    output = Path(args.output or "data/eval/latest_diff.md")
    output.parent.mkdir(parents=True, exist_ok=True)
    md = render_diff_markdown(diff)
    output.write_text(md, encoding="utf-8")
    logger.info(f"对比报告: {output}")


def cmd_report(args) -> None:
    """从 JSON 结果生成报告。"""
    import json
    from agentkb.eval.metrics import EvalResult

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    metrics = data.get("metrics", {})

    result = EvalResult(
        k_values=[int(k) for k in metrics.get("recall_at_k", {}).keys()],
        recall_at_k={int(k): v for k, v in metrics.get("recall_at_k", {}).items()},
        precision_at_k={int(k): v for k, v in metrics.get("precision_at_k", {}).items()},
        mrr=metrics.get("mrr", 0),
        ndcg_at_k={int(k): v for k, v in metrics.get("ndcg_at_k", {}).items()},
    )

    fmt = args.format or "md"
    if fmt == "md":
        md = render_result_markdown(result, title="AgentKB 检索评估报告")
        output = Path(args.output or "data/eval/latest_report.md")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(md, encoding="utf-8")
        logger.info(f"报告已生成: {output}")
        print(md)
    elif fmt == "json":
        output = Path(args.output or "data/eval/latest_report.json")
        save_json_report(result, output)
        logger.info(f"JSON 报告已生成: {output}")
    else:
        logger.error(f"不支持的格式: {fmt}（支持 md / json）")


async def cmd_generation(args) -> None:
    """执行生成质量评估。"""
    logger.info("正在执行生成质量评估……")

    from agentkb.storage.pg_database import get_db
    from agentkb.knowledge.embedder import get_embedder
    from agentkb.llm.factory import get_chat_model
    from agentkb.eval.testset import TestSet
    from agentkb.eval.evaluator import Evaluator

    cfg = Settings.load()
    _ensure_dirs()

    testset_path = args.testset or cfg.eval_testset_path
    testset = TestSet.load(testset_path)
    testset.items = testset.items[:args.sample_size]

    get_db()
    get_embedder()
    llm = get_chat_model(streaming=False)

    evaluator = Evaluator()
    result = await evaluator.evaluate_full(testset, llm_client=llm)

    # 保存
    import json
    output = Path(args.output or "data/eval/latest_generation_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("  生成质量评估摘要")
    print("=" * 60)
    gen = result["generation"]
    print(f"  平均 Faithfulness:     {gen['avg_faithfulness']:.3f}  (答案是否来自上下文)")
    print(f"  平均 Answer Relevance:  {gen['avg_answer_relevance']:.3f}  (答案是否回答正确)")
    print(f"  平均 Context Relevance: {gen['avg_context_relevance']:.3f}  (上下文是否相关)")
    print(f"  评估样本数:             {gen['total_evaluated']}")
    print("=" * 60)

    # 标记低 quality case
    low_quality = [
        (i, m) for i, m in enumerate(gen.get("per_sample", []))
        if m["faithfulness"] < 0.4
    ]
    if low_quality:
        print()
        print(f"  低忠实度样本（Faithfulness < 0.4）: {len(low_quality)} 条")
        for idx, metric in low_quality:
            q = testset.items[idx].query if idx < len(testset.items) else f"样本 #{idx}"
            print(f"    - [{q[:60]}] F={metric['faithfulness']:.2f} AR={metric['answer_relevance']:.2f}")
    print()
    logger.info(f"结果已保存: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentKB 检索评估框架",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", help="子命令")

    # generate
    p_gen = sub.add_parser("generate", help="生成银标测试集（从知识库采样 → LLM 生成问题）")
    p_gen.add_argument("--sample-size", type=int, default=None, help="采样 chunk 数量（默认 50）")
    p_gen.add_argument("--seed", type=int, default=42, help="随机种子")
    p_gen.add_argument("--output", type=str, default=None, help="输出路径")
    p_gen.set_defaults(func=cmd_generate)

    # run
    p_run = sub.add_parser("run", help="执行评估（含指标注释的报告）")
    p_run.add_argument("--testset", type=str, default=None, help="测试集路径")
    p_run.add_argument("--skip-reranker", action="store_true", help="跳过 reranker（用于对比）")
    p_run.add_argument("--output", type=str, default=None, help="输出路径")
    p_run.add_argument("--diff-baseline", type=str, default=None, help="基线 JSON，对比后 Recall@5 退化超过 2%% 则 exit 1")
    p_run.set_defaults(func=cmd_run)

    # compare
    p_cmp = sub.add_parser("compare", help="对比两次评估结果")
    p_cmp.add_argument("--baseline", type=str, required=True, help="基线评估 JSON")
    p_cmp.add_argument("--current", type=str, required=True, help="当前评估 JSON")
    p_cmp.add_argument("--output", type=str, default=None, help="输出路径")
    p_cmp.set_defaults(func=cmd_compare)

    # report
    p_rep = sub.add_parser("report", help="从 JSON 结果生成报告")
    p_rep.add_argument("--input", type=str, required=True, help="评估结果 JSON")
    p_rep.add_argument("--format", type=str, default="md", choices=["md", "json"])
    p_rep.add_argument("--output", type=str, default=None)
    p_rep.set_defaults(func=cmd_report)

    # generation
    p_gen_eval = sub.add_parser("generation", help="跑生成质量评估（Faithfulness / Answer Relevance）")
    p_gen_eval.add_argument("--testset", type=str, default=None, help="测试集路径")
    p_gen_eval.add_argument("--sample-size", type=int, default=10, help="评估样本数（默认 10）")
    p_gen_eval.add_argument("--output", type=str, default=None)
    p_gen_eval.set_defaults(func=cmd_generation)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
