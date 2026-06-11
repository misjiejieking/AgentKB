"""工具：search_knowledge_base — 混合检索 + 重排序 + 上下文截断 + 查询重写。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import re
import time
from typing import Any, Generator

from pydantic import BaseModel, Field
from loguru import logger

from agentkb.config.settings import Settings
from agentkb.tools.base import BaseTool, ToolResult
from agentkb.knowledge.retriever import get_retriever
from agentkb.knowledge.cache import get_cache

_chat_history: ContextVar[tuple[str, ...]] = ContextVar(
    "knowledge_search_chat_history",
    default=(),
)


@contextmanager
def chat_history_context(messages: list) -> Generator[None, None, None]:
    """为当前工具调用绑定会话历史，并发任务之间互不共享。"""
    history = tuple(
        (m.content if hasattr(m, "content") else str(m))[:256]
        for m in messages[-6:]
    )
    token = _chat_history.set(history)
    try:
        yield
    finally:
        _chat_history.reset(token)


def get_chat_history() -> list[str]:
    """返回当前工具调用可见的会话历史。"""
    return list(_chat_history.get())


class KnowledgeSearchInput(BaseModel):
    """知识库检索的输入参数。"""
    query: str = Field(description="搜索查询词，尽量使用关键词")


class KnowledgeSearchTool(BaseTool):
    """混合检索（dense + BM25 → RRF 融合 → Reranker 精排）本地知识库。"""

    @property
    def name(self) -> str:
        return "search_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "检索本地知识库中的文档内容。"
            "适合回答需要从用户上传的文件中查找信息的问题，"
            "例如公司制度、学习笔记、文档内容等。"
            "输入应为简练的搜索关键词或问题。"
            "不要用这个工具处理实时信息、天气、新闻、股价等非本地文档类问题。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return KnowledgeSearchInput

    async def _execute(self, query: str = "", **kwargs: Any) -> ToolResult:
        from agentkb.agent.query_rewriter import rewrite_query
        from agentkb.llm.factory import get_chat_model

        cfg = Settings.load()
        retriever = get_retriever()

        # 0. 查询重写——多轮对话场景指代消解 + 关键词提取
        search_query = query
        try:
            llm = get_chat_model(streaming=True)
            rewritten = await rewrite_query(
                query,
                get_chat_history(),
                llm,
            )
            search_query = rewritten.get("rewritten", query)
            logger.debug(f"查询重写: {query[:60]} → {search_query[:60]}")
        except Exception as e:
            logger.debug(f"查询重写跳过: {e}")

        # 0.5. 语义缓存——命中则跳过检索+重排
        from agentkb.knowledge.embedder import get_embedder
        cache = get_cache()
        embedder = get_embedder()
        cache_embedding = embedder.embed_query(search_query)
        cached_results = cache.get(cache_embedding)
        if cached_results:
            from agentkb.observability.metrics import get_metrics

            get_metrics().record_cache(True)
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": cached_results,
                    "total": len(cached_results),
                    "cached": True,
                },
            )
        from agentkb.observability.metrics import get_metrics

        get_metrics().record_cache(False)

        # 1. 混合检索 → 候选集
        retrieval_started_at = time.perf_counter()
        try:
            candidates = retriever.retrieve(search_query)
        finally:
            get_metrics().record_retrieval(
                (time.perf_counter() - retrieval_started_at) * 1000,
            )

        if not candidates:
            return ToolResult(
                tool_name=self.name,
                success=True,
                data={
                    "query": query,
                    "results": [],
                    "total": 0,
                    "degraded": True,
                    "reason": "检索无结果",
                    "hint": "知识库中没有找到相关内容",
                },
            )

        # 2. 按 RRF 分数排序（reranker 暂不使用：百炼免费额度已用完，本地 CPU 推理太慢）
        candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
        ranked = candidates[:cfg.retrieval_final_k]

        # 3. 格式化输出：优先使用 parent_content 保证完整上下文
        results = []
        for r in ranked:
            content = r.get("parent_content") or r.get("content", "")
            chunk_meta = r.get("chunk_metadata")
            if isinstance(chunk_meta, str):
                import json
                try:
                    chunk_meta = json.loads(chunk_meta)
                except json.JSONDecodeError:
                    chunk_meta = {}

            filename = (chunk_meta or {}).get("filename", "") or r.get("file_id", "unknown")
            results.append({
                "content": content,
                "filename": filename,
                "score": r.get("rerank_score") or r.get("rrf_score", 0),
            })

        # 4. 上下文截断：限制总 token 数，优先保留高分结果
        # deep copy 避免截断后的残缺内容写入语义缓存
        import copy
        results_for_cache = copy.deepcopy(results)
        results = self._truncate_context(results, max_tokens=cfg.llm_max_tokens)

        # 写入语义缓存（用未截断的完整内容）
        cache.set(cache_embedding, results_for_cache[:cfg.retrieval_final_k])

        # 检查降级标记
        degraded = any(r.get("degraded") for r in ranked)
        reason = ""
        if degraded:
            reason = ranked[0].get("degrade_reason", "部分检索组件降级")

        return ToolResult(
            tool_name=self.name,
            success=True,
            data={
                "query": query,
                "results": results,
                "total": len(results),
                "degraded": degraded,
                "reason": reason,
            } if degraded else {
                "query": query,
                "results": results,
                "total": len(results),
            },
        )

    @staticmethod
    def _truncate_context(results: list[dict], max_tokens: int = 4096) -> list[dict]:
        """限制检索上下文总 token 数，高分结果保留更多内容。"""
        budget = int(max_tokens * 0.6)  # 上下文预算 = LLM 窗口的 60%
        if not results:
            return results

        # 按 score 降序确保优先保留高分结果
        sorted_results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        per_item_budget = budget // len(sorted_results)

        # 估算 token 数：中文约 1.5 字符/token
        def estimate_tokens(text: str) -> int:
            return max(1, len(text) // 2)

        truncated = []
        total_tokens = 0
        for item in sorted_results:
            content = item.get("content", "")
            content_tokens = estimate_tokens(content)

            if content_tokens <= per_item_budget:
                truncated.append(item)
                total_tokens += content_tokens
            else:
                # 截断到句子边界
                max_chars = per_item_budget * 2
                truncated_content = KnowledgeSearchTool._cut_at_sentence(content, max_chars)
                item["content"] = truncated_content + "…(内容已截断)"
                truncated.append(item)
                total_tokens += per_item_budget

            if total_tokens >= budget:
                break

        return truncated

    @staticmethod
    def _cut_at_sentence(text: str, max_chars: int) -> str:
        """在 max_chars 内的最后一个句号/换行处截断。"""
        if len(text) <= max_chars:
            return text
        truncated = text[:max_chars]
        # 找最后一个句子边界
        m = re.search(r'[。！？.!?\n](?=[^。！？.!?\n]*$)', truncated)
        if m:
            return truncated[:m.end()]
        return truncated
