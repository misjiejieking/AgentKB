"""KnowledgeAgent——升级后的知识管理与检索 Specialist Agent。

支持：语义检索、文件列表浏览、自动总结、关联推荐。
"""

from __future__ import annotations

import json
import time
from typing import Any

from agentkb.agents.base import SpecialistAgent, AgentResult


class KnowledgeAgent(SpecialistAgent):
    """知识管理与检索专家。"""

    @property
    def name(self) -> str:
        return "knowledge_agent"

    @property
    def description(self) -> str:
        return "管理本地知识库：检索文档内容、浏览文件列表、自动生成摘要、发现知识关联"

    @property
    def intents(self) -> list[str]:
        return ["knowledge_search", "knowledge_management", "hybrid", "web_search"]

    async def execute(
        self,
        task: str,
        context: dict[str, Any] | None = None,
        tools: list | None = None,
    ) -> AgentResult:
        t0 = time.time()
        context = context or {}
        tokens_used = 0

        try:
            from agentkb.storage.pg_database import get_db
            from agentkb.config.settings import Settings

            cfg = Settings.load()
            db = get_db()

            # ── 元问题：查看知识库概览/文件列表 ──
            if self._is_meta_question(task):
                return await self._list_files(db, task, t0)

            # ── 正常检索流程 ──
            from agentkb.knowledge.retriever import get_retriever
            retriever = get_retriever()
            candidates = retriever.retrieve(task)
            graph = (
                db.search_knowledge_graph(task)
                if cfg.knowledge_graph_enabled and self._is_graph_question(task)
                else {"nodes": [], "edges": []}
            )

            if not candidates and not graph["edges"]:
                # 无结果时列出可用文件，帮助用户了解知识库范围
                files = db.list_knowledge_files()
                if files:
                    file_list = "\n".join(
                        f"- {f['filename']}（{f.get('chunk_count', 0)} 个片段, "
                        f"{f.get('file_type', 'unknown')}）"
                        for f in files
                    )
                    return AgentResult(
                        agent_name=self.name,
                        success=True,
                        output=(
                            f"知识库中未找到与「{task}」直接相关的内容。\n\n"
                            f"当前知识库共有 {len(files)} 个文件：\n{file_list}\n\n"
                            "您可以尝试：\n1. 换一种提问方式\n2. 上传更多相关文件\n3. 使用联网搜索"
                        ),
                        data={"candidates_count": 0, "files_in_kb": len(files)},
                        elapsed_ms=(time.time() - t0) * 1000,
                    )

                return AgentResult(
                    agent_name=self.name,
                    success=True,
                    output=(
                        f"知识库中暂未找到与「{task}」相关的内容。\n"
                        "您可以尝试上传相关文件到知识库，或使用联网搜索获取信息。"
                    ),
                    data={"candidates_count": 0},
                    elapsed_ms=(time.time() - t0) * 1000,
                )

            # 上下文组装
            contexts = []
            sources = set()
            for edge in graph["edges"]:
                contexts.append(
                    f"{edge['source']} --{edge['predicate']}--> {edge['target']}\n"
                    f"证据：{edge['evidence']}"
                )
                sources.add(edge["filename"])
            for c in candidates[:cfg.retrieval_final_k]:
                content = c.get("parent_content") or c.get("content", "")
                meta = c.get("chunk_metadata", {})
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except json.JSONDecodeError:
                        meta = {}
                filename = meta.get("filename", "") or c.get("file_id", "unknown")
                sources.add(filename)
                contexts.append(content)

            # LLM 生成回答
            llm = self.llm
            ctx_text = "\n\n---\n\n".join(contexts[:5])
            prompt = (
                f"基于以下知识库内容回答用户问题。\n\n"
                f"## 知识库内容\n{ctx_text[:3000]}\n\n"
                f"## 用户问题\n{task}\n\n"
                f"## 回答要求\n"
                f"- 基于知识库内容回答，不要编造\n"
                f"- 如果知识库内容不足以回答，诚实说明\n"
                f"- 引用具体文件名标注来源：[来源: xxx.md]\n"
                f"- 如果涉及多个文件的内容，综合回答\n\n"
                f"## 回答"
            )
            response = await llm.ainvoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            tokens_used = len(prompt) // 2 + len(answer) // 2

            source_list = "\n\n**参考文件：**\n" + "\n".join(f"- {s}" for s in sorted(sources))
            output = answer + source_list

            return AgentResult(
                agent_name=self.name,
                success=True,
                output=output,
                data={
                    "candidates_count": len(candidates),
                    "graph_relations": len(graph["edges"]),
                    "sources": list(sources),
                    "top_score": candidates[0].get("rrf_score", 0) if candidates else 0,
                },
                tokens_used=tokens_used,
                elapsed_ms=(time.time() - t0) * 1000,
            )

        except Exception as e:
            return AgentResult(
                agent_name=self.name,
                success=False,
                error=str(e),
                output="知识检索过程中遇到了问题，请稍后重试。",
                elapsed_ms=(time.time() - t0) * 1000,
            )

    # ── 内部方法 ──────────────────────────────────────────────────

    @staticmethod
    def _is_meta_question(task: str) -> bool:
        """检测是否为关于知识库本身的元问题（非内容检索）。"""
        q = task.strip().lower()
        meta_patterns = [
            "有哪些文件", "文件列表", "什么文件", "哪些文件",
            "我的文件", "上传的文件", "知识库有哪些", "知识库里有什么",
            "知识库内容", "files", "what files",
            "有多少文件", "几个文件", "文件数量",
            "知识库列表", "kb list",
        ]
        return any(p in q for p in meta_patterns)

    @staticmethod
    def _is_graph_question(task: str) -> bool:
        """识别需要实体关系推理的问题。"""
        patterns = [
            "关系", "关联", "依赖", "属于", "负责", "组成",
            "连接", "调用", "上下游", "包含", "隶属",
        ]
        return any(pattern in task for pattern in patterns)

    async def _list_files(self, db, task: str, t0: float) -> AgentResult:
        """列出知识库中的所有文件及其统计信息。"""
        files = db.list_knowledge_files()

        if not files:
            return AgentResult(
                agent_name=self.name,
                success=True,
                output=(
                    "知识库目前是空的，还没有上传任何文件。\n\n"
                    "您可以通过以下方式添加知识：\n"
                    "1. 点击顶栏的 📤 上传按钮\n"
                    "2. 拖拽文件到页面\n"
                    "支持格式：.md / .txt / .pdf / .docx / .csv / .json"
                ),
                data={"files_in_kb": 0},
                elapsed_ms=(time.time() - t0) * 1000,
            )

        # 统计
        total_chunks = sum(f.get("chunk_count", 0) for f in files)
        total_size = sum(f.get("file_size", 0) for f in files)
        type_counts: dict[str, int] = {}
        for f in files:
            ft = f.get("file_type", "unknown")
            type_counts[ft] = type_counts.get(ft, 0) + 1

        lines = [
            f"当前知识库共有 **{len(files)}** 个文件，**{total_chunks}** 个内容片段。",
            "",
            "| 文件名 | 类型 | 片段数 | 大小 |",
            "|--------|------|--------|------|",
        ]
        for f in files:
            fname = f.get("filename", "unknown")[:40]
            ftype = f.get("file_type", "?")
            chunks = f.get("chunk_count", 0)
            size_kb = (f.get("file_size", 0) or 0) / 1024
            size_str = f"{size_kb:.1f}KB" if size_kb < 1024 else f"{size_kb/1024:.1f}MB"
            lines.append(f"| {fname} | {ftype} | {chunks} | {size_str} |")

        lines.append("")
        type_summary = "、".join(f"{v} 个 {k}" for k, v in sorted(type_counts.items()))
        lines.append(f"文件类型分布：{type_summary}")

        # 用 LLM 生成自然语言描述
        try:
            llm = self.llm
            summary_prompt = (
                f"根据以下知识库信息，用2-3句话简要描述这个知识库包含的内容领域：\n"
                f"文件列表：{', '.join(f.get('filename', '') for f in files)}\n"
                f"文件类型：{type_summary}\n"
            )
            resp = await llm.ainvoke(summary_prompt)
            summary = resp.content if hasattr(resp, "content") else str(resp)
            lines.insert(1, f"📊 {summary.strip()}")
            lines.insert(2, "")
        except Exception:
            pass

        output = "\n".join(lines)

        return AgentResult(
            agent_name=self.name,
            success=True,
            output=output,
            data={
                "files_in_kb": len(files),
                "total_chunks": total_chunks,
                "total_size": total_size,
                "type_distribution": type_counts,
            },
            elapsed_ms=(time.time() - t0) * 1000,
        )
