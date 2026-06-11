"""测试集管理——加载/保存/自动生成银标测试集。

测试集分层:
  - 黄金集 (gold): 人工标注，绝对权威，用于发布前最终验证
  - 银标集 (silver): LLM 自动生成 + 可选人工抽检，用于日常迭代
  - 青铜集: 每次评估从知识库随机采样，全自动快速回归

数据格式 (JSON):
{
  "version": 1,
  "created_at": "iso_string",
  "items": [
    {
      "query": "公司年假有多少天？",
      "relevant_chunk_ids": ["uuid1", "uuid2"],
      "source_file": "policy.md",
      "generated_by": "llm"  // "llm" | "human"
    }
  ]
}
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger


@dataclass
class TestItem:
    """单条测试用例。"""
    query: str
    relevant_chunk_ids: list[str]
    source_file: str = ""
    generated_by: str = "human"  # "llm" | "human"

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "source_file": self.source_file,
            "generated_by": self.generated_by,
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestItem:
        return cls(
            query=d["query"],
            relevant_chunk_ids=d["relevant_chunk_ids"],
            source_file=d.get("source_file", ""),
            generated_by=d.get("generated_by", "human"),
        )


@dataclass
class TestSet:
    """测试集容器。"""
    items: list[TestItem] = field(default_factory=list)
    created_at: str = ""
    version: int = 1

    # ════════════════════════════════════════════════════════════
    #  序列化
    # ════════════════════════════════════════════════════════════

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "created_at": self.created_at or datetime.now(timezone.utc).isoformat(),
            "items": [item.to_dict() for item in self.items],
        }

    @classmethod
    def from_dict(cls, d: dict) -> TestSet:
        items = [TestItem.from_dict(i) for i in d.get("items", [])]
        return cls(
            items=items,
            created_at=d.get("created_at", ""),
            version=d.get("version", 1),
        )

    def save(self, path: str | Path) -> None:
        """保存测试集到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_dict()
        data["created_at"] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"测试集已保存: {path}（{len(self.items)} 条）")

    @classmethod
    def load(cls, path: str | Path) -> TestSet:
        """从 JSON 文件加载测试集。"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"测试集文件不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = cls.from_dict(data)
        logger.info(f"测试集已加载: {path}（{len(ts.items)} 条）")
        return ts

    @classmethod
    def from_queries(cls, queries: list[dict]) -> TestSet:
        """从原始 query dict 列表构建 TestSet（用于 HTTP API 直接提交 queries）。"""
        items = []
        for q in queries:
            items.append(TestItem(
                query=q.get("query", ""),
                relevant_chunk_ids=q.get("relevant_chunk_ids", []),
                source_file=q.get("source_file", ""),
                generated_by=q.get("generated_by", "human"),
            ))
        return cls(items=items, created_at=datetime.now(timezone.utc).isoformat())

    def validate(self) -> dict:
        """检查测试集数据完整性。

        返回 {"valid": bool, "issues": [str]}。
        """
        issues = []
        seen_queries = set()
        for i, item in enumerate(self.items):
            if not item.query or not item.query.strip():
                issues.append(f"第 {i} 条: query 为空")
            if item.query.strip() in seen_queries:
                issues.append(f"第 {i} 条: 重复 query: {item.query}")
            seen_queries.add(item.query.strip())
            if not item.relevant_chunk_ids:
                issues.append(f"第 {i} 条: relevant_chunk_ids 为空: {item.query}")
        return {"valid": len(issues) == 0, "issues": issues}

    # ════════════════════════════════════════════════════════════
    #  自动生成银标集（文档级窗口采样）
    # ════════════════════════════════════════════════════════════

    @classmethod
    def generate(
        cls,
        db,
        sample_size: int = 50,
        questions_per_chunk: int = 2,
        seed: int = 42,
    ) -> TestSet:
        """从知识库自动生成银标测试集（文档级窗口采样）。

        流程:
          1. 按文件分组取所有 chunk，按 chunk_index 排序
          2. 每文件内用滑动窗口合并相邻 chunk 组（默认窗口=3 chunk）
             → 窗口内容作为"完整上下文"让 LLM 生成问题
             → 窗口内所有 chunk 都标注为"相关"
             → 更接近真实检索场景：一个问题的答案往往跨多个相邻 chunk
          3. 分层采样：每个文件至少贡献 min_windows 个窗口
             剩余配额按文件 chunk 数量比例分配
          4. LLM 反向验证，过滤不合格问题

        Args:
            db:                   Database 实例
            embedder:             EmbedderService 实例
            sample_size:          目标窗口数量（每个窗口生成 1 组问题）
            questions_per_chunk:  每个窗口生成的问题数
            seed:                 随机种子

        Returns:
            TestSet 银标测试集
        """
        random.seed(seed)
        llm = cls._create_llm_client()

        # 1. 文档级窗口采样
        windows = cls._sample_chunk_groups(db, sample_size)
        if not windows:
            logger.warning("知识库中没有足够的 chunk，无法生成测试集")
            return cls()

        total_chunks = sum(len(w["chunk_ids"]) for w in windows)
        logger.info(
            f"已采样 {len(windows)} 个窗口（覆盖 {total_chunks} 个 chunk），"
            f"来自 {len(set(w['source'] for w in windows))} 个文件，开始生成问题……"
        )

        items: list[TestItem] = []
        for idx, win in enumerate(windows):
            content = win["content"]
            chunk_ids = win["chunk_ids"]
            source = win["source"]

            # 2. 用窗口完整上下文生成问题
            questions = cls._generate_questions(llm, content, questions_per_chunk)

            # 3. 反向验证——用合并文本判断
            valid_questions = []
            for q in questions:
                if cls._validate_question(llm, q, content):
                    valid_questions.append(q)

            # 4. 窗口内所有 chunk 都标注为相关
            for q in valid_questions:
                items.append(TestItem(
                    query=q,
                    relevant_chunk_ids=list(chunk_ids),  # 多个 chunk 标注
                    source_file=source,
                    generated_by="llm",
                ))

            if (idx + 1) % 10 == 0:
                logger.info(f"  进度: {idx + 1}/{len(windows)}，已生成 {len(items)} 条")

        ts = cls(items=items)
        logger.info(
            f"银标测试集生成完成: {len(items)} 条 "
            f"（{len(items) / max(len(windows), 1):.1f} 条/窗口）"
        )
        return ts

    # ── 内部方法 ──────────────────────────────────────────────

    @staticmethod
    def _sample_chunk_groups(
        db,
        sample_size: int,
        window_size: int = 2,
        min_windows_per_file: int = 2,
    ) -> list[dict]:
        """按文件分层滑动窗口采样。

        1. 从 DB 获取所有 chunk，按 (file_id, chunk_index) 排序
        2. 每个文件内用滑动窗口（大小 window_size）截取相邻 chunk 组
        3. 分层配额：每个文件至少 min_windows_per_file 个窗口
           剩余按文件 chunk 数比例分配
        4. 随机选取窗口，避免同一文件只取到开头部分

        返回:
          [{"content": "合并文本", "chunk_ids": [id1, id2, id3],
            "source": "文件.md"}, ...]
        """
        try:
            with db._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT kc.id, kc.content, kc.chunk_index, kc.file_id,
                                  kc.chunk_metadata, kf.filename
                           FROM knowledge_chunks kc
                           JOIN knowledge_files kf ON kc.file_id = kf.id
                           WHERE kf.status = 'active'
                           ORDER BY kc.file_id, kc.chunk_index"""
                    )
                    all_rows = [dict(r) for r in cur.fetchall()]
        except Exception as e:
            logger.error(f"chunk 分组采样失败: {e}")
            return []

        if not all_rows:
            return []

        # 按 file_id 分组
        file_groups: dict[str, list[dict]] = {}
        for row in all_rows:
            fid = row["file_id"]
            file_groups.setdefault(fid, []).append(row)

        # 每个文件生成滑动窗口
        file_windows: dict[str, list[dict]] = {}
        for fid, chunks in file_groups.items():
            filename = chunks[0].get("filename", "unknown")
            windows_for_file = []
            for start in range(0, len(chunks) - window_size + 1, max(1, window_size // 2)):
                window_chunks = chunks[start:start + window_size]
                merged_content = "\n\n".join(c["content"] for c in window_chunks)
                windows_for_file.append({
                    "content": merged_content,
                    "chunk_ids": [c["id"] for c in window_chunks],
                    "source": filename,
                })
            if windows_for_file:
                file_windows[fid] = windows_for_file

        if not file_windows:
            return []

        # 分层配额分配
        total_files = len(file_windows)
        fixed_quota = total_files * min_windows_per_file
        remaining = max(0, sample_size - fixed_quota)

        # 计算每个文件的 chunk 占比
        total_all_chunks = sum(len(file_groups[fid]) for fid in file_windows)
        selected = []

        for fid, windows in file_windows.items():
            # 固定配额
            n_fixed = min(min_windows_per_file, len(windows))
            sampled = random.sample(windows, n_fixed)
            selected.extend(sampled)

            # 比例配额
            if remaining > 0:
                file_chunk_count = len(file_groups[fid])
                proportion = file_chunk_count / total_all_chunks
                n_extra = min(
                    int(remaining * proportion),
                    len(windows) - n_fixed,  # 不能超过该文件可用窗口
                )
                if n_extra > 0:
                    remaining_samples = [w for w in windows if w not in sampled]
                    extra = random.sample(remaining_samples, n_extra)
                    selected.extend(extra)

        # 如果配额没满，从剩余窗口中补充
        if len(selected) < sample_size:
            all_windows = [w for windows in file_windows.values() for w in windows]
            remaining_pool = [w for w in all_windows if w not in selected]
            if remaining_pool:
                extra_needed = sample_size - len(selected)
                extra = random.sample(remaining_pool, min(extra_needed, len(remaining_pool)))
                selected.extend(extra)

        random.shuffle(selected)
        logger.info(
            f"分层采样: {total_files} 个文件, {len(selected)} 个窗口 "
            f"(配额: 固定={fixed_quota}, 比例={remaining})"
        )
        return selected[:sample_size]

    # ── LLM 客户端 ─────────────────────────────────────────────

    @staticmethod
    def _create_llm_client() -> dict:
        """返回 LLM 调用配置。"""
        from agentkb.config.settings import Settings
        cfg = Settings.load()
        return {
            "provider": cfg.llm_provider,
            "protocol": cfg.llm_protocol,
            "base_url": cfg.llm_base_url.rstrip("/"),
            "model": cfg.llm_generator_model_name,
            "timeout": cfg.llm_request_timeout,
            "api_key": cfg.llm_api_key,
        }

    @staticmethod
    def _llm_generate(client: dict, prompt: str) -> str:
        """按配置协议调用 LLM API 生成文本。"""
        import httpx

        protocol = client["protocol"]

        if protocol == "openai":
            try:
                resp = httpx.post(
                    f"{client['base_url']}/chat/completions",
                    json={
                        "model": client["model"],
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "temperature": 0.1,
                    },
                    headers={
                        "Authorization": f"Bearer {client.get('api_key', '')}",
                        "Content-Type": "application/json",
                    },
                    timeout=client["timeout"],
                )
                if resp.status_code == 200:
                    body = resp.json()
                    return body["choices"][0]["message"]["content"]
                logger.warning(
                    f"LLM Provider '{client['provider']}' 返回 {resp.status_code}"
                )
                return ""
            except Exception as e:
                logger.warning(
                    f"LLM Provider '{client['provider']}' 调用失败: {e}"
                )
                return ""

        if protocol != "ollama":
            logger.warning(f"不支持的 LLM 协议: {protocol}")
            return ""

        try:
            resp = httpx.post(
                f"{client['base_url']}/api/generate",
                json={"model": client["model"], "prompt": prompt, "stream": False},
                timeout=client["timeout"],
            )
            if resp.status_code == 200:
                return resp.json().get("response", "")
            logger.warning(f"Ollama API 返回 {resp.status_code}: {resp.text[:200]}")
            return ""
        except Exception as e:
            logger.warning(f"LLM 调用失败: {e}")
            return ""

    @staticmethod
    def _generate_questions(llm_client: dict, content: str, n: int) -> list[str]:
        """用 LLM 根据 chunk 内容生成问题。"""
        prompt = f"""你是一个测试数据标注员。根据以下文本内容，生成 {n} 个可以从这段文本中找到答案的中文问题。

要求:
1. 问题必须是自然的中文提问方式
2. 答案确实存在于给定的文本中
3. 问题多样化，不要都是一种问法
4. 每行一个问题，以数字序号开头（如 "1. "）
5. 只输出问题列表，不要其他内容

文本内容:
{content[:3000]}"""

        text = TestSet._llm_generate(llm_client, prompt)
        if not text:
            return []
        return TestSet._parse_questions(text, n)

    @staticmethod
    def _validate_question(llm_client: dict, question: str, content: str) -> bool:
        """验证问题是否能从 chunk 内容中回答。"""
        prompt = f"""判断以下问题是否能用给定的文本回答。

请严格只回答一个字：是 或 否。

问题: {question}

文本: {content[:2000]}"""

        text = TestSet._llm_generate(llm_client, prompt)
        return "是" in text[:10] if text else False

    @staticmethod
    def _parse_questions(text: str, max_n: int) -> list[str]:
        """从 LLM 输出中解析问题列表。"""
        questions = []
        for line in text.strip().split("\n"):
            line = line.strip()
            # 匹配 "1. xxx" "1、xxx" "1) xxx" 等格式
            import re
            match = re.match(r"^\d+[\.\)、]\s*(.*)", line)
            if match:
                q = match.group(1).strip()
                if q and len(q) > 3:
                    questions.append(q)
            elif line and len(line) > 5 and len(questions) < max_n:
                # 无序号的行也接受
                questions.append(line)
        return questions[:max_n]
