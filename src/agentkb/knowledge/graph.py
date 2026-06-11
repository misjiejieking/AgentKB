"""知识图谱抽取与持久化调度。"""

from __future__ import annotations

import asyncio
import re
import unicodedata
from contextvars import Context
from typing import Any

from loguru import logger
from pydantic import BaseModel, Field

from agentkb.config.settings import Settings
from agentkb.storage.pg_database import Database, get_db


class ExtractedEntity(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)


class ExtractedRelation(BaseModel):
    source: str = Field(min_length=1, max_length=200)
    target: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=120)
    evidence: str = Field(default="", max_length=500)
    confidence: float = Field(default=1, ge=0, le=1)


class KnowledgeGraphExtraction(BaseModel):
    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=30)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=40)


EXTRACTION_PROMPT = """从下面的知识片段中抽取可复用的实体和明确关系。

要求：
- 实体只保留有业务意义的专有名词、人物、组织、系统、项目、产品、地点、规则或关键概念
- entity_type 使用简短稳定的类别，例如 person、organization、system、project、concept、policy
- 关系 predicate 使用简短明确的动词或关系词
- 每条关系的 source 和 target 必须同时出现在 entities 中，名称必须完全一致
- evidence 必须来自原文，不得推断原文没有表达的关系
- 不要抽取“本文”“用户”“问题”等泛化实体
- 没有明确实体或关系时返回空列表

知识片段：
{content}
"""


def normalize_graph_term(value: str) -> str:
    """生成跨模型输出稳定的实体与关系去重键。"""
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    normalized = re.sub(r"\s+", "", normalized)
    return re.sub(r"[，。；：、,.!?！？;:()\[\]（）【】]", "", normalized)


class KnowledgeGraphIndexer:
    """将文件分块结构化抽取为 PostgreSQL 知识图谱。"""

    def __init__(
        self,
        db: Database | None = None,
        model: Any | None = None,
    ) -> None:
        self._db = db or get_db()
        self._model = model

    async def index_file(self, file_id: str) -> dict[str, int] | None:
        """抢占并执行单文件索引；未处于 queued 时返回 None。"""
        claimed = await asyncio.to_thread(
            self._db.claim_knowledge_graph_index,
            file_id,
        )
        if not claimed:
            return None

        cfg = Settings.load()
        try:
            chunks = await asyncio.to_thread(
                self._db.get_chunks_by_file_id,
                file_id,
                cfg.knowledge_graph_max_chunks_per_file,
            )
            model = self._model
            if model is None:
                from agentkb.llm.factory import get_chat_model

                model = get_chat_model(streaming=False)
            extractor = model.with_structured_output(KnowledgeGraphExtraction)

            records = []
            for chunk in chunks:
                content = str(chunk.get("content", "")).strip()
                if len(content) < cfg.knowledge_graph_min_chunk_chars:
                    continue
                extraction = await asyncio.wait_for(
                    extractor.ainvoke(
                        EXTRACTION_PROMPT.format(content=content[:6000])
                    ),
                    timeout=cfg.llm_request_timeout,
                )
                record = self._build_record(chunk, extraction)
                if record["entities"]:
                    records.append(record)

            stats = await asyncio.to_thread(
                self._db.replace_knowledge_graph,
                file_id,
                records,
            )
            await asyncio.to_thread(
                self._db.update_knowledge_graph_status,
                file_id,
                "ready",
            )
            logger.info(
                "知识图谱索引完成: file_id={}, entities={}, relations={}",
                file_id,
                stats["entities"],
                stats["relations"],
            )
            return stats
        except Exception as exc:
            await asyncio.to_thread(
                self._db.update_knowledge_graph_status,
                file_id,
                "failed",
                str(exc),
            )
            raise

    @staticmethod
    def _build_record(
        chunk: dict[str, Any],
        extraction: KnowledgeGraphExtraction,
    ) -> dict[str, Any]:
        entities = []
        known_names = set()
        for entity in extraction.entities:
            normalized_name = normalize_graph_term(entity.name)
            if not normalized_name:
                continue
            entities.append({
                "name": entity.name.strip(),
                "normalized_name": normalized_name,
                "entity_type": normalize_graph_term(entity.entity_type) or "concept",
                "description": entity.description.strip(),
            })
            known_names.add(normalized_name)

        relations = []
        for relation in extraction.relations:
            normalized_source = normalize_graph_term(relation.source)
            normalized_target = normalize_graph_term(relation.target)
            normalized_predicate = normalize_graph_term(relation.predicate)
            if (
                normalized_source not in known_names
                or normalized_target not in known_names
                or not normalized_predicate
            ):
                continue
            relations.append({
                "source": relation.source.strip(),
                "target": relation.target.strip(),
                "normalized_source": normalized_source,
                "normalized_target": normalized_target,
                "predicate": relation.predicate.strip(),
                "normalized_predicate": normalized_predicate,
                "evidence": relation.evidence.strip(),
                "confidence": relation.confidence,
            })

        return {
            "chunk_id": str(chunk["id"]),
            "context": str(chunk.get("content", ""))[:1000],
            "entities": entities,
            "relations": relations,
        }


_graph_tasks: dict[str, asyncio.Task[None]] = {}


def schedule_knowledge_graph_index(file_id: str) -> bool:
    """为文件创建独立后台索引任务，避免绑定上传请求上下文。"""
    if file_id in _graph_tasks:
        return False

    async def run() -> None:
        try:
            await KnowledgeGraphIndexer().index_file(file_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error(
                "知识图谱索引失败: file_id={}, error={}",
                file_id,
                exc,
            )

    task = asyncio.create_task(run(), context=Context())
    _graph_tasks[file_id] = task
    task.add_done_callback(
        lambda completed: _graph_tasks.pop(file_id, None)
        if _graph_tasks.get(file_id) is completed
        else None
    )
    return True


async def resume_knowledge_graph_indexing() -> int:
    """恢复中断任务并调度所有 queued 文件。"""
    cfg = Settings.load()
    db = get_db()
    if not cfg.knowledge_graph_enabled:
        return 0

    reset_count = await asyncio.to_thread(
        db.reset_interrupted_knowledge_graph_indexes
    )
    file_ids = await asyncio.to_thread(
        db.list_queued_knowledge_graph_files
    )
    for file_id in file_ids:
        schedule_knowledge_graph_index(file_id)
    if reset_count or file_ids:
        logger.info(
            "知识图谱任务恢复: reset={}, queued={}",
            reset_count,
            len(file_ids),
        )
    return len(file_ids)


async def cancel_knowledge_graph_index(file_id: str) -> bool:
    """取消当前进程中的指定索引任务。"""
    task = _graph_tasks.get(file_id)
    if task is None:
        return False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    return True
