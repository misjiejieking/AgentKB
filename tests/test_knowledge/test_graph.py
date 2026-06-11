from __future__ import annotations

from typing import Any, cast

from agentkb.config.settings import Settings
from agentkb.knowledge.graph import (
    ExtractedEntity,
    ExtractedRelation,
    KnowledgeGraphExtraction,
    KnowledgeGraphIndexer,
    normalize_graph_term,
)
from agentkb.storage.pg_database import Database


class _StructuredModel:
    def __init__(self, extraction: KnowledgeGraphExtraction) -> None:
        self._extraction = extraction

    def with_structured_output(self, schema):
        assert schema is KnowledgeGraphExtraction
        return self

    async def ainvoke(self, prompt: str) -> KnowledgeGraphExtraction:
        assert "AgentKB 使用 PostgreSQL" in prompt
        return self._extraction


class _GraphDatabase:
    def __init__(self) -> None:
        self.statuses: list[tuple[str, str]] = []
        self.records: list[dict[str, Any]] = []

    def claim_knowledge_graph_index(self, file_id: str) -> bool:
        return True

    def get_chunks_by_file_id(self, file_id: str, limit: int):
        return [{
            "id": "00000000-0000-0000-0000-000000000001",
            "content": "AgentKB 使用 PostgreSQL 持久化会话、知识和运行状态。" * 3,
        }]

    def replace_knowledge_graph(self, file_id: str, records: list[dict]):
        self.records = records
        return {"entities": 2, "relations": 1}

    def update_knowledge_graph_status(
        self,
        file_id: str,
        status: str,
        error: str = "",
    ) -> None:
        self.statuses.append((status, error))


def _graph_settings() -> Settings:
    return Settings({
        "knowledge_graph": {
            "enabled": True,
            "max_chunks_per_file": 10,
            "min_chunk_chars": 20,
        },
        "llm": {"request_timeout": 10},
    })


def test_normalize_graph_term_is_stable() -> None:
    assert normalize_graph_term("  Agent KB（系统） ") == "agentkb系统"


async def test_indexer_persists_structured_entities_and_relations(monkeypatch) -> None:
    monkeypatch.setattr(Settings, "_instance", _graph_settings())
    database = _GraphDatabase()
    extraction = KnowledgeGraphExtraction(
        entities=[
            ExtractedEntity(name="AgentKB", entity_type="system"),
            ExtractedEntity(name="PostgreSQL", entity_type="system"),
        ],
        relations=[
            ExtractedRelation(
                source="AgentKB",
                target="PostgreSQL",
                predicate="使用",
                evidence="AgentKB 使用 PostgreSQL 持久化",
                confidence=0.95,
            ),
        ],
    )
    indexer = KnowledgeGraphIndexer(
        cast(Database, database),
        _StructuredModel(extraction),
    )

    result = await indexer.index_file("file-1")

    assert result == {"entities": 2, "relations": 1}
    assert database.statuses == [("ready", "")]
    assert database.records[0]["relations"][0]["normalized_source"] == "agentkb"
    assert database.records[0]["relations"][0]["normalized_target"] == "postgresql"


def test_build_record_discards_relation_with_unknown_entity() -> None:
    extraction = KnowledgeGraphExtraction(
        entities=[ExtractedEntity(name="AgentKB", entity_type="system")],
        relations=[
            ExtractedRelation(
                source="AgentKB",
                target="PostgreSQL",
                predicate="使用",
            ),
        ],
    )

    record = KnowledgeGraphIndexer._build_record(
        {"id": "chunk-1", "content": "content"},
        extraction,
    )

    assert record["relations"] == []
