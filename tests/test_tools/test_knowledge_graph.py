from __future__ import annotations

from agentkb.tools.knowledge_graph import KnowledgeGraphQueryTool


class _GraphDatabase:
    def search_knowledge_graph(self, query: str):
        return {
            "nodes": [{"id": "1", "name": "AgentKB", "type": "system"}],
            "edges": [{
                "source": "AgentKB",
                "predicate": "使用",
                "target": "PostgreSQL",
            }],
        }


async def test_knowledge_graph_tool_returns_relation_count(monkeypatch) -> None:
    monkeypatch.setattr(
        "agentkb.tools.knowledge_graph.get_db",
        lambda: _GraphDatabase(),
    )

    result = await KnowledgeGraphQueryTool().execute(query="AgentKB")

    assert result.success is True
    assert result.data["relation_count"] == 1
