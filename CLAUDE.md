# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentKB is a local-first Personal Knowledge Agent built on LangGraph. Users upload Markdown/text notes, ask questions in natural language, and get answers from their local knowledge base — with web search as a fallback tool. The MVP runs entirely locally with no Docker, no external database, and one-command startup.

See `product.md` for the full PRD.

## Tech Stack

| Module | Technology |
|--------|------------|
| Agent orchestration | LangGraph |
| LLM framework | LangChain + langchain-ollama |
| UI | Gradio |
| Vector store | Qdrant (local file mode) |
| Metadata DB | SQLite (WAL mode) |
| Embedding model | BGE-M3 (via sentence-transformers) |
| Web search | DuckDuckGo |
| Local LLM | Ollama (default: qwen2.5:7b) |

## Architecture

```
Gradio UI → LangGraph Agent (streaming) → Tools → Knowledge Base / Web
                                      ↓
                               SQLite + Qdrant + FileSystem
```

- **LangGraph state machine**: `agent_node` ↔ `tools_node` loop until no more tool_calls
- **ToolRegistry** singleton holds all tools; Agent's LLM is bound via `.bind_tools()`
- **Knowledge base pipeline**: upload → TextLoader → RecursiveCharacterTextSplitter → BGE-M3 → Qdrant
- **Single-session mode** in MVP; schema supports multi-session

## Project Structure

```
src/agentkb/
├── main.py                 # Entry point (to be added in batch 2)
├── config/
│   ├── config.yaml         # All configuration
│   └── settings.py         # Singleton Settings (YAML + env override)
├── agent/
│   ├── graph.py            # LangGraph build + AgentGraph wrapper
│   ├── nodes.py            # agent_node, tools_node
│   ├── state.py            # AgentState (dict-based)
│   └── prompts.py          # System prompts
├── llm/
│   ├── base.py             # LLMProvider ABC
│   ├── ollama_provider.py  # Ollama implementation
│   └── factory.py          # create_llm / get_chat_model
├── tools/
│   ├── base.py             # BaseTool ABC + ToolResult
│   ├── registry.py         # ToolRegistry singleton
│   ├── knowledge_search.py # search_knowledge_base tool
│   └── web_search.py       # search_web tool (DuckDuckGo)
├── knowledge/
│   ├── loader.py           # FileLoader (save + validate .md/.txt)
│   ├── splitter.py         # TextSplitter (RecursiveCharacterTextSplitter)
│   ├── embedder.py         # EmbedderService (BGE-M3 via sentence-transformers)
│   └── vector_store.py     # VectorStore (Qdrant local)
├── storage/
│   ├── database.py         # SQLite connection + CRUD
│   └── models.py           # Pydantic models (Session, Message, KnowledgeFile)
└── utils/
    ├── logger.py           # Loguru config
    └── exceptions.py       # Exception hierarchy
```

## Key Patterns

- **Singletons** for heavy resources: `Settings`, `EmbedderService`, `VectorStore`, `Database`, `ToolRegistry` — loaded once at startup
- **Tool lifecycle**: `BaseTool._execute` is async, `ToolRegistry.execute` wraps with timing/errors, `ToolResult` standardizes output
- **Config override**: env var `AGENTKB_<SECTION>_<KEY>` overrides YAML values (e.g. `AGENTKB_LLM_MODEL_NAME`)
- **Error handling**: custom exception hierarchy; tools catch internally and return `ToolResult(success=False)`, never crash the agent loop

## Running

```bash
# Install
pip install -r requirements.txt

# Prerequisites: Ollama running + model pulled
ollama serve
ollama pull qwen2.5:7b

# Run (from repo root)
export PYTHONPATH="src:$PYTHONPATH"
python -m agentkb.main
```

Or use: `bash scripts/start.sh` / `scripts/start.bat`

App opens at http://127.0.0.1:7860

## AGENTS.md

This repo follows strict engineering standards defined in `AGENTS.md`:
- Chinese comments only, professional and concise
- No dead code, no unused imports, no commented-out blocks
- No compatibility shims or future-proof abstractions
- Every file and line must have a reason to exist
