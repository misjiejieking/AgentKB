# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentKB is a local-first Personal Knowledge Agent. Users upload notes (.md/.txt/.pdf/.docx/.csv/.json), ask questions in natural language, and get answers from their local knowledge base — with web search as a fallback. Runs entirely local with no Docker.

## Tech Stack

| Module | Technology |
|--------|------------|
| Agent orchestration | LangGraph + 自研 Multi-Agent Orchestrator |
| LLM | DeepSeek API（deepseek-chat，OpenAI 兼容协议） |
| Web framework | FastAPI + uvicorn |
| Database | PostgreSQL + pgvector (HNSW index) |
| Embedding | BGE-M3 (via sentence-transformers) |
| BM25 | PostgreSQL tsvector + jieba 分词 |
| Reranker | BGE-Reranker (本地 CrossEncoder，可切换百炼 API) |
| Web search | DuckDuckGo (DDG Lite HTML 回退) |
| Frontend | 原生 HTML/CSS/JS (static/) |
| Observability | 自研 OTel 兼容 Trace + Prometheus 指标 |

## Architecture

### V2 Multi-Agent 架构

```
用户 → static/ → FastAPI → Multi-Agent Orchestrator
                              ├── Supervisor (任务分解 + Agent 路由 + 结果聚合)
                              ├── KnowledgeAgent (知识检索与问答)
                              ├── ContentCreator (规划→写作→优化)
                              ├── TaskManager (任务拆解与跟踪)
                              ├── LearningTutor (学习路径与材料)
                              └── SocialWriter (多平台社媒内容)
                                    ↓
                              Tool Layer (检索/搜索/代码/浏览/日历)
                                    ↓
                              Memory (Working + LongTerm)
                                    ↓
                              PostgreSQL + pgvector
                                    ↓
                              Observability (Trace + Metrics → Exporters)
```

**请求流程**：
1. `ObservabilityMiddleware` 注入 trace_id
2. `SupervisorAgent.analyze()` → 意图分析 + 任务分解
3. `MultiAgentOrchestrator._execute_subtasks()` → 按依赖拓扑执行
4. 每个 Specialist Agent 独立执行，通过共享 context 传递中间结果
5. `ReflectionModule.critique()` → 自检 + 可选的自动修订
6. Trace 自动导出到 Console/File/LangFuse/LangSmith/Prometheus

### 原有单 Agent 模式（兼容保留）

```
用户 → static/ (HTML/CSS/JS) → FastAPI (/api/*) → LangGraph Agent → Tools
                                                       ↓
                                          HybridRetriever (dense + BM25 → RRF)
                                                       ↓
                                              PostgreSQL + pgvector

**Agent 决策流程**：

1. `agent_node` — 意图路由 → chat 直接回复，knowledge/web/hybrid 绑工具
2. `tools_node` — 执行工具调用，收集 ToolMessage 回 agent
3. 循环直到无需工具调用或达到 `max_recursion_limit`

**检索管线**：

```
Query → 查询重写(指代消解) → 语义缓存检查 → HybridRetriever
  ├─ Dense: pgvector cosine (embedding <=>)
  ├─ BM25:  jieba 分词 → tsvector @@ tsquery → ts_rank
  └─ RRF 融合 (dense_weight 0.6 + bm25_weight 0.4, k=60)
→ Reranker 精排(当前 skip，用 RRF 分数排序) → 上下文截断(60% LLM 窗口预算)
```

**分块策略**：上传时自动分析文档特征（标题密度、段落长度方差、短行占比）→ 选择滑动窗口/语义/父子分块。

## Running

```bash
# Prerequisites: PostgreSQL 16+ with pgvector, DeepSeek API Key
psql -U postgres -c "CREATE DATABASE agentkb;"
psql -U postgres -d agentkb -c "CREATE EXTENSION IF NOT EXISTS vector;"
set DEEPSEEK_API_KEY=sk-your-key-here   # Windows
export DEEPSEEK_API_KEY=sk-your-key-here  # macOS/Linux

# Install
pip install -r requirements.txt

# Run (from repo root)
export PYTHONPATH="src:$PYTHONPATH"   # Windows: set PYTHONPATH=src;%PYTHONPATH%
python -m agentkb.main
```

App opens at http://127.0.0.1:8000 (SSE streaming + 断点续传 via Last-Event-ID).

## Commands

```bash
# === 评估（HTTP API 方式，推荐）===
# 提交评估任务
curl -X POST http://127.0.0.1:8000/api/eval/submit \
  -H "Content-Type: application/json" \
  -d '{"testset_path": "data/eval/testset.json", "prompt_version": "v2"}'

# 查询进度
curl http://127.0.0.1:8000/api/eval/{job_id}/status

# 获取报告
curl http://127.0.0.1:8000/api/eval/{job_id}/report

# 对比两次评估
curl -X POST http://127.0.0.1:8000/api/eval/compare \
  -d '{"baseline_job_id": "abc", "current_job_id": "def"}'

# === 评估（CLI 方式，兼容保留）===
PYTHONPATH="src" python -m agentkb.eval generate --sample-size 50
PYTHONPATH="src" python -m agentkb.eval run
PYTHONPATH="src" python -m agentkb.eval run --diff-baseline data/eval/baseline.json

# === 可观测性 ===
curl http://127.0.0.1:8000/api/metrics    # Prometheus 指标
curl http://127.0.0.1:8000/api/health     # 健康检查
curl http://127.0.0.1:8000/api/trace/{id} # Trace 查询

# 测试
pytest tests/ -v
```

## Project Structure

```
src/agentkb/
├── main.py              # 入口：初始化 DB/LLM/Embedder → 注册工具 → 构建 Graph → 启动 uvicorn
├── config/
│   ├── config.yaml      # 所有配置（PostgreSQL / LLM / embedding / retrieval / chunking / eval）
│   └── settings.py      # Settings 单例：YAML + AGENTKB_<SECTION>_<KEY> 环境变量覆盖
├── agent/
│   ├── graph.py          # StateGraph 构建 + AgentGraph 流式封装（astream_events v2）
│   ├── nodes.py          # agent_node（意图路由 + LLM 调用）、tools_node（遍历 tool_calls 执行）
│   ├── router.py         # IntentRouter：快速关键词匹配 + LLM 分类（chat/knowledge/web/hybrid）
│   ├── query_rewriter.py # 指代消解 + BM25 关键词提取（多轮对话场景）
│   ├── prompts.py        # System prompt + 降级话术
│   └── state.py          # AgentState (dict-based)
├── api/
│   ├── server.py         # FastAPI 应用工厂，挂载路由 + 静态文件
│   ├── routes.py         # REST API：聊天 SSE、文件上传、会话管理、反馈、链路追踪
│   └── deps.py           # 依赖注入：get_graph / get_settings / get_session_mgr
├── llm/
│   ├── base.py           # LLMProvider ABC
│   ├── ollama_provider.py
│   └── factory.py        # create_llm / get_chat_model（模块级单例）
├── agents/                 # 【V2 新增】多 Agent 协作层
│   ├── base.py             # SpecialistAgent 基类 + AgentResult
│   ├── supervisor.py       # Supervisor 任务分解/路由/聚合
│   ├── orchestrator.py     # Multi-Agent 编排器
│   ├── registry.py         # Agent 注册表（插件化）
│   ├── reflection.py       # Reflection/Self-Critique 自检
│   ├── knowledge_agent.py  # 知识检索 Agent
│   ├── content_creator.py  # 内容创作 Agent（规划→写作→优化）
│   ├── task_manager.py     # 任务管理 Agent（拆解+跟踪）
│   ├── learning_tutor.py   # 学习导师 Agent
│   └── social_writer.py    # 社媒内容 Agent（多平台适配）
├── memory/                 # 【V2 新增】增强记忆层
│   ├── working.py          # 工作记忆（会话窗口+自动压缩）
│   └── long_term.py        # 长期记忆（向量化+语义检索）
├── observability/          # 【V2 新增】全链路可观测性
│   ├── tracer.py           # Trace/Span 管理（OTel 兼容）
│   ├── exporters.py        # 导出器（Console/File/LangFuse/LangSmith/Prometheus）
│   ├── metrics.py          # 指标收集器
│   └── middleware.py       # FastAPI 中间件（trace_id 注入）
├── tools/
│   ├── base.py           # BaseTool ABC + ToolResult
│   ├── registry.py       # ToolRegistry 单例
│   ├── knowledge_search.py # 混合检索 + 重排序 + 缓存
│   ├── web_search.py     # DuckDuckGo + DDG Lite 回退
│   ├── web_browser.py    # 【V2】网页浏览 + 正文提取
│   ├── code_executor.py  # 【V2】安全沙箱代码执行
│   └── mcp_client.py     # MCP 客户端（stdio 传输）
├── knowledge/
│   ├── loader.py         # FileLoader：pymupdf/pdf + python-docx + 纯文本
│   ├── chunker.py        # TextSplitter + 多策略分块（滑动窗口/语义/父子）+ 自动策略选择
│   ├── embedder.py       # EmbedderService 单例（BGE-M3）
│   ├── retriever.py      # HybridRetriever：dense + BM25 → RRF 融合，支持三级降级
│   ├── reranker.py       # RerankerService（LocalCrossEncoder / BailianAPI）
│   └── cache.py          # QueryCache：基于 embedding 余弦相似度的语义缓存
├── storage/
│   ├── pg_database.py    # Database 单例：psycopg2 连接池 + 建表 + CRUD + pgvector 检索
│   └── models.py         # Pydantic 数据模型 + ID/时间工具函数
├── session/
│   └── manager.py        # SessionManager：会话 CRUD + LangChain 消息序列化
├── eval/
│   ├── api.py             # 【V2】HTTP API（POST /eval/submit, GET /eval/{id}/status, GET /eval/{id}/report）
│   ├── jobs.py            # 【V2】异步任务管理器（内存队列 + 并发控制 + 实时进度）
│   ├── cli.py             # CLI 入口（generate / run / compare / report / generation）
│   ├── testset.py         # TestSet：银标测试集生成 + 加载/保存/验证 + from_queries
│   ├── evaluator.py       # Evaluator：串联检索管线 + 指标计算 + 对比报告
│   ├── metrics.py         # Recall@K / Precision@K / MRR / NDCG@K（含中文注释公式说明）
│   ├── generation_eval.py # 生成质量评估（Faithfulness / Answer Relevance / Context Relevance）
│   └── reporter.py       # Markdown 报告渲染
└── utils/
    ├── logger.py         # Loguru 配置
    ├── exceptions.py     # 异常层级
    └── tracer.py         # 轻量级 Trace 记录器（记录 LLM/tool/retrieval 耗时）
```

## Key Patterns

- **根目录 `main.py`** 是 PyCharm 生成的占位文件，不是入口。真正的入口是 `src/agentkb/main.py`
- **`requirements.txt`** 是实际依赖（比 `pyproject.toml` 新）；`pyproject.toml` 依赖列表已过时
- **Multi-Agent 模式**：`SupervisorAgent.analyze()` 做意图分类 + 任务分解，`MultiAgentOrchestrator` 按依赖拓扑执行 subtasks，默认启用 Reflection 自检
- **单 Agent 兼容模式**：保留原有 LangGraph ReAct loop，`IntentRouter` 先用关键词快速匹配再走 LLM
- **Router LLM 分离**：`router_model_name` 用于意图路由，`generator_model_name` 用于回复生成（默认都用 deepseek-chat）
- **Checkpointer** 用的是 aiosqlite（与 LangGraph 的 AsyncSqliteSaver），独立于 PG 业务数据库
- **Agent 注册表**：`AgentRegistry` 单例，插件化注册 Specialist Agent，按 intent 自动路由
- **可观测性**：`ObservabilityMiddleware` 自动注入 trace_id，`TraceManager` 管理 Span 层级树，多导出器可插拔配置
- **SSE 断点续传**：`SessionStream` 缓存事件（带自增 id），支持 Last-Event-ID 重连补发
- **评估 HTTP API**：异步任务提交 → 实时进度轮询 → 完整 JSON 报告，支持多任务对比
- **语义缓存**：文件上传/删除时失效；用 embedding 余弦相似度 > 0.95 判定命中
- **CI 评估门**：`.github/workflows/eval.yml` 在 PR 修改检索代码时自动跑评估，Recall@5 退化 >2% 则 fail

## AGENTS.md

本仓库严格遵循 `AGENTS.md` 工程标准：
- 中文注释，专业克制，只解释 WHY 不解释 WHAT
- 零死代码、零注释掉的旧实现、零兼容性 shim
- 禁止"为未来而设计"的抽象
- 涉及框架/库升级须参考最新官方文档（context7 MCP）
