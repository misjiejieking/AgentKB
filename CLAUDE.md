# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AgentKB is a local-first Personal Knowledge Agent. Users upload notes (.md/.txt/.pdf/.docx/.csv/.json), ask questions in natural language, and get answers from their local knowledge base — with web search as a fallback. Runs entirely local with no Docker.

## Tech Stack

| Module | Technology |
|--------|------------|
| Agent orchestration | LangGraph StateGraph + Supervisor 依赖 DAG |
| LLM | Ollama / DeepSeek / OpenAI 及其他 OpenAI 兼容 API |
| Web framework | FastAPI + uvicorn |
| Database | PostgreSQL + pgvector (HNSW index) |
| Embedding | BGE-M3 (via sentence-transformers) |
| BM25 | PostgreSQL tsvector + jieba 分词 |
| Knowledge graph | PostgreSQL 实体/关系/证据表 + LLM 结构化抽取 |
| Reranker | BGE-Reranker (本地 CrossEncoder，可切换百炼 API) |
| Web search | DuckDuckGo (DDG Lite HTML 回退) |
| Frontend | 原生 HTML/CSS/JS (static/) |
| Observability | 自研 OTel 兼容 Trace + Prometheus 指标 |

## Architecture

### V2 Multi-Agent 架构

```
用户 → static/ → FastAPI → Multi-Agent Graph
                              ├── Supervisor (任务分解 + Agent 路由 + 结果聚合)
                              ├── KnowledgeAgent (知识检索与问答)
                              ├── ContentCreator (规划→写作→优化)
                              ├── TaskManager (任务拆解与跟踪)
                              ├── LearningTutor (学习路径与材料)
                              └── SocialWriter (多平台社媒内容)
                                    ↓
                              Tool Layer (检索/搜索/代码/浏览/日历)
                                    ↓
                              Memory (会话摘要 + LongTerm)
                                    ↓
                              PostgreSQL + pgvector
                                    ↓
                              Observability (Trace + Metrics → Exporters)
```

**请求流程**：
1. `ObservabilityMiddleware` 注入 trace_id
2. `SupervisorAgent.analyze()` → 意图分析 + 任务分解
3. `MultiAgentGraph` → 按依赖拓扑并发执行
4. 每个 Specialist Agent 独立执行，通过共享 context 传递中间结果
5. `ReflectionModule.critique()` → 自检 + 可选的自动修订
6. Trace 自动导出到 Console/File/PostgreSQL/LangFuse/LangSmith

### 单 Agent 模式

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

**知识图谱链路**：文件分块入库后进入持久化索引状态机，使用
`with_structured_output` 抽取实体与关系，写入 PostgreSQL；服务重启会恢复
`queued/processing` 文件，关系查询始终返回文件和原文证据。

**分块策略**：上传时自动分析文档特征（标题密度、段落长度方差、短行占比）→ 选择滑动窗口/语义/父子分块。

## Running

```bash
# Prerequisites: PostgreSQL 16+ with pgvector
psql -U postgres -c "CREATE DATABASE agentkb;"
psql -U postgres -d agentkb -c "CREATE EXTENSION IF NOT EXISTS vector;"
# 默认使用本地 Ollama；切换云端厂商时配置对应 API Key
set AGENTKB_LLM_PROVIDER=deepseek
set DEEPSEEK_API_KEY=sk-your-key-here

# Install
pip install -r requirements.txt

# Run (from repo root)
export PYTHONPATH="src:$PYTHONPATH"   # Windows: set PYTHONPATH=src;%PYTHONPATH%
python -m agentkb.main
```

App opens at http://127.0.0.1:8000 (SSE streaming + 断点续传 via Last-Event-ID).

多模态配置位于 `multimodal`：视觉模型可使用与主 LLM 不同的 provider；
PDF 深度视觉解析默认关闭，语音按钮仅在转写服务启用时显示。

自定义 Agent 通过 `/api/agents/draft` 生成结构化草案，必须再调用
`POST /api/agents` 确认创建。定义保存在 PostgreSQL，运行时热注册；
需要人工确认的高风险工具禁止分配给自定义 Agent。

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

# 创建激活基线并执行质量门禁
curl -X POST http://127.0.0.1:8000/api/eval/baselines \
  -d '{"job_id": "abc", "name": "retrieval-v1", "scope": "default"}'
curl -X POST http://127.0.0.1:8000/api/eval/gates \
  -d '{"current_job_id": "def", "scope": "default"}'

# 对比两次评估
curl -X POST http://127.0.0.1:8000/api/eval/compare \
  -d '{"baseline_job_id": "abc", "current_job_id": "def"}'

# === 评估（CLI 方式）===
PYTHONPATH="src" python -m agentkb.eval generate --sample-size 50
PYTHONPATH="src" python -m agentkb.eval run
PYTHONPATH="src" python -m agentkb.eval baseline \
  --input data/eval/latest_eval.json --output data/eval/baseline.json
PYTHONPATH="src" python -m agentkb.eval run --gate-baseline data/eval/baseline.json

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
│   ├── openai_compatible_provider.py
│   └── factory.py        # 根据配置创建路由与生成模型
├── agents/                 # 【V2 新增】多 Agent 协作层
│   ├── base.py             # SpecialistAgent 基类 + AgentResult
│   ├── supervisor.py       # Supervisor 任务分解/路由/聚合
│   ├── registry.py         # Agent 注册表（插件化）
│   ├── reflection.py       # Reflection/Self-Critique 自检
│   ├── knowledge_agent.py  # 知识检索 Agent
│   ├── content_creator.py  # 内容创作 Agent（规划→写作→优化）
│   ├── task_manager.py     # 任务管理 Agent（拆解+跟踪）
│   ├── learning_tutor.py   # 学习导师 Agent
│   ├── social_writer.py    # 社媒内容 Agent（多平台适配）
│   └── memory_agent.py     # 跨会话个人记忆 Agent
├── memory/                 # 【V2 新增】增强记忆层
│   ├── context.py          # 会话窗口选择 + PostgreSQL 增量摘要
│   └── long_term.py        # 长期记忆（向量化+语义检索）
├── observability/          # 【V2 新增】全链路可观测性
│   ├── tracer.py           # Trace/Span 管理（OTel 兼容）
│   ├── exporters.py        # Trace 导出器（Console/File/PostgreSQL/LangFuse/LangSmith）
│   ├── metrics.py          # 指标收集器
│   └── middleware.py       # FastAPI 中间件（trace_id 注入）
├── tools/
│   ├── base.py           # BaseTool ABC + ToolResult
│   ├── registry.py       # ToolRegistry 单例
│   ├── knowledge_search.py # 混合检索 + 重排序 + 缓存
│   ├── web_search.py     # DuckDuckGo + DDG Lite 回退
│   ├── web_browser.py    # 【V2】网页浏览 + 正文提取
│   ├── code_executor.py  # 【V2】安全沙箱代码执行
│   └── personal_memory.py # 个人记忆搜索与显式保存
├── knowledge/
│   ├── loader.py         # FileLoader：pymupdf/pdf + python-docx + 纯文本
│   ├── chunker.py        # TextSplitter + 多策略分块（滑动窗口/语义/父子）+ 自动策略选择
│   ├── embedder.py       # EmbedderService 单例（BGE-M3）
│   ├── retriever.py      # HybridRetriever：dense + BM25 → RRF 融合，支持三级降级
│   ├── reranker.py       # RerankerService（LocalCrossEncoder / BailianAPI）
│   └── cache.py          # QueryCache：基于 embedding 余弦相似度的语义缓存
├── storage/
│   ├── pg_database.py    # Database 单例：连接池、CRUD、Run/Event 与 pgvector
│   ├── migrations.py     # 带校验和的顺序迁移
│   ├── checkpointer.py   # PostgreSQL LangGraph Checkpointer
│   └── models.py         # ID 工具函数
├── session/
│   └── manager.py        # SessionManager：会话 CRUD + LangChain 消息序列化
├── eval/
│   ├── api.py             # 【V2】HTTP API（POST /eval/submit, GET /eval/{id}/status, GET /eval/{id}/report）
│   ├── jobs.py            # 【V2】PostgreSQL 持久化任务管理器
│   ├── cli.py             # CLI 入口（generate / run / compare / report / generation）
│   ├── testset.py         # TestSet：银标测试集生成 + 加载/保存/验证 + from_queries
│   ├── evaluator.py       # Evaluator：串联检索管线 + 指标计算 + 对比报告
│   ├── metrics.py         # Recall@K / Precision@K / MRR / NDCG@K（含中文注释公式说明）
│   ├── generation_eval.py # 生成质量评估（Faithfulness / Answer Relevance / Context Relevance）
│   └── reporter.py       # Markdown 报告渲染
└── utils/
    ├── logger.py         # Loguru 配置
    └── exceptions.py     # 异常层级
```

## Key Patterns

- **应用入口**：`src/agentkb/main.py`
- **依赖声明**：`pyproject.toml` 与 `requirements.txt` 保持同步
- **Multi-Agent 模式**：`SupervisorAgent.analyze()` 做任务分解，`MultiAgentGraph` 按依赖拓扑并发执行 subtasks
- **单 Agent 模式**：LangGraph ReAct loop，`IntentRouter` 先用关键词快速匹配再走 LLM
- **LLM 配置**：`llm.provider` 选择厂商配置，路由模型与生成模型独立设置
- **Checkpointer**：使用统一 PostgreSQL 连接池持久化，可恢复中断与人工审批状态
- **Agent 注册表**：`AgentRegistry` 单例，插件化注册 Specialist Agent，按 intent 自动路由
- **可观测性**：`ObservabilityMiddleware` 自动注入 trace_id，`TraceManager` 管理 Span 层级树，多导出器可插拔配置
- **SSE 断点续传**：Run/Event 持久化事件并携带自增序号，支持 Last-Event-ID 重连补发
- **评估 HTTP API**：任务和进度持久化到 PostgreSQL，支持报告与多任务对比
- **语义缓存**：文件上传/删除时失效；用 embedding 余弦相似度 > 0.95 判定命中
- **CI 评估门**：`.github/workflows/eval.yml` 使用版本化基线执行 Recall、Precision、MRR、NDCG 多指标门禁

## AGENTS.md

本仓库严格遵循 `AGENTS.md` 工程标准：
- 中文注释，专业克制，只解释 WHY 不解释 WHAT
- 零死代码、零注释掉的旧实现、零兼容性 shim
- 禁止"为未来而设计"的抽象
- 涉及框架/库升级须参考最新官方文档（context7 MCP）
