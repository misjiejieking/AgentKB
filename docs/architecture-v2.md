# AgentKB V2 架构升级方案

> 从单 Agent 检索问答 → 多 Agent 协作知识工作台

---

## 一、现状分析与改造目标

### 当前架构 (V1)

```
用户 → FastAPI → LangGraph ReAct Loop (agent_node ⇄ tools_node)
                    ├── 意图路由 (chat/knowledge/web/hybrid)
                    ├── search_knowledge_base (混合检索 + 重排序)
                    └── search_web (DuckDuckGo)
                          ↓
                   PostgreSQL + pgvector
```

**现有能力**：知识检索、联网搜索、多会话管理、SSE 流式、断点续传、CLI 评估框架、基础 Trace

**核心瓶颈**：
1. 评测是离线 CLI 工具，无法集成到上线流程
2. Trace 信息分散，无结构化 Span 层级，无法对接可观测平台
3. 单 Agent 只能做检索问答，无法处理复杂多步骤任务
4. 工具只有 2 个，无法覆盖创作、任务管理等高频场景

### V2 目标架构

```
┌─────────────────────────────────────────────────────────────┐
│                     Frontend (static/)                       │
├─────────────────────────────────────────────────────────────┤
│                  FastAPI Gateway (/api/*)                    │
│   ┌──────────┬──────────────┬──────────────┬─────────────┐  │
│   │ Chat API │ Eval API     │ Trace API    │ Metrics API │  │
│   └────┬─────┴──────┬───────┴──────┬───────┴──────┬──────┘  │
│        │            │              │              │         │
├────────┼────────────┼──────────────┼──────────────┼─────────┤
│        ▼            ▼              ▼              ▼         │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              Multi-Agent Orchestrator                 │   │
│  │  ┌────────────┐  ┌──────────────────────────────┐   │   │
│  │  │ Supervisor │  │  Agent Registry (插件化)       │   │   │
│  │  │  任务分解   │  │  ├── KnowledgeAgent (检索)    │   │   │
│  │  │  Agent路由  │  │  ├── ContentCreator (创作)   │   │   │
│  │  │  结果聚合   │  │  ├── TaskManager (任务)      │   │   │
│  │  │  质量控制   │  │  ├── LearningTutor (学习)    │   │   │
│  │  └────────────┘  │  └── SocialWriter (社媒)      │   │   │
│  │                   └──────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│        │                                                    │
│  ┌─────┴──────────────────────────────────────────────┐    │
│  │                 Tool Layer (插件化)                  │    │
│  │  ┌──────────┬──────────┬──────────┬──────────────┐ │    │
│  │  │检索/搜索  │ 代码执行  │ Web浏览  │ 日历/任务    │ │    │
│  │  │(已有)    │ (新增)   │ (新增)   │ (新增)       │ │    │
│  │  └──────────┴──────────┴──────────┴──────────────┘ │    │
│  └────────────────────────────────────────────────────┘    │
│        │                                                    │
│  ┌─────┴──────────────────────────────────────────────┐    │
│  │            Memory & Knowledge Layer                  │    │
│  │  ┌────────────┬──────────────┬──────────────────┐  │    │
│  │  │Session Sum │ LongTerm Mem │ Knowledge Base   │  │    │
│  │  │(会话摘要)   │(向量化持久)   │(全局知识上下文)   │  │    │
│  │  └────────────┴──────────────┴──────────────────┘  │    │
│  └────────────────────────────────────────────────────┘    │
│        │                                                    │
│  ┌─────┴──────────────────────────────────────────────┐    │
│  │           Observability Layer (全链路)              │    │
│  │  ┌──────────┬──────────┬──────────┬──────────────┐ │    │
│  │  │Trace收集 │Metrics   │Log聚合   │导出器(可插拔) │ │    │
│  │  │(Span树)  │(Prom查询)│(结构化)  │LangFuse/...  │ │    │
│  │  └──────────┴──────────┴──────────┴──────────────┘ │    │
│  └────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│              PostgreSQL + pgvector (统一存储)                │
│ sessions │ messages │ agent_runs │ run_events │ checkpoints │
└─────────────────────────────────────────────────────────────┘
```

---

## 二、三大改造方向

### 2.1 评测系统 HTTP API 化

**现状**：CLI 工具 (`python -m agentkb.eval run`) ，手动执行、离线报告

**改造后**：

| 端点 | 方法 | 功能 |
|------|------|------|
| `/api/eval/submit` | POST | 提交评估任务（支持单条/批量测试集） |
| `/api/eval/{job_id}/status` | GET | 查询任务进度（pending/running/done） |
| `/api/eval/{job_id}/report` | GET | 获取完整评估报告（JSON） |
| `/api/eval/jobs` | GET | 列出历史评估任务 |
| `/api/eval/compare` | POST | 对比两次评估结果 |

扩展指标：工具调用准确率、回答相关性(LLM-as-Judge)、Token 消耗、端到端耗时、首 Token 延迟

### 2.2 全链路可观测性

**设计原则**：OpenTelemetry 兼容格式 + 可插拔导出器

**Span 层级**：

```
Request (trace_id)
├── Intent Classification (span)
│   ├── fast_match (event)
│   └── llm_classify (event)
├── Agent Execution (span)
│   ├── Query Rewrite (span)
│   ├── Tool: search_knowledge_base (span)
│   │   ├── cache_check (event)
│   │   ├── dense_search (event)
│   │   ├── bm25_search (event)
│   │   └── rrf_fusion (event)
│   └── LLM Generation (span)
│       ├── prompt_tokens (attribute)
│       ├── completion_tokens (attribute)
│       └── first_token_latency (attribute)
└── Response (span)
```

**导出器**：PostgreSQL(查询主存储) / Console / File / LangFuse / LangSmith。
Prometheus 指标直接聚合 HTTP、Agent Run、LLM、工具、检索、缓存和用户反馈事件。

### 2.3 Multi-Agent 架构

**Hierarchical Multi-Agent 模式**：

```
Supervisor Agent (主控)
    │
    ├── 意图分析 → 任务分解 → Agent 路由
    │
    ├── KnowledgeAgent      — 知识检索与问答
    ├── ContentCreator      — 内容创作 (规划→写作→优化)
    ├── TaskManager         — 任务管理 (拆解→分配→跟踪)
    ├── LearningTutor       — 个性化学习 (诊断→路径→材料)
    ├── SocialWriter        — 社媒内容 (平台适配→风格→优化)
    └── MemoryAgent         — 个人记忆 (显式保存→语义检索)
            │
            └── 共享 Context (Checkpoint + Session Summary + LongTerm)
```

**Agent 间通信**：通过共享 State 传递，Supervisor 做最终决策

### 2.4 数据库演进约束

- 当前 `pg_database.py` 中的启动 DDL 视为现有结构基线，不再直接追加业务迁移。
- 后续表结构变更必须使用带版本号、校验值和事务边界的顺序迁移。
- 应用启动时通过 PostgreSQL advisory lock 保证同一迁移只执行一次。
- `agent_runs` 保存生成任务状态，`run_events` 保存可重放的 SSE 事件。
- LangGraph Checkpoint 使用类型化序列化存入 PostgreSQL，服务重启后可恢复上下文。
- `messages.sequence` 提供稳定顺序，不依赖同一事务内相同的时间戳。
- 迁移失败必须阻止应用继续启动，不允许吞错或并行维护新旧结构。

### 2.5 四层上下文边界

| 层级 | 存储 | 生命周期 | 用途 |
|------|------|----------|------|
| 当前任务上下文 | LangGraph Checkpoint | 单图线程 | 节点状态、子任务、工具结果 |
| 会话上下文 | `messages` + `session_summaries` | 单会话 | 最近完整轮次与窗口外增量摘要 |
| 用户长期记忆 | `long_term_memories` | 跨会话 | 用户明确授权保存的事实、偏好与经验 |
| 全局知识上下文 | `knowledge_files` + `knowledge_chunks` | 全局 | 上传文档、业务资料与检索引用 |

Checkpoint 保留完整审计状态，模型调用只接收摘要和最近完整轮次。上下文截断从
`HumanMessage` 边界开始，避免拆断 `AIMessage.tool_calls` 与 `ToolMessage` 的协议配对。

### 2.6 高风险工具确认

- `execute_code` 声明 `requires_confirmation=True`。
- 工具节点先写入 `tool_approvals`，再通过 LangGraph `interrupt` 暂停。
- SSE 发送 `approval_required`，前端提交批准或拒绝后使用 `Command(resume=...)` 恢复。
- Checkpoint、审批和 Run 状态均在 PostgreSQL，刷新不丢失，等待审批时服务重启也可继续。
- 同一 Run 通过条件更新抢占，防止并发批准导致工具重复执行。

---

## 三、模块划分

```
src/agentkb/
├── agent/                  # 现有 Agent 核心 (保留)
│   ├── graph.py
│   ├── nodes.py
│   ├── router.py           # IntentRouter → 升级为 Supervisor 路由
│   ├── query_rewriter.py
│   ├── prompts.py
│   └── state.py            # 扩展状态字段
│
├── agents/                 # 【新增】多 Agent 协作层
│   ├── __init__.py
│   ├── base.py             # SpecialistAgent 基类
│   ├── supervisor.py       # Supervisor Agent
│   ├── registry.py         # AgentRegistry (插件化注册)
│   ├── graph.py            # 依赖 DAG 执行与结果聚合
│   ├── content_creator.py  # 内容创作 Agent 组
│   ├── knowledge_agent.py  # 知识管理 Agent (升级现有)
│   ├── task_manager.py     # 任务管理 Agent
│   ├── learning_tutor.py   # 学习助手 Agent
│   ├── social_writer.py    # 社媒内容 Agent
│   ├── memory_agent.py     # 跨会话个人记忆 Agent
│   └── reflection.py       # Reflection / Self-Critique 机制
│
├── memory/                 # 【新增】增强记忆层
│   ├── __init__.py
│   ├── context.py          # 最近完整轮次 + 持久化增量摘要
│   └── long_term.py        # 长期记忆 (向量化 + 语义检索)
│
├── observability/          # 【新增】可观测性
│   ├── __init__.py
│   ├── tracer.py           # Trace 收集器 (OTel 兼容)
│   ├── spans.py            # Span 管理
│   ├── exporters.py        # 导出器 (Console/File/LangFuse/LangSmith/Prometheus)
│   ├── metrics.py          # 指标收集 (Prometheus 格式)
│   └── middleware.py       # FastAPI 中间件 (自动注入 trace_id)
│
├── eval/                   # 【升级】评测系统
│   ├── __init__.py
│   ├── api.py              # HTTP API 路由
│   ├── jobs.py             # 异步任务管理
│   ├── evaluator.py        # 现有评估器 (保留+扩展)
│   ├── generation_eval.py  # 现有生成评估 (保留)
│   ├── metrics.py          # 现有指标 (保留+扩展)
│   ├── testset.py          # 现有测试集 (保留)
│   ├── reporter.py         # 现有报告 (保留)
│   └── cli.py              # CLI 入口 (保留作为备选)
│
├── tools/                  # 【升级】工具层
│   ├── base.py             # 现有 BaseTool (保留)
│   ├── registry.py         # 现有 ToolRegistry (扩展分类和权限)
│   ├── knowledge_search.py # 现有 (保留)
│   ├── web_search.py       # 现有 (保留)
│   ├── web_browser.py      # 【新增】网页浏览 (headless browser)
│   ├── code_executor.py    # 【新增】代码执行 (沙箱)
│   └── personal_memory.py  # 个人记忆搜索与显式保存
│
├── config/
│   └── config.yaml         # 扩展配置段
│
└── api/
    ├── routes.py           # 扩展 /eval/*、/trace/*、/metrics/* 端点
    └── server.py           # 扩展应用工厂
```

---

## 四、数据流 (以"写一篇小红书笔记"为例)

```
用户: "帮我写一篇关于AgentKB的小红书笔记"

1. FastAPI Middleware → 注入 trace_id = "abc123"
2. Supervisor Agent:
   ├── Span: intent_classification → "social_content"
   ├── Span: task_decomposition → [检索产品信息, 分析小红书风格, 撰写笔记, 优化]
   └── Route → SocialWriter Agent

3. SocialWriter Agent:
   ├── Step1: 调用 KnowledgeAgent 检索 AgentKB 产品信息
   │   └── Span: knowledge_search → dense + bm25 → RRF → 返回产品特性
   ├── Step2: 调用 WebSearch 分析小红书热门AI工具笔记风格
   │   └── Span: web_search → 返回风格参考
   ├── Step3: LLM 写作 (platform="xiaohongshu")
   │   └── Span: llm_generation (tokens_in, tokens_out, model)
   └── Step4: Self-Critique 自检 → 修改 → 最终输出

4. Supervisor: 聚合 → 返回用户

5. Trace 自动上报到 LangFuse (如配置)
6. Metrics: tool_calls_total{name="knowledge_search"} += 1
```

---

## 五、配置扩展

```yaml
# config.yaml 新增段

# 多 Agent 配置
agents:
  supervisor_model: "deepseek-chat"     # 主控模型
  specialist_model: "deepseek-chat"      # 专业 Agent 模型
  max_decomposition_depth: 3          # 最大任务分解深度
  reflection_enabled: true            # 是否启用自检
  reflection_rounds: 2                # 最大自检轮数

# 可观测性
observability:
  enabled: true
  trace_sample_rate: 1.0              # 采样率
  exporters:
    - console
    - file
    # - langfuse
    # - langsmith
    # - prometheus
  langfuse:
    public_key: ""
    secret_key: ""
    host: "https://cloud.langfuse.com"
  langsmith:
    api_key: ""
    project: "agentkb"
  prometheus:
    port: 9090

# 内存系统
memory:
  working_memory_max_turns: 10        # 工作记忆保留最近 N 轮
  long_term_enabled: true
  long_term_min_importance: 0.6       # 重要性阈值

# 知识图谱
knowledge_graph:
  enabled: true
  max_chunks_per_file: 80
  min_chunk_chars: 80

# 多模态
multimodal:
  vision:
    enabled: true
    provider: ollama
    model_name: qwen2.5vl:7b
    pdf_visual_analysis: false
    pdf_max_pages: 12
  transcription:
    enabled: false
    base_url: http://127.0.0.1:8001/v1
    model_name: whisper-1

# 评测 (新增 HTTP 相关)
eval:
  max_concurrent_jobs: 3
  job_timeout_minutes: 30
  default_judge_model: "deepseek-chat"
```

---

## 六、实施路线

| 阶段 | 内容 | 周期 |
|------|------|------|
| Phase 1 | 评测 API + 全链路 Trace | 1-2 周 |
| Phase 2 | Multi-Agent 架构 + 5 个 Specialist Agent | 3-4 周 |
| Phase 3 | 新工具 + Memory 增强 + Reflection | 4-6 周 |
| Phase 4 | 知识图谱（已完成）+ 多模态 + 性能优化 | 6-8 周 |

### 自定义 Agent 生命周期

```text
自然语言描述
  -> AgentDraft 结构化草案
  -> 用户检查职责与工具权限
  -> 确认创建
  -> PostgreSQL custom_agents
  -> AgentRegistry 热注册
  -> Supervisor 动态路由
```

停用或删除会同步移出 `AgentRegistry`。服务启动时仅加载 `active` 定义。
自定义 Agent 的工具集合采用白名单，并排除 `requires_confirmation=True` 的工具。
