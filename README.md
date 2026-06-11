# AgentKB

本地优先的个人知识 Agent 工作台（Personal Knowledge Agent Workbench），从单 Agent 检索问答升级为 **多 Agent 协作知识工作台**。

## 功能

### 核心能力
- **智能对话**：自然语言提问，流式 Markdown 回复（SSE + 断点续传）
- **双 LLM 架构**：Router（轻量意图路由）+ Generator（深度生成），支持 Ollama、DeepSeek、OpenAI 兼容厂商灵活切换
- **多模态知识库**：上传文档或 JPEG / PNG / WebP 图片，支持可选 PDF 扫描页与图表视觉解析
- **知识图谱**：上传后后台抽取实体关系，保留原文证据并支持关系查询与可视化浏览
- **多策略分块**：滑动窗口 / 语义分块 / 父子分块，自动根据文档特征选择
- **联网搜索**：DuckDuckGo（含 DDG Lite 回退）+ 网页全文浏览
- **多会话**：完整的会话生命周期管理，刷新/断线不丢回复

### V2 新增能力
- **多 Agent 协作**：Supervisor + 6 个 Specialist Agent（知识检索 / 内容创作 / 任务管理 / 学习助手 / 社媒内容 / 个人记忆）
- **LLM 分层路由**：Router LLM（轻量意图分类）+ Generator LLM（深度生成），独立配置不同模型
- **智能内容创作**：生成文章、短视频脚本、朋友圈文案、简历、报告等（规划→写作→优化三阶段）
- **任务管理**：自动拆解任务、创建项目计划、进度跟踪
- **个性化学习**：诊断知识水平、生成学习路径、定制学习材料
- **社媒内容分发**：适配小红书/抖音/公众号/LinkedIn/朋友圈的定制内容生成
- **全链路可观测**：PostgreSQL Trace、Prometheus 指标、多导出器（Console/File/LangFuse/LangSmith）
- **HTTP 评估 API**：PostgreSQL 持久化任务、实时进度查询、多指标对比报告
- **Reflection 自检**：自动审查输出质量、修正幻觉和错误
- **分层上下文**：完整 Checkpoint、持久化会话摘要、跨会话个人记忆和全局知识库分层存储
- **人机确认**：代码执行使用持久化审批，可在刷新或重启后继续

## 快速开始

### 环境要求

- Python 3.11+
- [PostgreSQL](https://www.postgresql.org/) 16+ 已安装并运行
- pgvector 扩展
- LLM：任选其一
  - [Ollama](https://ollama.com/) 本地模型（推荐开发环境：`ollama pull qwen2.5:7b`）
  - [DeepSeek API Key](https://platform.deepseek.com/)（云端模型）
  - OpenAI 兼容 API 服务

### 安装与启动

```bash
# 0. 创建数据库并启用 pgvector 扩展
psql -U postgres -c "CREATE DATABASE agentkb;"
psql -U postgres -d agentkb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 1. 配置 LLM —— 编辑 src/agentkb/config/config.yaml
#    各厂商配置档相互独立，切换时只需修改 llm.provider

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动
# Windows:
set PYTHONPATH=src;%PYTHONPATH%
python -m agentkb.main

# macOS / Linux:
export PYTHONPATH="src:$PYTHONPATH"
python -m agentkb.main
```

浏览器访问 http://127.0.0.1:8000

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 编排 | LangGraph StateGraph + Supervisor 依赖 DAG |
| LLM | Ollama 本地模型 / DeepSeek、OpenAI 及其他 OpenAI 兼容 API |
| 向量存储 | pgvector (PostgreSQL, HNSW 索引) |
| 向量模型 | BGE-M3 |
| Web 框架 | FastAPI + uvicorn |
| 前端 | 原生 HTML/CSS/JS (SSE streaming) |
| 混合检索 | 稠密向量 + BM25 (jieba + tsvector, RRF 融合) |
| 重排序 | BGE-Reranker (本地 + 百炼 API) |
| 可观测性 | 自研 OTel 兼容 Trace + Prometheus 指标 |
| 评估 | HTTP API 异步评估 + LLM-as-Judge 生成质量 |

## 项目结构

```
src/agentkb/
├── main.py              # 入口：初始化 → 注册所有组件 → 启动 uvicorn
├── config/              # YAML 配置 + Settings 单例（env var 覆盖）
├── agent/               # LangGraph 状态机 + 意图路由 + 查询重写
├── agents/              # 【V2 新增】Multi-Agent 协作层
│   ├── supervisor.py    #   Supervisor：任务分解、Agent 路由、结果聚合
│   ├── graph.py         #   依赖分析、并行执行、聚合与 Reflection
│   ├── base.py          #   SpecialistAgent 基类
│   ├── registry.py      #   Agent 注册表（插件化）
│   ├── reflection.py    #   Reflection 自检模块
│   ├── knowledge_agent.py   # 知识检索 Agent
│   ├── content_creator.py   # 内容创作 Agent（规划→写作→优化）
│   ├── task_manager.py      # 任务管理 Agent
│   ├── learning_tutor.py    # 学习导师 Agent
│   ├── social_writer.py     # 社媒内容 Agent
│   └── memory_agent.py      # 跨会话个人记忆 Agent
├── memory/              # 【V2 新增】增强记忆层
│   ├── context.py       #   会话窗口选择 + PostgreSQL 增量摘要
│   └── long_term.py     #   长期记忆（向量化持久 + 语义检索）
├── observability/       # 【V2 新增】全链路可观测性
│   ├── tracer.py        #   Trace/Span 管理（OTel 兼容）
│   ├── exporters.py     #   Trace 导出器（Console/File/PostgreSQL/LangFuse/LangSmith）
│   ├── metrics.py       #   指标收集器（Prometheus 格式）
│   └── middleware.py    #   FastAPI 中间件（自动注入 trace_id）
├── api/                 # FastAPI REST API
│   ├── routes.py        #   聊天/上传/会话/反馈/Trace/Metrics/Health
│   └── server.py        #   应用工厂 + 静态文件挂载
├── tools/               # 工具系统（插件化注册）
│   ├── knowledge_search.py  # 混合检索 + 重排序 + 缓存
│   ├── web_search.py        # DuckDuckGo + DDG Lite 回退
│   ├── web_browser.py       # 【V2 新增】网页浏览与正文提取
│   ├── code_executor.py     # 【V2 新增】带超时的进程级代码执行
│   └── personal_memory.py   # 个人记忆搜索与显式保存工具
├── knowledge/           # 知识库管道
│   ├── chunker.py       #   多策略分块 + 自动策略选择
│   ├── graph.py         #   结构化实体关系抽取 + 可恢复后台索引
│   ├── retriever.py     #   混合检索 + RRF 融合 + 三级降级
│   ├── reranker.py      #   精排（本地/百炼 API）
│   ├── cache.py         #   语义缓存
│   └── embedder.py      #   BGE-M3 向量化
├── eval/                # 评估框架
│   ├── api.py           #   【V2 新增】HTTP API（提交/进度/报告/对比）
│   ├── jobs.py          #   【V2 新增】异步任务管理器
│   ├── evaluator.py     #   检索评估编排
│   ├── metrics.py       #   Recall@K / MRR / NDCG@K
│   └── cli.py           #   CLI 入口（保留）
├── session/             # 会话管理
├── storage/             # PostgreSQL、顺序迁移、Run/Event 与 Checkpoint
└── utils/               # 日志与异常定义
```

## 评估

```bash
# === HTTP API 方式（推荐）===

# 提交评估任务
curl -X POST http://127.0.0.1:8000/api/eval/submit \
  -H "Content-Type: application/json" \
  -d '{"testset_path": "data/eval/testset.json", "k_values": [5, 10, 20], "prompt_version": "v2"}'

# 查询进度
curl http://127.0.0.1:8000/api/eval/abc123/status

# 获取报告
curl http://127.0.0.1:8000/api/eval/abc123/report

# 创建并激活质量基线
curl -X POST http://127.0.0.1:8000/api/eval/baselines \
  -H "Content-Type: application/json" \
  -d '{"job_id": "abc123", "name": "retrieval-v1", "scope": "default"}'

# 对已完成任务执行质量门禁
curl -X POST http://127.0.0.1:8000/api/eval/gates \
  -H "Content-Type: application/json" \
  -d '{"current_job_id": "def456", "scope": "default"}'

# 对比任务
curl -X POST http://127.0.0.1:8000/api/eval/compare \
  -H "Content-Type: application/json" \
  -d '{"baseline_job_id": "abc123", "current_job_id": "def456"}'

# === CLI 方式 ===

# 生成银标测试集
PYTHONPATH="src" python -m agentkb.eval generate --sample-size 50

# 跑评估
PYTHONPATH="src" python -m agentkb.eval run

# 从评估结果创建版本化基线
PYTHONPATH="src" python -m agentkb.eval baseline \
  --input data/eval/latest_eval.json \
  --output data/eval/baseline.json

# 执行质量门禁，任一规则失败时返回非零退出码
PYTHONPATH="src" python -m agentkb.eval run \
  --gate-baseline data/eval/baseline.json

# 对比评估
PYTHONPATH="src" python -m agentkb.eval compare --baseline baseline.json --current after.json
```

`.github/workflows/eval.yml` 使用带 `agentkb-eval` 标签的 self-hosted Runner，
以复用本地模型和完整评估语料；云端空数据库不会被误当作有效评估环境。

## 多模态

视觉模型与主对话模型独立配置。例如主模型使用 DeepSeek、图片理解使用本地 Ollama：

```yaml
llm:
  provider: deepseek

multimodal:
  vision:
    enabled: true
    provider: ollama
    model_name: qwen2.5vl:7b
    pdf_visual_analysis: false
  transcription:
    enabled: false
    base_url: http://127.0.0.1:8001/v1
    model_name: whisper-1
```

```bash
# 准备本地视觉模型
ollama pull qwen2.5vl:7b
```

- 输入框图片作为当前会话附件，视觉描述会持久化并随历史消息恢复。
- 顶部上传按钮可将图片作为知识内容入库。
- `pdf_visual_analysis` 启用后，会分析扫描页及含图片页面，最多处理 `pdf_max_pages` 页。
- 语音输入需要 OpenAI 兼容的 `/audio/transcriptions` 服务；启用后前端才显示录音按钮。

## 自定义 Agent

顶栏的 Agent 工坊支持用自然语言创建专业 Agent：

1. 描述职责，例如“创建一个需求评审 Agent，检查完整性并输出验收标准”。
2. 系统生成结构化草案，展示路由描述、执行指令、意图和工具权限。
3. 用户检查并确认后，Agent 才写入 PostgreSQL 并注册到 Supervisor。
4. Agent 可在界面中即时启用、停用或删除，重启后自动恢复激活配置。

自定义 Agent 只能选择无需人工确认的工具。涉及高风险操作的工具不会出现在可选列表中，
防止动态 Agent 绕过现有审批机制。

## MCP 工具生态

顶栏的 MCP 服务管理支持接入官方协议服务：

- `stdio`：启动本机 MCP Server，适合文件系统、Git、数据库等本地工具。
- `Streamable HTTP`：连接远程 MCP 端点，请求头可使用环境变量模板。
- 工具发现后以 `mcp__服务名__工具名` 注册，避免与内置工具重名。
- 默认 `writes` 策略只放行明确声明 `readOnlyHint=true` 的工具；其余工具调用前必须人工确认。
- 服务配置、发现结果和工具启停状态持久化到 PostgreSQL，连接后可在重启时恢复。

密钥不要直接写入界面。使用 `${ENV_VAR}`，例如：

```json
{"Authorization": "Bearer ${MCP_TOKEN}"}
```

新建配置不会立即执行命令。用户显式点击“连接”且发现工具成功后，服务才会标记为启用。

## 可观测性

```bash
# Prometheus 指标
curl http://127.0.0.1:8000/api/metrics

# 查询 Trace
curl http://127.0.0.1:8000/api/trace/{trace_id}

# 健康检查
curl http://127.0.0.1:8000/api/health
```

在 `config.yaml` 中配置导出器:

```yaml
observability:
  enabled: true
  exporters:
    - console       # 开发环境
    - file          # JSON 持久化到 data/traces/
    # - langfuse     # 对接 LangFuse 平台
    # - prometheus   # Prometheus 指标
```

## 配置

编辑 `src/agentkb/config/config.yaml`：

```yaml
llm:
  provider: ollama
  providers:
    ollama:
      protocol: ollama
      base_url: http://localhost:11434
      api_key: ""
      router_model_name: qwen2.5:7b
      generator_model_name: qwen2.5:7b
    deepseek:
      protocol: openai
      base_url: https://api.deepseek.com/v1
      api_key: ""
      router_model_name: deepseek-chat
      generator_model_name: deepseek-chat
    openai:
      protocol: openai
      base_url: https://api.openai.com/v1
      api_key: ""
      router_model_name: gpt-4.1-mini
      generator_model_name: gpt-4.1
```

切换到 DeepSeek 时设置 `AGENTKB_LLM_PROVIDER=deepseek` 和
`DEEPSEEK_API_KEY`；切换到 OpenAI 时设置 `AGENTKB_LLM_PROVIDER=openai`
和 `OPENAI_API_KEY`。URL、协议和模型会从对应配置档读取，不会沿用 Ollama
配置。

其他 OpenAI 兼容厂商只需在 `providers` 下新增配置档：

```yaml
llm:
  provider: siliconflow
  providers:
    siliconflow:
      protocol: openai
      base_url: https://api.siliconflow.cn/v1
      api_key: ""
      router_model_name: Qwen/Qwen2.5-7B-Instruct
      generator_model_name: Qwen/Qwen2.5-72B-Instruct
```

自定义厂商的密钥环境变量名称为 `<PROVIDER>_API_KEY`，例如
`SILICONFLOW_API_KEY`。所有配置项也支持完整路径环境变量覆盖，例如
`AGENTKB_LLM_PROVIDERS_DEEPSEEK_GENERATOR_MODEL_NAME`。

## 文档

- **[AGENTS.md](AGENTS.md)** — AI 协作工程标准（中文注释规范、零死代码、禁止过度抽象）
- **[product.md](product.md)** — 产品需求文档（用户故事、功能规格、交互流程）
- **[CLAUDE.md](CLAUDE.md)** — Claude Code 项目指令（架构说明、开发命令、关键模式）

## Roadmap

### V2.0（当前）

- [x] 评测系统 HTTP API 化（异步任务 + 实时进度 + 扩展指标）
- [x] 全链路可观测性（OTel 兼容 Trace + Prometheus 指标 + 多导出器）
- [x] Multi-Agent 架构（Supervisor + 5 个 Specialist Agent）
- [x] 智能内容创作（三阶段 Pipeline: 规划→写作→优化）
- [x] 任务与工作流管理（自动拆解 + 进度跟踪）
- [x] 个性化学习助手（学习路径 + 定制材料）
- [x] 社媒内容分发（多平台适配）
- [x] Reflection / Self-Critique 机制
- [x] 增强记忆系统（工作记忆 + 长期记忆）
- [x] 新工具（代码执行、网页浏览）

### V2.1（下一迭代）

- [x] 知识图谱构建（实体关系抽取 + 图查询）
- [x] 多模态支持（图片理解、PDF 深度解析、可配置语音转写）
- [x] Agent 对话式注册（草案确认 + 权限约束 + 持久化热加载）
- [x] 评估基线自动管理（持久化基线 + 多指标 CI/CD 质量门禁）
- [x] MCP 生态深度集成（stdio / Streamable HTTP、工具发现、权限确认、热管理）

### V3.0（远期）

- [ ] 多用户 + 权限系统
- [ ] 云端同步 + 协作
- [ ] Electron 桌面客户端
- [ ] 企业知识库（多空间 + 团队共享）
- [ ] 主动 Agent（定时巡检 + 自动任务执行）
- [ ] 插件市场
