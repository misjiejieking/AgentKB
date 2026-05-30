# AgentKB

本地优先的个人知识 Agent 工作台（Personal Knowledge Agent Workbench），从单 Agent 检索问答升级为 **多 Agent 协作知识工作台**。

## 功能

### 核心能力
- **智能对话**：自然语言提问，流式 Markdown 回复（SSE + 断点续传）
- **双 LLM 架构**：Router（轻量意图路由）+ Generator（深度生成），支持 Ollama/DeepSeek 灵活切换
- **知识库**：上传 .md / .txt / .pdf / .docx / .csv / .json，混合检索（pgvector dense + jieba BM25 + RRF 融合）+ 重排序
- **多策略分块**：滑动窗口 / 语义分块 / 父子分块，自动根据文档特征选择
- **联网搜索**：DuckDuckGo（含 DDG Lite 回退）+ 网页全文浏览
- **多会话**：完整的会话生命周期管理，刷新/断线不丢回复

### V2 新增能力
- **多 Agent 协作**：Supervisor + 5 个 Specialist Agent（知识检索 / 内容创作 / 任务管理 / 学习助手 / 社媒内容）
- **LLM 分层路由**：Router LLM（轻量意图分类）+ Generator LLM（深度生成），独立配置不同模型
- **智能内容创作**：生成文章、短视频脚本、朋友圈文案、简历、报告等（规划→写作→优化三阶段）
- **任务管理**：自动拆解任务、创建项目计划、进度跟踪
- **个性化学习**：诊断知识水平、生成学习路径、定制学习材料
- **社媒内容分发**：适配小红书/抖音/公众号/LinkedIn/朋友圈的定制内容生成
- **全链路可观测**：OpenTelemetry 兼容 Trace、Prometheus 指标、多导出器（Console/File/LangFuse/LangSmith）
- **HTTP 评估 API**：异步提交评估任务、实时进度查询、多指标对比报告
- **Reflection 自检**：自动审查输出质量、修正幻觉和错误

## 快速开始

### 环境要求

- Python 3.11+
- [PostgreSQL](https://www.postgresql.org/) 16+ 已安装并运行
- pgvector 扩展
- LLM：任选其一
  - [Ollama](https://ollama.com/) 本地模型（推荐开发环境：`ollama pull qwen2.5:7b`）
  - [DeepSeek API Key](https://platform.deepseek.com/)（云端模型）

### 安装与启动

```bash
# 0. 创建数据库并启用 pgvector 扩展
psql -U postgres -c "CREATE DATABASE agentkb;"
psql -U postgres -d agentkb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 1. 配置 LLM —— 编辑 src/agentkb/config/config.yaml
#    默认使用 Ollama + qwen2.5:7b，切换 DeepSeek 只需改 provider 和 api_key

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
| Agent 编排 | LangGraph + 自研 Multi-Agent Orchestrator |
| LLM | Ollama 本地模型（qwen2.5:7b 默认）/ DeepSeek API（deepseek-chat） |
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
│   ├── orchestrator.py  #   编排器：按依赖拓扑执行 subtasks
│   ├── base.py          #   SpecialistAgent 基类
│   ├── registry.py      #   Agent 注册表（插件化）
│   ├── reflection.py    #   Reflection 自检模块
│   ├── knowledge_agent.py   # 知识检索 Agent
│   ├── content_creator.py   # 内容创作 Agent（规划→写作→优化）
│   ├── task_manager.py      # 任务管理 Agent
│   ├── learning_tutor.py    # 学习导师 Agent
│   └── social_writer.py     # 社媒内容 Agent
├── memory/              # 【V2 新增】增强记忆层
│   ├── working.py       #   工作记忆（会话上下文窗口 + 自动压缩）
│   └── long_term.py     #   长期记忆（向量化持久 + 语义检索）
├── observability/       # 【V2 新增】全链路可观测性
│   ├── tracer.py        #   Trace/Span 管理（OTel 兼容）
│   ├── exporters.py     #   多导出器（Console/File/LangFuse/LangSmith/Prometheus）
│   ├── metrics.py       #   指标收集器（Prometheus 格式）
│   └── middleware.py    #   FastAPI 中间件（自动注入 trace_id）
├── api/                 # FastAPI REST API
│   ├── routes.py        #   聊天/上传/会话/反馈/Trace/Metrics/Health
│   └── server.py        #   应用工厂 + 静态文件挂载
├── tools/               # 工具系统（插件化注册）
│   ├── knowledge_search.py  # 混合检索 + 重排序 + 缓存
│   ├── web_search.py        # DuckDuckGo + DDG Lite 回退
│   ├── web_browser.py       # 【V2 新增】网页浏览与正文提取
│   └── code_executor.py     # 【V2 新增】安全沙箱代码执行
├── knowledge/           # 知识库管道
│   ├── chunker.py       #   多策略分块 + 自动策略选择
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
├── storage/             # PostgreSQL + pgvector 持久化
└── utils/               # 日志、异常、基础 Trace
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

# 对比任务
curl -X POST http://127.0.0.1:8000/api/eval/compare \
  -H "Content-Type: application/json" \
  -d '{"baseline_job_id": "abc123", "current_job_id": "def456"}'

# === CLI 方式（兼容保留）===

# 生成银标测试集
PYTHONPATH="src" python -m agentkb.eval generate --sample-size 50

# 跑评估
PYTHONPATH="src" python -m agentkb.eval run

# 对比评估
PYTHONPATH="src" python -m agentkb.eval compare --baseline baseline.json --current after.json
```

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
  provider: ollama              # 或 deepseek
  model_name: qwen2.5:7b        # ollama 本地模型
  router_model_name: qwen2.5:7b # 路由用小模型
  generator_model_name: qwen2.5:7b
  base_url: http://localhost:11434
```

也可通过环境变量覆盖（如 `AGENTKB_LLM_PROVIDER=deepseek` 切换云端模型）。

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

- [ ] 知识图谱构建（实体关系抽取 + 图查询）
- [ ] 多模态支持（图片理解、PDF 深度解析、语音输入）
- [ ] Agent 对话式注册（自然语言创建自定义 Agent）
- [ ] 评估基线自动管理（CI/CD 评估门禁完善）
- [ ] MCP 生态深度集成

### V3.0（远期）

- [ ] 多用户 + 权限系统
- [ ] 云端同步 + 协作
- [ ] Electron 桌面客户端
- [ ] 企业知识库（多空间 + 团队共享）
- [ ] 主动 Agent（定时巡检 + 自动任务执行）
- [ ] 插件市场
