# AgentKB

本地优先的个人知识助手（Personal Knowledge Agent），基于 LangGraph 构建。

## 功能

- **智能对话**：自然语言提问，流式 Markdown 回复（SSE）
- **知识库**：上传 .md / .txt / .pdf / .docx / .csv / .json 文件，混合检索（稠密向量 + BM25） + 重排序
- **联网搜索**：DuckDuckGo 搜索作为补充
- **多会话**：完整的会话生命周期管理
- **本地运行**：无需 Docker，一键启动，数据不上云

## 快速开始

### 环境要求

- Python 3.11+
- [Ollama](https://ollama.com/) 已安装并运行
- [PostgreSQL](https://www.postgresql.org/) 16+ 已安装并运行
- pgvector 扩展

### 安装与启动

```bash
# 0. 创建数据库并启用 pgvector 扩展
psql -U postgres -c "CREATE DATABASE agentkb;"
psql -U postgres -d agentkb -c "CREATE EXTENSION IF NOT EXISTS vector;"

# 1. 下载模型
ollama pull qwen2.5:7b

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动（Windows）
set PYTHONPATH=src;%PYTHONPATH%
python -m agentkb.main

# 3. 启动（macOS / Linux）
export PYTHONPATH="src:$PYTHONPATH"
python -m agentkb.main
```

浏览器访问 http://127.0.0.1:8000

也可使用脚本启动：`scripts/start.bat`（Windows）或 `scripts/start.sh`（Unix）。

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| LLM | Ollama（qwen2.5:7b） |
| 向量存储 | pgvector（PostgreSQL） |
| 向量模型 | BGE-M3 |
| 数据库 | PostgreSQL |
| Web 框架 | FastAPI + uvicorn |
| 搜索 | DuckDuckGo |
| 混合检索 | 稠密向量 + BM25（RRF 融合） |
| 重排序 | BGE-Reranker |

## 项目结构

```
src/agentkb/
├── main.py              # 入口
├── config/              # 配置系统
├── agent/               # LangGraph 状态机
├── api/                 # FastAPI REST API（SSE 流式传输）
├── llm/                 # LLM Provider
├── tools/               # 工具系统
├── knowledge/           # 知识库管道（加载→分块→嵌入→检索→重排序）
├── session/             # 会话管理
├── storage/             # PostgreSQL + pgvector 持久化
├── ui/                  # 前端界面
└── utils/               # 日志与异常
```

## 配置

编辑 `src/agentkb/config/config.yaml` 修改配置，或通过环境变量覆盖（如 `AGENTKB_LLM_MODEL_NAME=qwen2.5:14b`）。

### PostgreSQL 配置

默认连接：`postgres:postgres@127.0.0.1:5432/agentkb`

可通过环境变量覆盖：
```bash
set AGENTKB_POSTGRESQL_HOST=127.0.0.1
set AGENTKB_POSTGRESQL_DBNAME=agentkb
set AGENTKB_POSTGRESQL_USER=postgres
set AGENTKB_POSTGRESQL_PASSWORD=postgres
```
