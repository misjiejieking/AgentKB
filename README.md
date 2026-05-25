# AgentKB

本地优先的个人知识助手（Personal Knowledge Agent），基于 LangGraph 构建。

## 功能

- **智能对话**：自然语言提问，流式 Markdown 回复（SSE + 断点续传）
- **知识库**：上传 .md / .txt / .pdf / .docx / .csv / .json，混合检索（pgvector dense + jieba BM25 + RRF 融合）+ 重排序
- **多策略分块**：滑动窗口 / 语义分块 / 父子分块，自动根据文档特征选择
- **文本清洗**：自动去除 PDF 中文字间空格、合并空白、过滤噪音内容
- **联网搜索**：DuckDuckGo（含 DDG Lite 回退）
- **多会话**：完整的会话生命周期管理，刷新/断线不丢回复
- **评估框架**：自动化测试集生成 + Recall/Precision/MRR/NDCG 指标 + 对比报告
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
├── agent/               # LangGraph 状态机 + 查询重写
├── api/                 # FastAPI REST API（SSE 流式 + 断点续传）
├── llm/                 # LLM Provider
├── tools/               # 工具系统（知识检索 / 联网搜索）
├── knowledge/           # 知识库管道（加载→清洗→分块→嵌入→混合检索→重排序→缓存）
├── eval/                # 评估框架（测试集生成 / 指标计算 / 对比报告）
├── session/             # 会话管理
├── storage/             # PostgreSQL + pgvector 持久化
└── utils/               # 日志与异常
```

## 评估

```bash
# 生成银标测试集（从知识库采样 → LLM 自动生成问题）
PYTHONPATH="src" python -m agentkb.eval generate --sample-size 50

# 跑评估（输出带中文注释的 Markdown 报告）
PYTHONPATH="src" python -m agentkb.eval run

# 对比两次结果
PYTHONPATH="src" python -m agentkb.eval run --output baseline.json
# 修改配置/参数后...
PYTHONPATH="src" python -m agentkb.eval run --output after.json
PYTHONPATH="src" python -m agentkb.eval compare --baseline baseline.json --current after.json

# 从 JSON 重新生成报告
PYTHONPATH="src" python -m agentkb.eval report --input eval_result.json --format md
```

评估指标：Recall@K（召回率）、Precision@K（精确率）、MRR（第一个正确答案排名）、NDCG@K（排序质量）。

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
