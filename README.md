# AgentKB

本地优先的个人知识助手（Personal Knowledge Agent），基于 LangGraph 构建。

## 功能

- **智能对话**：自然语言提问，流式 Markdown 回复
- **知识库**：上传 .md / .txt 文件，语义检索
- **联网搜索**：DuckDuckGo 搜索作为补充
- **本地运行**：无需 Docker，一键启动，数据不上云

## 快速开始

### 环境要求

- Python 3.11+
- [Ollama](https://ollama.com/) 已安装并运行

### 安装与启动

```bash
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

浏览器访问 http://127.0.0.1:7860

也可使用脚本启动：`scripts/start.bat`（Windows）或 `scripts/start.sh`（Unix）。

## 技术栈

| 模块 | 技术 |
|------|------|
| Agent 编排 | LangGraph |
| LLM | Ollama（qwen2.5:7b） |
| 向量存储 | Qdrant（本地模式） |
| 向量模型 | BGE-M3 |
| 数据库 | SQLite（WAL） |
| UI | Gradio |
| 搜索 | DuckDuckGo |

## 项目结构

```
src/agentkb/
├── main.py          # 入口
├── config/          # 配置系统
├── agent/           # LangGraph 状态机
├── llm/             # LLM Provider
├── tools/           # 工具系统
├── knowledge/       # 知识库管道
├── session/         # 会话管理
├── storage/         # SQLite 持久化
├── ui/              # Gradio 界面
└── utils/           # 日志与异常
```

## 配置

编辑 `src/agentkb/config/config.yaml` 修改配置，或通过环境变量覆盖（如 `AGENTKB_LLM_MODEL_NAME=qwen2.5:14b`）。
