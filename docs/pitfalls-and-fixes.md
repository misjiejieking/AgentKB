# AgentKB 开发踩坑记录

## 1. StateGraph(dict) 导致消息覆盖而非追加

**现象**：LLM 调用工具后，第二轮 agent_node 的 `invoke_messages` 只剩 `[SystemMessage, ToolMessage]`，HumanMessage 和带 tool_calls 的 AIMessage 全部消失。

**根因**：`StateGraph(dict)` 使用 `dict.update()` 语义，节点 `return {"messages": [...]}` 时直接覆盖整个列表，不是追加。

```
agent_node → return {"messages": [AIMessage(tool_calls)]} 
  → state["messages"] = [AIMessage]  ← 之前的 [System, Human] 被覆盖

tools_node → return {"messages": [ToolMessage]}
  → state["messages"] = [ToolMessage]  ← AIMessage 也被覆盖
```

**修复**：改用 `StateGraph(MessagesState)`，内置 `add_messages` reducer 正确追加消息。
```python
from langgraph.graph import MessagesState
workflow = StateGraph(MessagesState)
```

---

## 2. Checkpointer 必须完整保留消息类型

**现象**：Checkpointer 往返序列化后，如果 `AIMessage.tool_calls` 丢失，后续
`ToolMessage` 会失去对应调用，OpenAI 兼容接口将拒绝请求。

**根因**：仅保存普通 JSON 文本无法完整恢复 LangChain 消息对象及其嵌套类型。

**修复**：使用 LangGraph SerializerProtocol 对完整 Checkpoint 做类型化序列化，
并持久化到 PostgreSQL。服务重启后可恢复状态，消息类型和 `tool_calls` 结构也不会丢失。
```python
from agentkb.storage.checkpointer import PostgresCheckpointSaver

checkpointer = PostgresCheckpointSaver(database)
```

---

## 3. LLM 模型切换导致 checkpointer 格式不兼容

**现象**：从 Ollama 切到 DeepSeek 后，旧 checkpointer 中的 tool_calls 格式与新模型不兼容，API 报 400。

**根因**：Ollama 的 tool_calls 缺少 `id` 字段，DeepSeek 要求 `id` + `type` + `function.arguments` 必须是 JSON 字符串。

**修复**：`thread_id` 同时加入 Provider、模型和图类型，不同运行协议的状态相互隔离。
```python
thread_id = f"{session_id}:{provider}:{model_name}:{graph_name}"
```

---

## 4. System Prompt 过宽导致所有问题都走知识库检索

**现象**：问"阿里巴巴什么时候成立"也触发 `search_knowledge_base`。

**根因**：Prompt 中"用户询问事实性问题时，优先使用 search_knowledge_base"太宽泛，LLM 把常识问题也归类为"事实性"。

**修复**：分三类明确定义——常识/通用知识直接回复、涉及用户文档才搜知识库、实时信息才联网搜索。

---

## 5. 中文 BM25（tsvector）分词失效

**现象**：`tsvector` 全文检索对中文永远返回 0 条结果。

**根因**：`to_tsvector('simple', content)` 使用空格分词，中文无空格，整个 chunk 被当作一个巨大 token，`tsquery` 永远匹配不到。

**修复**：写入前用 jieba 分词，空格拼接后再 `to_tsvector`；查询时同样 jieba 分词再用 `plainto_tsquery`。

---

## 6. PDF 文本提取带中文字间空格

**现象**：PDF 提取的文本如"员 工 手 册"，直接 embedding 严重影响检索质量。

**修复**：上传时自动清洗——去中文字间空格、合并多余空白、过滤 <10 字符的无意义 chunk。
```python
re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", text)
```

---

## 7. 搜索结果 score 全部为 0

**现象**：前端来源卡片显示 0%，`_truncate_context` 排序失效。

**根因**：`knowledge_search.py` 读 `r.get("rerank_score", 0)`，但 retriever 存的键是 `rrf_score`。键名不匹配，所有结果 score=0。

**修复**：统一使用 `rrf_score`。

---

## 8. SSE 流式中断不恢复

**现象**：用户刷新页面后 LLM 回复丢失，看不到吐字效果。

**修复**：
1. LLM 运行与 SSE 连接解耦——后台任务独立执行，每 N token 写 DB
2. Run/Event 持久化事件 + SSE `id:` 字段 + GET 断点续传端点
3. 前端加载历史时自动重连未完成的流

---

## 9. 本地 Reranker CPU 推理极慢

**现象**：BAAI/bge-reranker-v2-m3 在 CPU 上跑一条查询要 121 秒。

**根因**：2.1GB 模型全量在 CPU 上做交叉编码，每条查询需 20 次 model.predict。

**方案**：优先使用百炼 API（<1s），本地模型仅作备选。百炼免费额度耗尽时，退回到 RRF 分数排序（毫秒级）。
