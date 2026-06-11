# AgentKB 后端优化迭代 — Codex 开发提示词

> 历史执行提示词：其中的任务与目标文件可能已被当前实现替代，不应作为现状说明。
>
> 本文档是给予 Codex（AI 开发助手）的后端优化指导提示词。
> 完整上下文见 `docs/optimization-analysis.md`，请先阅读该文档后再执行以下任务。
> 所有代码改动必须遵守仓库根目录 `AGENTS.md` 中的工程规范。

---

## 项目背景速览

你正在开发一个基于 LangGraph 的个人知识助手（AgentKB）。技术栈：

- **编排**: LangGraph (ReAct loop: agent_node ↔ tools_node)
- **LLM**: DeepSeek API (默认 deepseek-chat)
- **向量库**: PostgreSQL + pgvector (HNSW 索引)
- **检索**: Dense(pgvector cosine) + BM25(tsvector) → RRF 融合 → Reranker 精排
- **工具**: search_knowledge_base, search_web
- **评估**: 检索指标 (Recall@K, MRR, NDCG@K) + 自动测试集生成

核心文件路径（均在 `src/agentkb/` 下）：

| 文件 | 职责 |
|------|------|
| `agent/graph.py` | LangGraph 图构建 + 流式封装 |
| `agent/nodes.py` | agent_node (LLM决策) + tools_node (工具执行) |
| `agent/prompts.py` | System Prompt + 降级话术 |
| `tools/knowledge_search.py` | 知识库检索工具 |
| `tools/web_search.py` | 联网搜索工具 |
| `knowledge/retriever.py` | 混合检索 (HybridRetriever) |
| `knowledge/reranker.py` | 精排 (LocalReranker / BailianReranker) |
| `knowledge/embedder.py` | BGE-M3 向量化 (EmbedderService) |
| `knowledge/chunker.py` | 多策略分块 (TextSplitter) |
| `config/config.yaml` | 全局配置 |
| `eval/evaluator.py` | 检索评估编排 |
| `eval/metrics.py` | Recall/MRR/NDCG 指标计算 |
| `eval/testset.py` | 测试集管理 + 自动生成 |

---

## 迭代路线图

按三期执行，每期内部按编号顺序实施。每个任务完成后运行 `python -m agentkb.eval` 确认检索指标无退化。

---

# 第一期（1~2 周）：体验与质量快速赢点

---

## 任务 B-Q1：Prompt 优化 — 工具选择 few-shot + 拒答强化

**目标文件**: `agent/prompts.py`

**要做什么**:

1. 在 `SYSTEM_PROMPT` 中增加 3 个 few-shot 示例，覆盖以下场景：
   - 用户问知识性问题 → 应调用 search_knowledge_base
   - 用户闲聊/打招呼 → 直接回复，不调工具
   - 用户问实时/新闻类问题 → 应调用 search_web

2. 增加拒答强约束，格式如下：
   ```
   ### 拒答规则（必须遵守）
   - 当 search_knowledge_base 返回空结果或所有结果的 relevance 为 false 时，
     你必须**仅**回复："知识库中未找到相关信息，您可以尝试联网搜索或上传相关文件。"
     禁止在此情况下编造任何内容。
   - 当 search_web 也返回空结果时，诚实告知用户，不要编造。
   ```

3. 每个工具的 description 增加一句负面示例："不要用这个工具处理……"

**验收标准**:
- 测试问 "你好" → Agent 不调用任何工具，直接回复
- 测试问 "今天天气怎么样" → Agent 调用 search_web
- 测试问知识库中不存在的内容 → Agent 回复拒答话术，不编造

---

## 任务 B-Q2：上下文窗口截断

**目标文件**: `tools/knowledge_search.py`

**要做什么**:

在 `_execute` 方法的第 3 步（格式化输出）之后，增加一个 `_truncate_context` 步骤：

1. 引入 token 计数工具（使用 `tiktoken` 或简单按字符数估算：中文约 1.5 字符/token）
2. 设定上下文预算 = `max_tokens` 的 60%（约为 2400 tokens），优先保留 rerank_score 高的结果
3. 每条结果的 `content`（parent_content）做 smart truncation：
   - 如果超过 `预算 / 结果数`，截断到该长度
   - 截断时保留完整句子（以句号/换行分隔）
4. 截断后追加 `...(内容已截断)` 标记

**验收标准**:
- 单次检索返回的 context 总 token 数不超过预算上限
- rerank_score 高的结果内容更完整
- 截断位置在句子边界，不会从词语中间断开

---

## 任务 B-Q3：Embedding GPU 自动检测 + FP16 推理

**目标文件**: `knowledge/embedder.py`, `config/config.yaml`

**要做什么**:

1. 在 `EmbedderService.__init__` 中增加设备自动检测逻辑：
   ```python
   def _auto_device() -> str:
       if torch.cuda.is_available():
           return "cuda"
       if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
           return "mps"
       return "cpu"
   ```
2. 当 `device` 为 `"auto"` 时使用自动检测结果
3. 当设备为 cuda 时调用 `self._model.half()` 启用 FP16 推理
4. `config.yaml` 中 `embedding.device` 默认值改为 `"auto"`

**注意**: 不要删除 `local_files_only=True`，保持离线优先。

**验收标准**:
- CPU 环境下行为不变
- CUDA 环境下显存占用降低约 50%，推理速度提升 2x
- 检索质量（Recall@5）无退化

---

## 任务 B-Q5：检索优雅降级链路

**目标文件**: `knowledge/retriever.py`, `tools/knowledge_search.py`

**要做什么**:

1. 在 `HybridRetriever.retrieve` 中实现三层降级：
   ```
   try:
       dense_results = self._db.search_dense(...)
   except Exception:
       dense_results = []  # dense 失败不崩溃

   try:
       bm25_results = self._db.search_bm25(...)
   except Exception:
       bm25_results = []  # bm25 失败不崩溃

   if dense_results and bm25_results:
       merged = self._rrf_fusion(dense_results, bm25_results, ...)
   elif dense_results:
       merged = dense_results  # 降级到 pure dense
   elif bm25_results:
       merged = bm25_results  # 降级到 pure bm25
   else:
       return []  # 全部失败，返回空
   ```

2. 在 `knowledge_search.py` 的 `_execute` 中增加降级感知：
   - 当 `candidates` 为空时，返回的 data 中增加 `degraded: true` 和 `reason: "检索无结果"`
   - 当只用了一种检索方式时，返回 `degraded: true` 和 `reason: "降级到 dense-only"` 或 `"降级到 bm25-only"`

3. LLM 调用超时处理：在 `agent/nodes.py` 的 `agent_node` 中给 `llm_with_tools.ainvoke` 增加超时控制，超时后返回友好降级消息。

**验收标准**:
- 停掉 pgvector 扩展后，dense search 抛出异常但 BM25 仍可用，检索不崩溃
- 降级信息正确传递到前端（通过 tool_result 中的 degraded 字段）

---

# 第二期（3~4 周）：Agent 能力增强 + 可观测性

---

## 任务 B-M1：Query Rewriting 查询重写节点

**目标文件**: 新建 `agent/query_rewriter.py`，修改 `agent/graph.py`、`agent/nodes.py`

**要做什么**:

1. 新建 `agent/query_rewriter.py`，实现 `rewrite_query` 函数：
   - 输入：当前 query + 最近 3 轮对话历史
   - 用 LLM 做指代消解 + 生成 BM25 关键词
   - 输出 dict：`{"rewritten": "...", "keywords": "..."}`
   - 使用轻量 prompt（约 200 tokens），避免增加过多延迟

2. 在 `AgentState` 中增加字段 `rewritten_query: str` 和 `search_keywords: str`

3. 修改 `agent/graph.py` 的图结构：
   ```
   agent (决策) → tools (执行) → agent   # 现有
   agent → rewrite → tools → agent       # 新增：工具调用前先重写
   ```
   具体做法：在 `tools_node` 中，当检测到 tool_call 是 search_knowledge_base 时，先调 rewrite 再传改写后的 query。

   更简单的做法（推荐）：直接在 `knowledge_search.py._execute` 开头增加轻量查询重写——不用改 graph 结构。

4. 在 `knowledge_search.py._execute` 中：
   - 用改写后的 query 做 dense search
   - 用提取的关键词做 BM25 search
   - 原始 query 保留作为 fallback

**验收标准**:
- 多轮对话场景：用户问 "那第二条呢" → 改写为具体实体相关 query
- Recall@5 多轮对话场景提升 >10%
- 单轮对话 Recall@5 无退化

---

## 任务 B-M2：全链路 Trace 集成

**目标文件**: 新建 `utils/tracer.py`，修改 `agent/graph.py`、`tools/knowledge_search.py`

**要做什么**:

1. 新建 `utils/tracer.py`，实现轻量级 trace 记录器（不引入外部依赖，用 loguru + JSON 格式输出到 `data/traces/`）：

   ```python
   class TraceContext:
       trace_id: str
       span_id: str
       parent_span_id: str | None

   def trace_span(name: str) -> TraceContext:  # context manager
       ...

   def log_span_event(name: str, data: dict) -> None:  # 记录结构化事件
       ...
   ```

2. 在以下关键节点埋入 trace：
   - `agent_node`: 记录 input_tokens, output_tokens, tool_calls_decision
   - `tools_node`: 记录每个 tool 的调用耗时
   - `knowledge_search._execute`: 记录 query, candidates_count, rerank_scores, final_count
   - `retriever.retrieve`: 记录 dense_count, bm25_count, fusion_method

3. 在 `graph.py` 的 `stream()` 方法中，每个 event yield 时附上 `trace_id`

4. 新增一个简单的 trace 查询 API (`GET /trace/{trace_id}`)，返回该次请求的完整链路

**验收标准**:
- 每次用户提问生成一个唯一的 trace_id
- trace 日志可读、可查询、可关联前后端
- 不引入新的第三方依赖（纯 loguru + JSON）

---

## 任务 B-M3：检索结果语义缓存

**目标文件**: 新建 `knowledge/cache.py`，修改 `tools/knowledge_search.py`

**要做什么**:

1. 新建 `knowledge/cache.py`，实现 `QueryCache`：

   ```python
   class QueryCache:
       def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.95):
           ...
       def get(self, query_embedding: list[float]) -> list[dict] | None:
           """返回缓存命中结果或 None"""
       def set(self, query_embedding: list[float], results: list[dict]) -> None:
           """存入缓存"""
       def invalidate(self) -> None:
           """知识库更新后清空全部缓存"""
   ```

   实现方式：用 `numpy` 计算余弦相似度，维护一个 LRU dict + embedding 列表。

2. 在 `knowledge_search._execute` 中：
   - 步骤 1 之前先查缓存
   - 命中则跳过检索+重排，直接返回缓存结果（标记 `cached: true`）
   - 未命中则在返回前写入缓存

3. 在文件上传/删除的 API 中调用 `cache.invalidate()`

**验收标准**:
- 同一 query 第二次检索命中缓存，延迟 <50ms
- 上传新文件后缓存自动失效
- 相似但不同的 query 不会误命中

---

## 任务 B-M5：生成质量评估

**目标文件**: 新建 `eval/generation_eval.py`，修改 `eval/evaluator.py`

**要做什么**:

1. 新建 `eval/generation_eval.py`，实现 `GenerationEval`：

   ```python
   @dataclass
   class GenerationMetrics:
       faithfulness: float  # 答案是否来自检索上下文 (0~1)
       answer_relevance: float  # 答案是否回答了问题 (0~1)
       context_relevance: float  # 检索上下文是否与问题相关 (0~1)

   def evaluate_generation(
       query: str,
       answer: str,
       contexts: list[str],
       llm_client,
   ) -> GenerationMetrics:
       ...
   ```

   每个指标用 LLM-as-judge 评估，给 1~5 分后归一化。

2. 在 `eval/evaluator.py` 的 `Evaluator` 中增加 `evaluate_full` 方法：
   - 先跑检索评估（现有逻辑）
   - 对每条 query，用 Agent 生成答案
   - 对每条答案跑生成质量评估
   - 输出综合报告

3. 新增 CLI 命令 `python -m agentkb.eval generation` 专门跑生成评估

**验收标准**:
- 能输出每条 query 的 Faithfulness 分数
- 低 faithfulness 的 case 能被标记出来供人工审查
- Judge LLM 使用独立的 prompt，不影响主链路

---

# 第三期（5~8 周）：架构升级

---

## 任务 B-L1：Multi-Agent 编排

**目标文件**: 新建 `agent/router.py`，重构 `agent/graph.py`、`agent/nodes.py`

**要做什么**:

1. 新建 `agent/router.py`，实现 `IntentRouter`：
   - 用轻量 prompt + 小模型做意图分类：`chat | knowledge_search | web_search | hybrid`
   - `chat` → 直接回复，不调工具
   - `knowledge_search` → 只调知识检索
   - `web_search` → 只调联网搜索
   - `hybrid` → 两者都调

2. 重构 `agent/graph.py` 的图结构为条件分支：
   ```
   user_input → router_node → chat_node (闲聊) → END
                            → search_node (检索) → gen_node (生成) → END
   ```

3. 支持 "Router 用小模型 + Generator 用大模型" 的配置：
   - `config.yaml` 增加 `llm.router_model_name` 和 `llm.generator_model_name`
   - `llm/factory.py` 增加 `get_router_llm()` 和 `get_generator_llm()`

**验收标准**:
- 闲聊场景延迟降低 >50%（跳过检索）
- 检索场景首 token 延迟无增加
- 工具选择准确率 >90%（人工抽样 50 条）

---

## 任务 B-L2：MCP 工具生态 MVP

**目标文件**: 新建 `tools/mcp_client.py`，修改 `tools/registry.py`

**要做什么**:

1. 新建 `tools/mcp_client.py`，实现 MCP Client：
   - 支持 stdio 传输（本地 MCP Server）
   - 启动时自动连接配置的 MCP Server
   - 将 MCP tools 自动注册到 ToolRegistry

2. `config.yaml` 增加 MCP 配置段：
   ```yaml
   mcp:
     servers:
       - name: filesystem
         command: npx
         args: ["-y", "@modelcontextprotocol/server-filesystem", "./data/mcp"]
     enabled: false
   ```

3. ToolRegistry 增加 `register_mcp_tools()` 方法，启动时自动发现并注册

**验收标准**:
- 配置一个 MCP Server 后，其提供的工具自动出现在 Agent 可用工具列表中
- MCP 工具执行失败不影响内置工具
- 不引入 MCP 时不增加启动开销

---

## 任务 B-L4：CI/CD 评估自动跑

**目标文件**: 新建 `.github/workflows/eval.yml`

**要做什么**:

1. 新建 `.github/workflows/eval.yml`：
   ```yaml
   name: Eval Regression Check
   on:
     pull_request:
       paths:
         - 'src/agentkb/knowledge/**'
         - 'src/agentkb/tools/knowledge_search.py'
   jobs:
     eval:
       runs-on: ubuntu-latest
       services:
         postgres:
           image: pgvector/pgvector:pg16
           ...
       steps:
         - uses: actions/checkout@v4
         - name: Setup Python & deps
         - name: Run eval
           run: python -m agentkb.eval run --gate-baseline data/eval/baseline.json
         - name: Check regression
           run: |
             # 任一质量规则不满足则失败
             ...
   ```

2. 在 `eval/cli.py` 增加 `--gate-baseline` 选项：
   - 加载基线 EvalResult
   - 运行当前评估
   - 生成 DiffReport
   - 任一基线规则失败时 exit code 非 0

**验收标准**:
- PR 修改检索相关代码时自动触发评估
- Recall 退化超过阈值时 CI 变红
- 评估运行时间 <5 分钟

---

## 通用约束（所有任务必须遵守）

1. **代码洁癖**: 不保留注释掉的代码、不引入无用抽象、不写 "以后可能用到" 的代码
2. **注释**: 使用中文，只解释 WHY 不解释 WHAT，一行以内
3. **兼容性**: 不要写 fallback/兼容代码，选择当前最优方案即可
4. **错误处理**: 不能吞错，但工具内部必须 catch 并返回友好结果
5. **每个任务完成后**: 运行 `python -m agentkb.eval` 确认无退化
6. **文件变更后**: 删除不再使用的 import
