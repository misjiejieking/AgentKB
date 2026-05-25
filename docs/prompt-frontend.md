# AgentKB 前端优化迭代 

> 本文档是给予 Codex（AI 开发助手）的前端优化指导提示词。
> 完整上下文见 `docs/optimization-analysis.md`，请先阅读该文档后再执行以下任务。
> 所有代码改动必须遵守仓库根目录 `AGENTS.md` 中的工程规范。

---

## 项目背景速览

你正在开发 AgentKB 的前端界面。这是一个**纯原生 HTML/CSS/JS** 的前后端分离项目。

| 文件 | 行数 | 职责 |
|------|------|------|
| `static/index.html` | 129 | 页面结构：侧边栏 + 顶栏 + 聊天区 + 输入区 + Toast 容器 |
| `static/js/app.js` | 739 | 全部前端逻辑：会话管理、流式聊天、文件上传、SSE 解析 |
| `static/css/style.css` | 847 | 完整设计系统：CSS 变量、侧边栏深色主题、聊天气泡、响应式 |

**当前已具备的能力**（无需重复实现）：
- 侧边栏：会话列表（切换/删除）、知识库文件列表、新建会话
- 流式聊天：SSE 逐 token 渲染 + throttle（50ms）、工具状态（🔍/✅）、Markdown 渲染（marked.js）
- 断点续传：SSE Last-Event-ID 机制，刷新页面后可续流
- 文件上传：点击上传 + 拖拽上传，Toast 通知
- 欢迎页：hint-chips 快速提问入口
- 响应式：768px 断点，侧边栏折叠

**后端 API 清单**（全部已实现，前端可直接调用）：

| 端点 | 方法 | 前端当前使用 |
|------|------|-------------|
| `/api/chat/stream` | POST | ✅ sendMessage() |
| `/api/chat/stream/{sid}` | GET | ✅ reconnectStream() |
| `/api/upload` | POST | ✅ uploadFiles() |
| `/api/knowledge/files` | GET | ✅ loadKnowledgeFiles() |
| `/api/knowledge/files/{id}` | DELETE | ❌ 未使用 |
| `/api/sessions` | GET | ✅ loadSessionList() |
| `/api/session/{id}` | GET | ✅ switchSession() |
| `/api/session/{id}/messages` | GET | ✅ loadMessageHistory() |
| `/api/session/{id}` | DELETE | ✅ deleteSession() |
| `/api/session/clear` | POST | ✅ clearSession() |

**SSE 事件类型**（`sendMessage()` 中已解析）：

| event.type | 含义 | 当前处理 |
|-----------|------|---------|
| `token` | LLM 流式 token | 累积 + throttle 渲染 |
| `tool_start` | 工具调用开始 | 显示 "正在调用 xxx……" |
| `tool_end` | 工具调用结束 | "已完成" + 可能包含 `sources` 字段 |
| `done` | 生成完成 | 无额外处理 |
| `error` | 错误 | 红色文字显示 |

---

## 迭代路线图

按三期执行，每期内部按编号顺序实施。每个任务完成后在浏览器中验证交互正常。

---

# 第一期（1~2 周）：信任感 + 反馈闭环

这一期的核心目标：让用户**信任答案**，并建立**反馈回路**。

---

## 任务 F-1：答案引用来源卡片

**目标文件**: `static/js/app.js`、`static/css/style.css`

**当前状态**：
- `tool_end` 事件中后端可携带 `sources` 数组（`graph.py` 已预留逻辑），但前端未解析渲染
- 检索返回的 `filename` 和 `score` 信息在 `tool_end` 的 output JSON 中，但 `sendMessage()` 未提取

**要做什么**：

### Step 1 — 解析 tool_end 中的 sources

在 `sendMessage()` 的 `case "tool_end"` 分支中，解析 `event.sources` 字段：

```javascript
case "tool_end":
  toolLines = toolLines.map((l) =>
    l.replace('class="spinner"', 'class="done-icon"').replace("正在调用", "已完成")
  );
  // 新增：提取检索来源
  if (event.name === "search_knowledge_base" && event.sources && event.sources.length > 0) {
    window._lastSources = event.sources;  // 暂存，等 done 后渲染
  }
  flushContent();
  break;
```

### Step 2 — 在 done 事件后渲染引用卡片

在 `case "done"` 分支中，当 `_lastSources` 有数据时，在答案末尾追加引用卡片：

```javascript
case "done":
  if (window._lastSources && window._lastSources.length > 0) {
    accumulated += renderSourceCard(window._lastSources);
    window._lastSources = null;
  }
  break;
```

`renderSourceCard(sources)` 函数生成 HTML：

```javascript
function renderSourceCard(sources) {
  const items = sources.map((s) => {
    const pct = Math.round((s.score || 0) * 100);
    const cls = pct >= 70 ? "high" : pct >= 40 ? "mid" : "low";
    const icon = pct < 40 ? " ⚠️" : "";
    return `<div class="source-item ${cls}">
      <span class="source-icon">📄</span>
      <span class="source-name">${escapeHtml(s.filename || "unknown")}</span>
      <span class="source-score">${pct}%${icon}</span>
    </div>`;
  }).join("");
  return `\n\n<div class="source-card">
    <div class="source-card-title">📎 引用来源</div>
    ${items}
  </div>`;
}
```

### Step 3 — 添加 CSS 样式

在 `static/css/style.css` 末尾添加：

```css
/* 引用来源卡片 */
.source-card {
  margin-top: 14px;
  padding: 12px 14px;
  background: #f8fafc;
  border: 1px solid var(--border);
  border-radius: var(--radius);
}
.source-card-title {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-secondary);
  margin-bottom: 8px;
}
.source-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 5px 0;
  font-size: 0.8rem;
}
.source-item + .source-item { border-top: 1px solid var(--border-light); }
.source-icon { flex-shrink: 0; font-size: 0.85rem; }
.source-name { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.source-score {
  font-weight: 600;
  flex-shrink: 0;
  font-size: 0.75rem;
  padding: 2px 6px;
  border-radius: 4px;
}
.source-item.high .source-score { color: #15803d; background: #f0fdf4; }
.source-item.mid  .source-score { color: #a16207; background: #fefce8; }
.source-item.low  .source-score { color: #b91c1c; background: #fef2f2; }
```

**验收标准**:
- 知识库问答后，答案下方出现 "📎 引用来源" 卡片
- 每个来源显示文件名 + 相关度百分比
- 高相关度绿色、中相关度黄色、低相关度红色 + ⚠️ 标记
- 直接对话（不调知识检索）时不显示引用卡片

---

## 任务 F-2：答案内联引用高亮

**目标文件**: `static/js/app.js`、`static/css/style.css`

**当前状态**：System prompt 要求 LLM 用 `[来源: 文件名]` 格式标注引用，但前端只做纯 Markdown 渲染，这些标记看起来像普通文本。

**要做什么**：

### Step 1 — 后处理 Markdown 输出中的引用标记

在 `finalizeAssistant` 之前，对 `accumulated` 做一次正则替换：

```javascript
function highlightCitations(html, sources) {
  // 将 [来源: xxx.md] 替换为带样式的引用标签
  // sources 数组用于查找对应分数做 tooltip
  const sourceMap = {};
  if (sources) {
    sources.forEach((s) => {
      sourceMap[s.filename] = Math.round((s.score || 0) * 100);
    });
  }
  return html.replace(/\[来源:\s*([^\]]+)\]/g, (match, filename) => {
    const fname = filename.trim();
    const score = sourceMap[fname];
    const title = score !== undefined ? `相关度: ${score}%` : fname;
    return `<span class="citation" title="${escapeHtml(title)}">📄 ${escapeHtml(fname)}</span>`;
  });
}
```

在 `sendMessage()` 的 `done` 分支中调用：
```javascript
case "done":
  if (window._lastSources && window._lastSources.length > 0) {
    // 先对内联引用做高亮
    let html = renderMarkdown(accumulated);
    html = highlightCitations(html, window._lastSources);
    assistantEl.innerHTML = html;
    // 再追加底部引用卡片
    assistantEl.innerHTML += renderSourceCardHtml(window._lastSources);
    window._lastSources = null;
  } else {
    assistantEl.innerHTML = renderMarkdown(accumulated);
  }
  finalizeAssistant(assistantEl);
  scrollToBottom();
  break;
```

注意：`done` 分支当前是空处理，token 最后一个 flush 之后不会再次渲染。需要在 stream 循环结束后（`reader` 完成时）替换当前逻辑，统一在 `done` 后做最终渲染 + 内联引用高亮。

### Step 2 — 添加 citation CSS

```css
.citation {
  color: #2563eb;
  cursor: help;
  border-bottom: 1px dashed #93c5fd;
  font-weight: 500;
  white-space: nowrap;
}
.citation:hover {
  background: #eff6ff;
  border-bottom-color: #2563eb;
}
```

**验收标准**:
- 答案中的 `[来源: xxx.md]` 被渲染为蓝色可交互标签
- 悬停时显示相关度 tooltip
- 非引用场景不影响正常 Markdown 渲染

---

## 任务 F-3：👍/👎 用户反馈

**目标文件**: `static/js/app.js`、`static/css/style.css`、`static/index.html`

**当前状态**：没有用户反馈机制。用户无法标记答案质量。

**要做什么**：

### Step 1 — 每条助手消息底部追加反馈按钮

修改消息渲染逻辑。在 `addAssistantPlaceholder()` 创建的 DOM 元素结构中增加反馈按钮区域。由于 `sendMessage` 中直接操作 `assistantEl.innerHTML`，改为在流式完成后，给 `assistantEl` 追加一个 footer：

```javascript
function addFeedbackButtons(messageEl, messageId) {
  const footer = document.createElement("div");
  footer.className = "message-feedback";
  footer.innerHTML = `
    <button class="feedback-btn up" data-action="up" title="有用">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 22V11M2 13v7a2 2 0 0 0 2 2h12.4a2 2 0 0 0 1.94-1.52l2.1-8.4A2 2 0 0 0 18.5 9.6H14l1-5a2 2 0 0 0-3.46-1.46L7 11"/></svg>
    </button>
    <button class="feedback-btn down" data-action="down" title="无用">
      <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M17 2v11m5-1v-7a2 2 0 0 0-2-2H7.6a2 2 0 0 0-1.94 1.52l-2.1 8.4A2 2 0 0 0 5.5 14.4H10l-1 5a2 2 0 0 0 3.46 1.46L17 13"/></svg>
    </button>
  `;
  messageEl.appendChild(footer);

  footer.querySelectorAll(".feedback-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const action = btn.dataset.action;
      // 禁用重复点击
      footer.querySelectorAll(".feedback-btn").forEach(b => b.disabled = true);

      if (action === "down") {
        const reason = await promptFeedbackReason();
        if (!reason) {
          footer.querySelectorAll(".feedback-btn").forEach(b => b.disabled = false);
          return;
        }
        await submitFeedback(messageId, "down", reason);
        showToast("感谢反馈，我们会持续改进", "info");
      } else {
        await submitFeedback(messageId, "up", "");
        showToast("感谢反馈！", "success");
      }

      // 高亮选中按钮
      btn.classList.add("active");
    });
  });
}
```

### Step 2 — 👎 原因弹窗

用简单的 DOM 弹窗（不用第三方库）：

```javascript
function promptFeedbackReason() {
  return new Promise((resolve) => {
    const overlay = document.createElement("div");
    overlay.className = "feedback-overlay";
    overlay.innerHTML = `
      <div class="feedback-dialog">
        <div class="feedback-dialog-title">请选择问题类型</div>
        <div class="feedback-reasons">
          <button data-reason="不相关">内容不相关</button>
          <button data-reason="不准确">信息不准确</button>
          <button data-reason="不完整">回答不完整</button>
          <button data-reason="其他">其他</button>
        </div>
        <button class="feedback-cancel">取消</button>
      </div>
    `;
    document.body.appendChild(overlay);

    overlay.querySelectorAll(".feedback-reasons button").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.body.removeChild(overlay);
        resolve(btn.dataset.reason);
      });
    });
    overlay.querySelector(".feedback-cancel").addEventListener("click", () => {
      document.body.removeChild(overlay);
      resolve(null);
    });
  });
}
```

### Step 3 — 提交反馈 API

```javascript
async function submitFeedback(messageId, rating, reason) {
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        rating: rating,
        reason: reason,
        query: "",  // 从最近的用户消息获取
      }),
    });
  } catch (err) {
    // 静默失败
  }
}
```

### Step 4 — 后端反馈 API

同步在 `src/agentkb/api/routes.py` 中新增 `/api/feedback` 端点，写入 `feedback` 表。

### Step 5 — CSS

```css
.message-feedback {
  display: flex;
  gap: 4px;
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid var(--border-light);
}
.feedback-btn {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  border: 1px solid var(--border);
  border-radius: 50%;
  background: var(--surface);
  color: var(--text-dim);
  cursor: pointer;
  transition: all var(--transition);
}
.feedback-btn:hover { border-color: var(--primary-light); color: var(--primary); }
.feedback-btn.active { background: var(--primary-bg); border-color: var(--primary); color: var(--primary); }
.feedback-btn:disabled { opacity: 0.4; cursor: not-allowed; }

.feedback-overlay {
  position: fixed; inset: 0; z-index: 2000;
  background: rgba(15,23,42,0.4);
  display: flex; align-items: center; justify-content: center;
}
.feedback-dialog {
  background: var(--surface);
  border-radius: var(--radius-lg);
  padding: 24px;
  box-shadow: var(--shadow-xl);
  min-width: 280px;
  text-align: center;
}
.feedback-dialog-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 14px; }
.feedback-reasons { display: flex; flex-direction: column; gap: 6px; margin-bottom: 12px; }
.feedback-reasons button {
  padding: 8px; border: 1px solid var(--border); border-radius: var(--radius-sm);
  background: var(--surface); cursor: pointer; font-family: inherit; font-size: 0.875rem;
}
.feedback-reasons button:hover { background: var(--border-light); }
.feedback-cancel {
  border: none; background: none; color: var(--text-dim); cursor: pointer; font-size: 0.8rem;
}
```

**验收标准**:
- 每条助手消息底部有 👍/👎 按钮
- 点击 👍 直接提交 + 按钮高亮
- 点击 👎 弹出原因选择弹窗 → 选择后提交
- 同一消息不能重复反馈

---

# 第二期（3~4 周）：交互体验升级

---

## 任务 F-4：知识库文件管理增强

**目标文件**: `static/js/app.js`、`static/css/style.css`

**当前状态**：侧边栏文件列表只展示文件名，无法删除，无可视化信息。

**要做什么**：

### Step 1 — 文件项增加删除按钮

修改 `loadKnowledgeFiles()` 中 `kbFileList` 的渲染：

```javascript
kbFileList.innerHTML = files.map((f) => {
  const name = escapeHtml(f.filename || "unknown");
  const chunks = f.chunk_count || 0;
  const size = formatFileSize(f.file_size || 0);
  const type = (f.file_type || "").toUpperCase();
  return `
    <div class="kb-file-item" title="${name} · ${chunks} 分块 · ${size}">
      <span class="file-type-badge">${type}</span>
      <span class="file-name-text">${name}</span>
      <span class="file-meta">${chunks}块</span>
      <button class="file-delete-btn" data-id="${f.id}" title="删除文件">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
      </button>
    </div>`;
}).join("");
```

绑定删除事件：
```javascript
kbFileList.querySelectorAll(".file-delete-btn").forEach((btn) => {
  btn.addEventListener("click", async (e) => {
    e.stopPropagation();
    const fileId = btn.dataset.id;
    if (confirm("确定要删除该文件及其知识索引吗？")) {
      await deleteKnowledgeFile(fileId);
    }
  });
});
```

### Step 2 — 删除文件函数

```javascript
async function deleteKnowledgeFile(fileId) {
  try {
    const resp = await fetch(`/api/knowledge/files/${fileId}`, { method: "DELETE" });
    if (resp.ok) {
      showToast("文件已删除", "success");
      loadKnowledgeFiles();
    } else {
      showToast("删除失败", "error");
    }
  } catch (err) {
    showToast("删除失败: " + err.message, "error");
  }
}
```

### Step 3 — 文件类型图标 + 辅助函数

```javascript
function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / 1048576).toFixed(1) + "MB";
}
```

### Step 4 — CSS

```css
.kb-file-item {
  display: flex; align-items: center; gap: 6px;
  padding: 5px 6px;
}
.file-type-badge {
  font-size: 0.6rem; font-weight: 700;
  padding: 1px 4px; border-radius: 3px;
  background: var(--sidebar-hover); color: var(--sidebar-text-dim);
  flex-shrink: 0;
}
.file-name-text { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.file-meta { font-size: 0.65rem; color: var(--sidebar-text-dim); flex-shrink: 0; }
.file-delete-btn {
  opacity: 0; flex-shrink: 0; width: 20px; height: 20px;
  display: flex; align-items: center; justify-content: center;
  border: none; border-radius: 3px; background: transparent;
  color: var(--sidebar-text-dim); cursor: pointer;
}
.kb-file-item:hover .file-delete-btn { opacity: 1; }
.file-delete-btn:hover { color: #f87171; background: rgba(239,68,68,0.15); }
```

**验收标准**:
- 文件列表显示：类型徽章 + 文件名 + 分块数
- 悬停时出现删除按钮
- 删除有确认提示

---

## 任务 F-5：工具执行详情展示

**目标文件**: `static/js/app.js`、`static/css/style.css`

**当前状态**：工具调用显示 "🔍 正在调用 search_knowledge_base……" → "✅ 已完成 search_knowledge_base"。用户看不到检索到了多少结果、重排效果如何。

**要做什么**：

### Step 1 — tool_end 中展示更多信息

在 `case "tool_end"` 中，解析 output 提取摘要信息：

```javascript
case "tool_end":
  // 替换状态图标
  toolLines = toolLines.map((l) =>
    l.replace('class="spinner"', 'class="done-icon"').replace("正在调用", "已完成")
  );
  // 提取工具结果摘要
  if (event.name === "search_knowledge_base") {
    try {
      const output = typeof event.output === "string"
        ? JSON.parse(event.output) : event.output;
      const total = output?.data?.total ?? 0;
      const topScore = output?.data?.results?.[0]?.score ?? 0;
      const topPct = Math.round(topScore * 100);
      if (total > 0) {
        toolLines.push(`📊 检索到 ${total} 条结果，最佳匹配 ${topPct}%`);
      } else {
        toolLines.push("📊 知识库中未找到相关内容");
      }
    } catch (e) {}
  }
  if (event.name === "search_web") {
    try {
      const output = typeof event.output === "string"
        ? JSON.parse(event.output) : event.output;
      const total = output?.data?.total ?? 0;
      if (total > 0) {
        toolLines.push(`🌐 搜索到 ${total} 条网页结果`);
      }
    } catch (e) {}
  }
  // 暂存 sources
  if (event.name === "search_knowledge_base" && event.sources && event.sources.length > 0) {
    window._lastSources = event.sources;
  }
  flushContent();
  break;
```

**验收标准**:
- 知识检索完成后显示 "检索到 X 条结果，最佳匹配 Y%"
- 无结果时显示 "知识库中未找到相关内容"
- 联网搜索完成后显示 "搜索到 X 条网页结果"

---

## 任务 F-6：空状态与加载态优化

**目标文件**: `static/js/app.js`、`static/css/style.css`、`static/index.html`

**当前状态**：空会话显示欢迎页（OK），但发送消息后等待首个 token 期间没有任何反馈（用户可能以为没反应）。

**要做什么**：

### Step 1 — 发送消息后立即显示思考状态

在 `sendMessage()` 中，创建 `assistantEl` 后立即显示一个思考动画：

```javascript
const assistantEl = addAssistantPlaceholder();
// 立即显示思考状态
assistantEl.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
scrollToBottom();
```

收到第一个 token 时清除（当前的 `flushContent` 会覆盖 innerHTML，无需特殊处理）。

### Step 2 — 思考动画 CSS

```css
.thinking-indicator {
  display: flex; gap: 5px; padding: 8px 0;
}
.thinking-indicator span {
  width: 7px; height: 7px;
  background: var(--text-dim); border-radius: 50%;
  animation: thinkingBounce 1.2s infinite;
}
.thinking-indicator span:nth-child(2) { animation-delay: 0.2s; }
.thinking-indicator span:nth-child(3) { animation-delay: 0.4s; }

@keyframes thinkingBounce {
  0%, 80%, 100% { transform: translateY(0); opacity: 0.3; }
  40% { transform: translateY(-6px); opacity: 1; }
}
```

### Step 3 — 知识库为空的欢迎引导

当 `loadKnowledgeFiles()` 返回空时，在文件列表区域添加上传引导：

当前已有 `<div class="kb-empty">暂无文件</div>`，可升级为可点击的上传入口：

```html
<div class="kb-empty">
  <span>暂无文件</span>
  <button class="kb-upload-hint" onclick="document.getElementById('upload-input').click()">
    + 上传文件
  </button>
</div>
```

**验收标准**:
- 发送消息后立即出现三个跳动圆点的思考动画
- 收到第一个 token 后思考动画消失
- 空文件列表有上传引导按钮

---

# 第三期（5~8 周）：高级产品化

---

## 任务 F-7：查询耗时分解面板

**目标文件**: `static/js/app.js`、`static/css/style.css`

**依赖**: 后端 B-M2（全链路 Trace）已实现，SSE 事件中附带 `trace` 字段

**要做什么**：

在每条最终答案下方添加可折叠的耗时分解面板：

```javascript
function renderTimingPanel(traceData) {
  if (!traceData) return "";
  const rows = [
    { label: "向量检索", ms: traceData.dense_search_ms },
    { label: "BM25 检索", ms: traceData.bm25_search_ms },
    { label: "RRF 融合", ms: traceData.rrf_ms },
    { label: "重排序", ms: traceData.rerank_ms },
    { label: "LLM 生成", ms: traceData.llm_gen_ms },
  ].filter((r) => r.ms !== undefined);

  if (rows.length === 0) return "";

  const total = rows.reduce((s, r) => s + r.ms, 0);
  const barHtml = rows.map((r) => {
    const pct = Math.round((r.ms / total) * 100);
    return `<div class="timing-row">
      <span class="timing-label">${r.label}</span>
      <span class="timing-bar-wrap"><span class="timing-bar" style="width:${pct}%"></span></span>
      <span class="timing-ms">${r.ms}ms</span>
    </div>`;
  }).join("");

  return `
    <details class="timing-panel">
      <summary>⏱ 本次查询耗时 ${total}ms</summary>
      ${barHtml}
    </details>`;
}
```

`<details>` 元素自带折叠行为，无需 JS。

**验收标准**:
- 默认收起，点击展开
- 每个环节显示耗时和占比条
- 无 trace 数据时不渲染

---

## 任务 F-8：知识库健康度仪表板

**目标文件**: 新建 `static/kb-dashboard.html`（或作为 `index.html` 的 Tab）、`static/js/dashboard.js`、`static/css/dashboard.css`

**依赖**: 后端新增 `GET /api/knowledge/health` 端点

**要做什么**：

1. 在顶栏或侧边栏增加一个 "📊 知识库" 入口，点击切换到仪表板视图
2. 仪表板包含：
   - **概览卡片**：文件数 / 分块总数 / 总字符数 / 覆盖主题数
   - **文件质量表**：文件名、分块数、质量评分（颜色编码）、状态标签
   - **知识盲区**：基于用户 👎 反馈中 "不相关" 统计出的高频未覆盖问题
3. 通过 `fetch("/api/knowledge/health")` 获取数据

由于仪表板是独立视图，建议使用 CSS 的显示/隐藏切换，而非多页面。

**验收标准**:
- 仪表板入口可见
- 概览数据正确
- 文件质量评分有参考价值

---

## 通用约束（所有任务必须遵守）

1. **不引入第三方框架**：保持原生 HTML/CSS/JS，不引入 React/Vue/jQuery
2. **保持现有代码结构**：函数命名、模块组织方式与现有 `app.js` 一致
3. **CSS 变量复用**：使用 `style.css` 中已定义的 CSS 自定义属性，不硬编码颜色
4. **动画克制**：transition 不超过 0.2s，不添加装饰性动效
5. **错误静默降级**：新功能失败时不影响核心聊天流程
6. **中文文案**：所有用户可见文案使用中文
7. **逐任务验证**：每个任务完成后在浏览器中手动验证交互正确
8. **不修改 marked.min.js**：这是第三方库

---

## 与后端的协作接口

以下是前端需要但后端可能尚未实现的接口，建议与后端同事对齐：

| 接口 | 方法 | 请求体 | 响应体 | 用途 |
|------|------|--------|--------|------|
| `/api/feedback` | POST | `{session_id, rating, reason, query}` | `{ok: true}` | F-3 用户反馈 |
| `/api/knowledge/health` | GET | - | `{total_files, total_chunks, files: [...], blind_spots: [...]}` | F-8 健康度仪表板 |

SSE `tool_end` 事件需包含 `sources` 字段（F-1 引用卡片依赖）：
```json
{
  "type": "tool_end",
  "name": "search_knowledge_base",
  "sources": [
    {"filename": "考勤制度.md", "score": 0.92},
    {"filename": "员工手册.md", "score": 0.78}
  ]
}
```
