/* global marked */
// ── AgentKB 前端逻辑 ──────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════
//  状态
// ══════════════════════════════════════════════════════════════

let sessionId = localStorage.getItem("agentkb_session_id") || "";
let isStreaming = false;
let sidebarVisible = true;
let lastEventId = 0;  // SSE 断点续传：当前会话最后收到的 event id

// ══════════════════════════════════════════════════════════════
//  DOM 引用
// ══════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const chatContainer = $("#chat");
const msgInput = $("#msg-input");
const sendBtn = $("#send-btn");
const uploadInput = $("#upload-input");
const sessionList = $("#session-list");
const kbFileList = $("#kb-file-list");
const kbCount = $("#kb-count");
const currentSessionTitle = $("#current-session-title");
const welcomeScreen = $("#welcome-screen");
const sidebar = $("#sidebar");
const toastContainer = $("#toast-container");

// ══════════════════════════════════════════════════════════════
//  工具函数
// ══════════════════════════════════════════════════════════════

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function renderMarkdown(text) {
  if (typeof marked !== "undefined" && marked.parse) {
    return marked.parse(text);
  }
  return escapeHtml(text).replace(/\n/g, "<br>");
}

function scrollToBottom() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

function showToast(message, type = "info", duration = 3500) {
  const el = document.createElement("div");
  el.className = `toast ${type}`;
  el.textContent = message;
  toastContainer.appendChild(el);
  setTimeout(() => {
    el.classList.add("removing");
    setTimeout(() => el.remove(), 300);
  }, duration);
}

function formatDate(isoStr) {
  if (!isoStr) return "";
  const d = new Date(isoStr);
  const now = new Date();
  const diffMs = now - d;
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffDays === 0) {
    return d.toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" });
  } else if (diffDays === 1) {
    return "昨天";
  } else if (diffDays < 7) {
    return `${diffDays} 天前`;
  } else {
    return d.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
  }
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + "B";
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + "KB";
  return (bytes / 1048576).toFixed(1) + "MB";
}

function highlightCitations(html, sources) {
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

function renderSourceCard(sources) {
  // 按文件名去重，取最高分
  const seen = {};
  sources.forEach((s) => {
    const fname = s.filename || "unknown";
    if (!seen[fname] || seen[fname] < s.score) {
      seen[fname] = s.score;
    }
  });
  const unique = [];
  for (const [fname, score] of Object.entries(seen)) {
    unique.push({ filename: fname, score: score });
  }
  unique.sort((a, b) => b.score - a.score);

  const items = unique.map((s) => {
    const pct = Math.round((s.score || 0) * 100);
    const cls = pct >= 70 ? "high" : pct >= 40 ? "mid" : "low";
    const icon = pct < 40 ? " ⚠️" : "";
    return `<div class="source-item ${cls}">
      <span class="source-icon">📄</span>
      <span class="source-name">${escapeHtml(s.filename)}</span>
      <span class="source-score">${pct}%${icon}</span>
    </div>`;
  }).join("");
  return `<div class="source-card">
    <div class="source-card-title">📎 引用来源</div>
    ${items}
  </div>`;
}

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

      btn.classList.add("active");
    });
  });
}

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

async function submitFeedback(messageId, rating, reason) {
  try {
    await fetch("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message_id: messageId || "",
        rating: rating,
        reason: reason,
        query: "",
      }),
    });
  } catch (err) {
    // 静默失败
  }
}

// 更新顶部标题
function setTitle(title) {
  currentSessionTitle.textContent = title || "New Chat";
}

// ══════════════════════════════════════════════════════════════
//  会话列表
// ══════════════════════════════════════════════════════════════

async function loadSessionList() {
  try {
    const resp = await fetch("/api/sessions");
    const data = await resp.json();
    renderSessionList(data.sessions || []);
  } catch (err) {
    // 静默处理
  }
}

function renderSessionList(sessions) {
  // 过滤掉无消息的空会话
  const activeSessions = sessions.filter((s) => (s.message_count || 0) > 0);

  if (activeSessions.length === 0) {
    sessionList.innerHTML = '<div class="session-empty">暂无会话</div>';
    return;
  }

  sessionList.innerHTML = activeSessions
    .map((s) => {
      const isActive = s.id === sessionId;
      const title = escapeHtml(s.title || "New Chat");
      const time = formatDate(s.updated_at);
      const count = s.message_count || 0;
      return `
        <div class="session-item${isActive ? " active" : ""}" data-id="${s.id}">
          <span class="session-title" title="${title}">${title}</span>
          <span style="font-size:0.7rem;color:var(--sidebar-text-dim);flex-shrink:0">${count}</span>
          <button class="session-delete" data-id="${s.id}" title="删除会话">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
              <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            </svg>
          </button>
        </div>`;
    })
    .join("");

  // 绑定点击切换
  sessionList.querySelectorAll(".session-item").forEach((el) => {
    el.addEventListener("click", (e) => {
      if (e.target.closest(".session-delete")) return;
      const sid = el.dataset.id;
      if (sid && sid !== sessionId) {
        switchSession(sid);
      }
    });
  });

  // 绑定删除
  sessionList.querySelectorAll(".session-delete").forEach((btn) => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const sid = btn.dataset.id;
      await deleteSession(sid);
    });
  });
}

async function switchSession(sid) {
  try {
    const resp = await fetch(`/api/session/${sid}`);
    const data = await resp.json();
    sessionId = data.session_id;
    localStorage.setItem("agentkb_session_id", sessionId);
    setTitle(data.title);

    // 加载消息历史
    if (data.message_count > 0) {
      hideWelcome();
      loadMessageHistory(data.session_id);
    } else {
      chatContainer.innerHTML = "";
      chatContainer.appendChild(welcomeScreen);
      welcomeScreen.style.display = "";
    }

    loadSessionList();
  } catch (err) {
    showToast("切换会话失败", "error");
  }
}

async function loadMessageHistory(sid) {
  try {
    const resp = await fetch(`/api/session/${sid}`);
    const data = await resp.json();
    // 需要加一个加载消息的端点，或者扩展现有端点
    // 这里我们使用 loadMessages 直接在后端获取
    const msgsResp = await fetch(`/api/session/${sid}/messages`);
    const msgsData = await msgsResp.json();
    renderHistoryMessages(msgsData.messages || []);
  } catch (err) {
    // 降级：清空聊天区
    chatContainer.innerHTML = "";
    chatContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
  }
}

function renderHistoryMessages(messages) {
  hideWelcome();

  // 移除之前的消息
  chatContainer.querySelectorAll(".message").forEach((el) => el.remove());

  let lastAiMsg = null;
  messages.forEach((m, idx) => {
    if (m.role === "human") {
      addUserMessage(m.content);
    } else if (m.role === "ai") {
      const isLast = idx === messages.length - 1;
      if (isLast) {
        lastAiMsg = m;
        // 最后一条 AI 消息用 placeholder 渲染，方便重连时填入内容
        const el = addAssistantPlaceholder();
        if (m.content) {
          el.innerHTML = renderMarkdown(m.content);
        }
      } else {
        addAssistantMessage(m.content);
      }
    }
  });

  scrollToBottom();

  // 最后一条是 AI 消息 → 尝试重连（任务在跑则续流，已跑完则秒回 done）
  if (lastAiMsg) {
    setTimeout(() => reconnectStream(sessionId), 300);
  }
}

async function reconnectStream(sid) {
  // SSE Last-Event-ID 断点续传
  const assistantEl = document.getElementById("current-assistant");
  if (!assistantEl) return;
  if (isStreaming) return;

  isStreaming = true;
  sendBtn.disabled = true;

  // 保留已有内容，后续 token 往后追加
  let accumulated = assistantEl.textContent.trim() || "";
  let toolLines = [];
  let throttleTimer = null;
  let finalized = false;

  function flushContent() {
    let display = accumulated;
    if (toolLines.length > 0) {
      display += '\n\n<div class="tool-status">' + toolLines.join("<br>") + "</div>";
    }
    assistantEl.innerHTML = renderMarkdown(display);
    scrollToBottom();
  }

  try {
    const response = await fetch(`/api/chat/stream/${sid}`, {
      headers: { "Last-Event-ID": String(lastEventId) },
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("id: ")) {
          lastEventId = parseInt(line.slice(4)) || lastEventId;
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case "token":
              accumulated += event.content;
              if (!throttleTimer) {
                throttleTimer = setTimeout(() => { flushContent(); throttleTimer = null; }, 50);
              }
              break;
            case "tool_start":
              toolLines.push('<span class="spinner"></span> 正在调用 <b>' + escapeHtml(event.name) + "</b>……");
              flushContent();
              break;
            case "tool_end":
              toolLines = toolLines.map((l) =>
                l.replace('class="spinner"', 'class="done-icon"').replace("正在调用", "已完成")
              );
              if (event.name === "search_knowledge_base") {
                try {
                  const output = typeof event.output === "string"
                    ? JSON.parse(event.output) : event.output;
                  const results = output?.data?.results || [];
                  if (results.length > 0) {
                    window._lastSources = results.map((r) => ({
                      filename: r.filename || "unknown",
                      score: r.score || 0,
                    }));
                  }
                } catch (e) {}
              }
              flushContent();
              break;
            case "done":
              if (window._lastSources && window._lastSources.length > 0) {
                let html = renderMarkdown(accumulated);
                html = highlightCitations(html, window._lastSources);
                assistantEl.innerHTML = html;
                assistantEl.innerHTML += renderSourceCard(window._lastSources);
                window._lastSources = null;
              }
              finalized = true;
              break;
            case "error":
              accumulated += '\n\n<span style="color:var(--danger)">' + escapeHtml(event.message || "未知错误") + "</span>";
              flushContent();
              break;
          }
        } catch (e) {}
      }
    }

    if (throttleTimer) { clearTimeout(throttleTimer); throttleTimer = null; }
    if (!finalized) {
      assistantEl.innerHTML = renderMarkdown(accumulated);
    }
    finalizeAssistant(assistantEl);
    scrollToBottom();
  } catch (err) {
    // 重连失败（生成已结束），从 DB 加载完整内容兜底
    try {
      const resp = await fetch(`/api/session/${sid}/messages`);
      const data = await resp.json();
      const last = (data.messages || []).slice(-1)[0];
      if (last && last.role === "ai" && last.content) {
        assistantEl.innerHTML = renderMarkdown(last.content);
        finalizeAssistant(assistantEl);
        scrollToBottom();
      }
    } catch (e) {}
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
  }
}

function createNewSession() {
  if (isStreaming) return;
  // 仅在本地生成 ID，后端在首条消息发送时才 ensure_session
  sessionId = crypto.randomUUID().slice(0, 12);
  localStorage.setItem("agentkb_session_id", sessionId);
  setTitle("New Chat");
  chatContainer.innerHTML = "";
  chatContainer.appendChild(welcomeScreen);
  welcomeScreen.style.display = "";
  msgInput.value = "";
  msgInput.style.height = "auto";
  msgInput.focus();
}

async function deleteSession(sid) {
  try {
    await fetch(`/api/session/${sid}`, { method: "DELETE" });
    if (sid === sessionId) {
      // 删除的是当前会话，创建新会话
      await createNewSession();
    } else {
      await loadSessionList();
    }
    showToast("会话已删除", "info");
  } catch (err) {
    showToast("删除失败", "error");
  }
}

// ══════════════════════════════════════════════════════════════
//  知识库文件列表
// ══════════════════════════════════════════════════════════════

async function loadKnowledgeFiles() {
  try {
    const resp = await fetch("/api/knowledge/files");
    const data = await resp.json();
    const files = data.files || [];
    kbCount.textContent = files.length;

    if (files.length === 0) {
      kbFileList.innerHTML = `<div class="kb-empty">
        <span>暂无文件</span>
        <button class="kb-upload-hint" onclick="document.getElementById('upload-input').click()">
          + 上传文件
        </button>
      </div>`;
      return;
    }

    kbFileList.innerHTML = files
      .map((f) => {
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
      })
      .join("");

    // 绑定删除事件
    kbFileList.querySelectorAll(".file-delete-btn").forEach((btn) => {
      btn.addEventListener("click", async (e) => {
        e.stopPropagation();
        const fileId = btn.dataset.id;
        if (confirm("确定要删除该文件及其知识索引吗？")) {
          await deleteKnowledgeFile(fileId);
        }
      });
    });
  } catch (err) {
    // 静默
  }
}

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

// ══════════════════════════════════════════════════════════════
//  消息渲染
// ══════════════════════════════════════════════════════════════

function hideWelcome() {
  if (welcomeScreen && welcomeScreen.style.display !== "none") {
    welcomeScreen.style.display = "none";
  }
}

function addUserMessage(text) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "message user";
  el.textContent = text;
  chatContainer.appendChild(el);
  scrollToBottom();
}

function addAssistantPlaceholder() {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "message assistant";
  el.id = "current-assistant";
  chatContainer.appendChild(el);
  return el;
}

function addAssistantMessage(text) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "message assistant";
  el.innerHTML = renderMarkdown(text);
  chatContainer.appendChild(el);
}

function finalizeAssistant(el) {
  if (el) el.removeAttribute("id");
}

function updateSessionTitleInSidebar() {
  // 更新侧边栏中当前会话的标题显示
  loadSessionList();
}

// ══════════════════════════════════════════════════════════════
//  流式聊天
// ══════════════════════════════════════════════════════════════

async function sendMessage(text) {
  if (isStreaming || !text.trim()) return;
  isStreaming = true;
  sendBtn.disabled = true;

  addUserMessage(text);
  const assistantEl = addAssistantPlaceholder();
  // 思考动画
  assistantEl.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
  scrollToBottom();

  let accumulated = "";
  let toolLines = [];
  let throttleTimer = null;
  let finalized = false;
  window._lastSources = null;
  window._lastTrace = null;
  window._lastMsgId = null;

  function flushContent() {
    let display = accumulated;
    if (toolLines.length > 0) {
      display += '\n\n<div class="tool-status">' + toolLines.join("<br>") + "</div>";
    }
    assistantEl.innerHTML = renderMarkdown(display);
    scrollToBottom();
  }

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (line.startsWith("id: ")) {
          lastEventId = parseInt(line.slice(4)) || lastEventId;
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case "message_id":
              window._lastMsgId = event.message_id;
              break;

            case "token":
              accumulated += event.content;
              if (!throttleTimer) {
                throttleTimer = setTimeout(() => {
                  flushContent();
                  throttleTimer = null;
                }, 50);
              }
              break;

            case "tool_start":
              toolLines.push(
                '<span class="spinner"></span> 正在调用 <b>' +
                  escapeHtml(event.name) +
                  "</b>……"
              );
              flushContent();
              break;

            case "tool_end":
              toolLines = toolLines.map((l) =>
                l.replace('class="spinner"', 'class="done-icon"').replace(
                  "正在调用",
                  "已完成"
                )
              );
              // 提取检索来源和统计信息
              if (event.name === "search_knowledge_base") {
                try {
                  const output = typeof event.output === "string"
                    ? JSON.parse(event.output) : event.output;
                  const results = output?.data?.results || [];
                  if (results.length > 0) {
                    window._lastSources = results.map((r) => ({
                      filename: r.filename || "unknown",
                      score: r.score || 0,
                    }));
                    const total = output.data.total || results.length;
                    const topPct = Math.round((results[0].score || 0) * 100);
                    toolLines.push(`📊 检索到 ${total} 条结果，最佳匹配 ${topPct}%`);
                  } else {
                    toolLines.push("📊 知识库中未找到相关内容");
                  }
                } catch (e) {}
              } else if (event.name === "search_web") {
                try {
                  const output = typeof event.output === "string"
                    ? JSON.parse(event.output) : event.output;
                  const total = output?.data?.total ?? 0;
                  if (total > 0) {
                    toolLines.push(`🌐 搜索到 ${total} 条网页结果`);
                  }
                } catch (e) {}
              }
              // 提取 trace 数据
              if (event.trace) {
                window._lastTrace = event.trace;
              }
              flushContent();
              break;

            case "done":
              if (throttleTimer) {
                clearTimeout(throttleTimer);
                throttleTimer = null;
              }
              if (window._lastSources && window._lastSources.length > 0) {
                let html = renderMarkdown(accumulated);
                html = highlightCitations(html, window._lastSources);
                assistantEl.innerHTML = html;
                assistantEl.innerHTML += renderSourceCard(window._lastSources);
              } else {
                assistantEl.innerHTML = renderMarkdown(accumulated);
              }
              if (window._lastTrace) {
                assistantEl.innerHTML += renderTimingPanel(window._lastTrace);
              }
              addFeedbackButtons(assistantEl, window._lastMsgId);
              finalizeAssistant(assistantEl);
              scrollToBottom();
              finalized = true;
              break;

            case "error":
              accumulated +=
                '\n\n<span style="color:var(--danger)">' +
                escapeHtml(event.message || "未知错误") +
                "</span>";
              flushContent();
              break;
          }
        } catch (e) {
          // 忽略 JSON 解析错误
        }
      }
    }

    if (!finalized) {
      if (throttleTimer) {
        clearTimeout(throttleTimer);
        throttleTimer = null;
      }
      assistantEl.innerHTML = renderMarkdown(accumulated);
      addFeedbackButtons(assistantEl, window._lastMsgId);
      finalizeAssistant(assistantEl);
      scrollToBottom();
    }

    // 更新侧边栏会话列表（标题可能变化）
    loadSessionList();
  } catch (err) {
    assistantEl.innerHTML = renderMarkdown(
      accumulated +
        '\n\n<span style="color:var(--danger)">请求失败: ' +
        escapeHtml(err.message) +
        "</span>"
    );
    addFeedbackButtons(assistantEl, window._lastMsgId);
    finalizeAssistant(assistantEl);
  } finally {
    isStreaming = false;
    sendBtn.disabled = false;
    msgInput.focus();
  }
}

// ══════════════════════════════════════════════════════════════
//  文件上传
// ══════════════════════════════════════════════════════════════

async function uploadFiles(files) {
  if (files.length === 0) return;

  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }

  showToast("正在上传并处理文件……", "info", 2000);

  try {
    const resp = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });
    const data = await resp.json();

    const okCount = data.results.filter((r) => r.status === "ok").length;
    const errCount = data.results.filter((r) => r.status === "error").length;

    if (errCount === 0) {
      showToast(`✅ 成功上传 ${okCount} 个文件`, "success");
    } else if (okCount === 0) {
      showToast("❌ 上传失败", "error");
    } else {
      showToast(`上传完成：${okCount} 成功，${errCount} 失败`, "info");
    }

    loadKnowledgeFiles();
  } catch (err) {
    showToast("上传失败: " + err.message, "error");
  }
}

// ══════════════════════════════════════════════════════════════
//  清空会话
// ══════════════════════════════════════════════════════════════

async function clearSession() {
  if (isStreaming) return;
  try {
    const resp = await fetch("/api/session/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await resp.json();
    sessionId = data.session_id;
    localStorage.setItem("agentkb_session_id", sessionId);
    setTitle(data.title);
    chatContainer.innerHTML = "";
    chatContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
    msgInput.value = "";
    loadSessionList();
    showToast("会话已清空", "info");
  } catch (err) {
    // 本地清空
    chatContainer.innerHTML = "";
    chatContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
  }
}

// ══════════════════════════════════════════════════════════════
//  侧边栏切换
// ══════════════════════════════════════════════════════════════

function toggleSidebar() {
  sidebarVisible = !sidebarVisible;
  if (sidebarVisible) {
    sidebar.classList.remove("collapsed");
  } else {
    sidebar.classList.add("collapsed");
  }
  localStorage.setItem("agentkb_sidebar_visible", sidebarVisible);
}

// ══════════════════════════════════════════════════════════════
//  初始化和事件绑定
// ══════════════════════════════════════════════════════════════

async function init() {
  // 恢复侧边栏状态
  const savedVisibility = localStorage.getItem("agentkb_sidebar_visible");
  if (savedVisibility === "false") {
    sidebarVisible = false;
    sidebar.classList.add("collapsed");
  }

  // 加载知识库文件
  loadKnowledgeFiles();

  // 初始化会话
  let hasHistory = false;
  if (sessionId) {
    try {
      const resp = await fetch(`/api/session/${sessionId}`);
      const data = await resp.json();
      if (data.message_count > 0) {
        setTitle(data.title);
        loadMessageHistory(sessionId);
        hasHistory = true;
      }
    } catch (err) {
      sessionId = "";
    }
  }

  if (!hasHistory) {
    sessionId = crypto.randomUUID().slice(0, 12);
    localStorage.setItem("agentkb_session_id", sessionId);
  }

  loadSessionList();

  // 设置用户信息
  const userName = localStorage.getItem("agentkb_user_name") || "本地用户";
  $("#user-name").textContent = userName;
  $("#user-avatar").textContent = userName.charAt(0).toUpperCase();

  // 自动调整输入框高度
  msgInput.addEventListener("input", () => {
    msgInput.style.height = "auto";
    msgInput.style.height = Math.min(msgInput.scrollHeight, 150) + "px";
  });
}

// ── 事件绑定 ──────────────────────────────────────────────────

sendBtn.addEventListener("click", () => {
  const text = msgInput.value.trim();
  if (!text) return;
  msgInput.value = "";
  msgInput.style.height = "auto";
  sendMessage(text);
});

msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = msgInput.value.trim();
    if (!text) return;
    msgInput.value = "";
    msgInput.style.height = "auto";
    sendMessage(text);
  }
});

$("#btn-upload").addEventListener("click", () => uploadInput.click());

uploadInput.addEventListener("change", () => {
  if (uploadInput.files.length > 0) {
    uploadFiles(uploadInput.files);
    uploadInput.value = "";
  }
});

$("#btn-clear").addEventListener("click", clearSession);
$("#btn-new-chat").addEventListener("click", createNewSession);
$("#btn-toggle-sidebar").addEventListener("click", toggleSidebar);

// 欢迎页提示词点击
document.addEventListener("click", (e) => {
  const chip = e.target.closest(".hint-chip");
  if (chip && chip.dataset.text) {
    sendMessage(chip.dataset.text);
  }
});

// 拖拽上传
document.addEventListener("dragover", (e) => {
  e.preventDefault();
});

document.addEventListener("drop", (e) => {
  e.preventDefault();
  if (e.dataTransfer.files.length > 0) {
    uploadFiles(e.dataTransfer.files);
  }
});

// ── 启动 ──────────────────────────────────────────────────────
init();
