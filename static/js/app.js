/* global marked */
// ── AgentKB 前端逻辑 ──────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════
//  状态
// ══════════════════════════════════════════════════════════════

let sessionId = localStorage.getItem("agentkb_session_id") || "";
let sidebarVisible = true;
let agentMode = localStorage.getItem("agentkb_agent_mode") || "auto";  // "auto" | "simple"
const generatingSessions = new Set();
const streamControllers = new Map();
const sessionEventIds = new Map();
let knowledgeRefreshTimer = null;
let pendingAttachments = [];
let mediaRecorder = null;
let recordedChunks = [];

// ══════════════════════════════════════════════════════════════
//  DOM 引用
// ══════════════════════════════════════════════════════════════

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const chatContainer = $("#chat");
const msgInput = $("#msg-input");
const sendBtn = $("#send-btn");
const uploadInput = $("#upload-input");
const chatImageInput = $("#chat-image-input");
const attachmentTray = $("#attachment-tray");
const recordButton = $("#btn-record");
const sessionList = $("#session-list");
const kbFileList = $("#kb-file-list");
const kbCount = $("#kb-count");
const currentSessionTitle = $("#current-session-title");
const welcomeScreen = $("#welcome-screen");
const sidebar = $("#sidebar");
const sidebarToggle = $("#btn-toggle-sidebar");
const sidebarBackdrop = $("#sidebar-backdrop");
const toastContainer = $("#toast-container");
const graphOverlay = $("#graph-overlay");
const graphStats = $("#graph-stats");
const graphResults = $("#graph-results");
const graphQuery = $("#graph-query");
const evalOverlay = $("#eval-overlay");
const evalJobs = $("#eval-jobs");
const evalBaselines = $("#eval-baselines");
const evalGates = $("#eval-gates");
const agentsOverlay = $("#agents-overlay");
const agentDraftForm = $("#agent-draft-form");
const agentConfirmForm = $("#agent-confirm-form");
const customAgentList = $("#custom-agent-list");
const agentTools = $("#agent-tools");
const mcpOverlay = $("#mcp-overlay");
const mcpServerForm = $("#mcp-server-form");
const mcpServerList = $("#mcp-server-list");
let availableAgentTools = [];

// ══════════════════════════════════════════════════════════════
//  工具函数
// ══════════════════════════════════════════════════════════════

function updateComposerState() {
  sendBtn.disabled = generatingSessions.has(sessionId);
}

function renderAttachmentTray() {
  attachmentTray.hidden = pendingAttachments.length === 0;
  attachmentTray.innerHTML = pendingAttachments.map((attachment) => `
    <div class="attachment-chip">
      <img src="${attachment.url}" alt="">
      <span>${escapeHtml(attachment.name)}</span>
      <button type="button" data-remove-attachment="${escapeHtml(attachment.id)}" aria-label="移除图片">×</button>
    </div>
  `).join("");
}

function disconnectSessionStream(sid) {
  const controller = streamControllers.get(sid);
  if (controller) {
    controller.abort();
    streamControllers.delete(sid);
  }
}

function beginSessionStream(sid) {
  disconnectSessionStream(sid);
  const controller = new AbortController();
  streamControllers.set(sid, controller);
  generatingSessions.add(sid);
  updateComposerState();
  return controller;
}

function finishSessionStream(sid, controller) {
  if (streamControllers.get(sid) === controller) {
    streamControllers.delete(sid);
  }
  generatingSessions.delete(sid);
  updateComposerState();
}

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

function renderApprovalCard(event) {
  const argumentsText = JSON.stringify(event.arguments || {}, null, 2);
  return `
    <div class="tool-approval" data-approval-id="${escapeHtml(event.approval_id)}">
      <div class="tool-approval__title">需要人工确认</div>
      <div class="tool-approval__message">${escapeHtml(event.message || "该工具需要确认")}</div>
      <pre>${escapeHtml(argumentsText)}</pre>
      <div class="tool-approval__actions">
        <button type="button" data-decision="reject">拒绝</button>
        <button type="button" class="primary" data-decision="approve">批准执行</button>
      </div>
      <div class="tool-approval__status"></div>
    </div>
  `;
}

function bindApprovalButtons(container) {
  container.querySelectorAll(".tool-approval").forEach((card) => {
    if (card.dataset.bound === "true") return;
    card.dataset.bound = "true";
    const approvalId = card.dataset.approvalId;
    const buttons = card.querySelectorAll("button[data-decision]");
    const status = card.querySelector(".tool-approval__status");

    buttons.forEach((button) => {
      button.addEventListener("click", async () => {
        buttons.forEach((item) => { item.disabled = true; });
        const approved = button.dataset.decision === "approve";
        status.textContent = approved ? "正在提交批准…" : "正在提交拒绝…";
        try {
          const response = await fetch(
            `/api/tool-approvals/${encodeURIComponent(approvalId)}/decision`,
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ approved }),
            }
          );
          if (!response.ok) {
            const data = await response.json();
            throw new Error(data.detail || `HTTP ${response.status}`);
          }
          status.textContent = approved ? "已批准，正在执行…" : "已拒绝，正在生成说明…";
        } catch (error) {
          status.textContent = `提交失败：${error.message}`;
          buttons.forEach((item) => { item.disabled = false; });
        }
      });
    });
  });
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
  currentSessionTitle.textContent = title || "新对话";
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
      const title = escapeHtml(s.title || "新对话");
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
  const previousSessionId = sessionId;
  disconnectSessionStream(previousSessionId);

  try {
    const resp = await fetch(`/api/session/${sid}`);
    const data = await resp.json();
    discardPendingAttachments();
    sessionId = data.session_id;
    localStorage.setItem("agentkb_session_id", sessionId);
    setTitle(data.title);
    if (data.is_generating) {
      generatingSessions.add(sessionId);
    } else {
      generatingSessions.delete(sessionId);
    }
    updateComposerState();

    // 加载消息历史
    if (data.message_count > 0 || data.is_generating) {
      hideWelcome();
      await loadMessageHistory(data.session_id, data.is_generating);
    } else {
      chatContainer.innerHTML = "";
      chatContainer.appendChild(welcomeScreen);
      welcomeScreen.style.display = "";
    }

    loadSessionList();
  } catch (err) {
    showToast("切换会话失败", "error");
    if (previousSessionId === sessionId && generatingSessions.has(sessionId)) {
      reconnectStream(sessionId);
    }
  }
}

async function loadMessageHistory(sid, isGenerating = false) {
  try {
    const msgsResp = await fetch(`/api/session/${sid}/messages`);
    const msgsData = await msgsResp.json();
    if (sid !== sessionId) return;
    renderHistoryMessages(msgsData.messages || [], sid, isGenerating);
  } catch (err) {
    if (sid !== sessionId) return;
    // 降级：清空聊天区
    chatContainer.innerHTML = "";
    chatContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
  }
}

function renderHistoryMessages(messages, sid, isGenerating) {
  hideWelcome();

  // 移除之前的消息
  chatContainer.querySelectorAll(".message").forEach((el) => el.remove());

  let lastAiMsg = null;
  messages.forEach((m, idx) => {
    if (m.role === "human") {
      addUserMessage(m.content, m.attachments || []);
    } else if (m.role === "ai") {
      const isLast = idx === messages.length - 1;
      if (isLast) {
        lastAiMsg = m;
        // 最后一条 AI 消息用 placeholder 渲染，方便重连时填入内容
        const el = addAssistantPlaceholder();
        if (m.content) {
          el.innerHTML = renderMarkdown(m.content);
        } else {
          el.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
        }
      } else {
        addAssistantMessage(m.content);
      }
    }
  });

  if (isGenerating && !lastAiMsg) {
    lastAiMsg = { role: "ai", content: "" };
    const el = addAssistantPlaceholder();
    el.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
  }

  scrollToBottom();

  if (lastAiMsg && isGenerating) {
    setTimeout(() => reconnectStream(sid), 50);
  }
}

async function reconnectStream(sid) {
  if (sid !== sessionId || streamControllers.has(sid)) return;

  // SSE Last-Event-ID 断点续传
  const assistantEl = document.getElementById("current-assistant");
  if (!assistantEl) return;

  const controller = beginSessionStream(sid);
  let terminal = false;
  const resumeFromEventId = sessionEventIds.get(sid) || 0;

  // 刷新后事件 ID 丢失，服务端会完整回放；此时必须从空内容重建，避免重复。
  let accumulated = resumeFromEventId > 0
    ? assistantEl.textContent.trim() || ""
    : "";
  let toolLines = [];
  let throttleTimer = null;
  let finalized = false;
  let lastSources = null;
  let lastMessageId = null;

  function flushContent() {
    let display = accumulated;
    if (toolLines.length > 0) {
      display += '\n\n<div class="tool-status">' + toolLines.join("<br>") + "</div>";
    }
    assistantEl.innerHTML = renderMarkdown(display);
    bindApprovalButtons(assistantEl);
    scrollToBottom();
  }

  try {
    const response = await fetch(`/api/chat/stream/${sid}`, {
      headers: {
        "Last-Event-ID": String(resumeFromEventId),
      },
      signal: controller.signal,
    });
    if (!response.ok) {
      terminal = true;
      throw new Error(`SSE 请求失败: ${response.status}`);
    }

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
          const eventId = parseInt(line.slice(4));
          if (eventId) sessionEventIds.set(sid, eventId);
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case "message_id":
              lastMessageId = event.message_id;
              break;
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
                    lastSources = results.map((r) => ({
                      filename: r.filename || "unknown",
                      score: r.score || 0,
                    }));
                  }
                } catch (e) {}
              }
              flushContent();
              break;
            case "approval_required":
              toolLines.push(renderApprovalCard(event));
              flushContent();
              break;
            case "done":
              if (throttleTimer) {
                clearTimeout(throttleTimer);
                throttleTimer = null;
              }
              if (lastSources && lastSources.length > 0) {
                let html = renderMarkdown(accumulated);
                html = highlightCitations(html, lastSources);
                assistantEl.innerHTML = html;
                assistantEl.innerHTML += renderSourceCard(lastSources);
              } else {
                assistantEl.innerHTML = renderMarkdown(accumulated);
              }
              addFeedbackButtons(assistantEl, lastMessageId);
              finalized = true;
              terminal = true;
              break;
            case "error":
              accumulated += '\n\n<span style="color:var(--danger)">' + escapeHtml(event.message || "未知错误") + "</span>";
              flushContent();
              terminal = true;
              break;
          }
        } catch (e) {}
      }
    }

    terminal = true;
    if (throttleTimer) { clearTimeout(throttleTimer); throttleTimer = null; }
    if (!finalized) {
      assistantEl.innerHTML = renderMarkdown(accumulated);
    }
    finalizeAssistant(assistantEl);
    scrollToBottom();
  } catch (err) {
    if (err.name === "AbortError") return;

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
    if (terminal) {
      finishSessionStream(sid, controller);
    } else if (streamControllers.get(sid) === controller) {
      streamControllers.delete(sid);
      updateComposerState();
    }
  }
}

function createNewSession() {
  disconnectSessionStream(sessionId);
  discardPendingAttachments();
  // 仅在本地生成 ID，后端在首条消息发送时才 ensure_session
  sessionId = crypto.randomUUID().slice(0, 12);
  localStorage.setItem("agentkb_session_id", sessionId);
  setTitle("新对话");
  chatContainer.innerHTML = "";
  chatContainer.appendChild(welcomeScreen);
  welcomeScreen.style.display = "";
  msgInput.value = "";
  msgInput.style.height = "auto";
  renderAttachmentTray();
  updateComposerState();
  msgInput.focus();
}

function discardPendingAttachments() {
  const attachments = pendingAttachments;
  pendingAttachments = [];
  renderAttachmentTray();
  attachments.forEach((attachment) => {
    fetch(
      `/api/chat/attachments/${encodeURIComponent(attachment.id)}?session_id=${encodeURIComponent(sessionId)}`,
      { method: "DELETE" }
    ).catch(() => {});
  });
}

async function deleteSession(sid) {
  if (generatingSessions.has(sid)) {
    showToast("该会话仍在生成回复，暂时无法删除", "info");
    return;
  }

  try {
    disconnectSessionStream(sid);
    generatingSessions.delete(sid);
    sessionEventIds.delete(sid);
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
        const graphStatus = f.graph_status || "queued";
        const graphLabels = {
          disabled: "图谱关闭",
          queued: "图谱排队",
          processing: "图谱构建",
          ready: "图谱就绪",
          failed: "图谱失败",
        };
        return `
          <div class="kb-file-item" title="${name} · ${chunks} 分块 · ${size}">
            <span class="file-type-badge">${type}</span>
            <span class="file-name-text">${name}</span>
            <span class="graph-status ${graphStatus}" title="${escapeHtml(f.graph_error || "")}">
              ${graphLabels[graphStatus] || graphStatus}
            </span>
            ${graphStatus === "failed" ? `
              <button class="file-reindex-btn" data-id="${f.id}" title="重建知识图谱">↻</button>
            ` : ""}
            <button class="file-delete-btn" data-id="${f.id}" title="删除文件">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
            </button>
          </div>`;
      })
      .join("");

    if (knowledgeRefreshTimer) clearTimeout(knowledgeRefreshTimer);
    if (files.some((file) => ["queued", "processing"].includes(file.graph_status))) {
      knowledgeRefreshTimer = setTimeout(loadKnowledgeFiles, 3000);
    }

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

    kbFileList.querySelectorAll(".file-reindex-btn").forEach((btn) => {
      btn.addEventListener("click", async (event) => {
        event.stopPropagation();
        const resp = await fetch(
          `/api/knowledge/graph/reindex/${encodeURIComponent(btn.dataset.id)}`,
          { method: "POST" }
        );
        if (resp.ok) {
          showToast("知识图谱已重新排队", "info");
          loadKnowledgeFiles();
        } else {
          const data = await resp.json();
          showToast(data.detail || "重建失败", "error");
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

async function loadKnowledgeGraph(query = "") {
  graphResults.innerHTML = '<div class="graph-empty">正在读取知识图谱…</div>';
  try {
    const params = new URLSearchParams();
    if (query.trim()) params.set("query", query.trim());
    const resp = await fetch(`/api/knowledge/graph?${params.toString()}`);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    const stats = data.stats || {};
    graphStats.innerHTML = `
      <div><strong>${stats.entities || 0}</strong><span>实体</span></div>
      <div><strong>${stats.relations || 0}</strong><span>关系</span></div>
      <div><strong>${stats.indexed_files || 0}</strong><span>已索引文件</span></div>
      <div><strong>${stats.pending_files || 0}</strong><span>排队中</span></div>
    `;
    const edges = data.edges || [];
    if (!query.trim()) {
      graphResults.innerHTML = '<div class="graph-empty">输入实体名称以浏览知识库中的关系证据。</div>';
      return;
    }
    if (edges.length === 0) {
      graphResults.innerHTML = '<div class="graph-empty">没有找到相关实体关系，可尝试更短的实体名称。</div>';
      return;
    }
    graphResults.innerHTML = edges.map((edge) => `
      <article class="graph-edge">
        <div class="graph-edge__relation">
          <strong>${escapeHtml(edge.source)}</strong>
          <span>${escapeHtml(edge.predicate)}</span>
          <strong>${escapeHtml(edge.target)}</strong>
        </div>
        <p>${escapeHtml(edge.evidence || "无证据摘要")}</p>
        <small>来源：${escapeHtml(edge.filename || edge.file_id)}</small>
      </article>
    `).join("");
  } catch (error) {
    graphResults.innerHTML = `<div class="graph-empty">读取失败：${escapeHtml(error.message)}</div>`;
  }
}

function openKnowledgeGraph() {
  graphOverlay.hidden = false;
  document.body.classList.add("modal-open");
  loadKnowledgeGraph();
  graphQuery.focus();
}

function closeKnowledgeGraph() {
  graphOverlay.hidden = true;
  document.body.classList.remove("modal-open");
}

async function loadEvaluationCenter() {
  evalJobs.innerHTML = '<div class="graph-empty">正在读取评估任务…</div>';
  try {
    const [jobsResp, baselinesResp, gatesResp] = await Promise.all([
      fetch("/api/eval/jobs?limit=20"),
      fetch("/api/eval/baselines?limit=20"),
      fetch("/api/eval/gates?limit=20"),
    ]);
    if (!jobsResp.ok || !baselinesResp.ok || !gatesResp.ok) {
      throw new Error("评估数据读取失败");
    }
    const jobs = (await jobsResp.json()).jobs || [];
    const baselines = (await baselinesResp.json()).baselines || [];
    const gates = (await gatesResp.json()).gates || [];
    const activeScopes = new Set(
      baselines.filter((item) => item.is_active).map((item) => item.scope)
    );

    const completed = jobs.filter((job) => job.status === "done");
    evalJobs.innerHTML = completed.length ? completed.map((job) => `
      <article class="eval-item">
        <div><strong>${escapeHtml(job.params.prompt_version || "default")}</strong><small>${escapeHtml(formatDate(job.finished_at))}</small></div>
        <button type="button" data-create-baseline="${escapeHtml(job.job_id || job.id)}">设为基线</button>
      </article>
    `).join("") : '<div class="graph-empty">暂无已完成评估。</div>';

    evalBaselines.innerHTML = baselines.length ? baselines.map((baseline) => `
      <article class="eval-item">
        <div>
          <strong>${escapeHtml(baseline.name)}</strong>
          <small>${escapeHtml(baseline.scope)}${baseline.is_active ? " · 当前激活" : ""}</small>
        </div>
        ${baseline.is_active ? "" : `<button type="button" data-activate-baseline="${escapeHtml(baseline.id)}">激活</button>`}
      </article>
    `).join("") : '<div class="graph-empty">暂无评估基线。</div>';

    evalGates.innerHTML = gates.length ? gates.map((gate) => `
      <article class="eval-item">
        <div>
          <strong class="${gate.status === "passed" ? "eval-passed" : "eval-failed"}">${gate.status === "passed" ? "通过" : "失败"}</strong>
          <small>${escapeHtml(gate.baseline_name)} · ${escapeHtml(formatDate(gate.created_at))}</small>
        </div>
      </article>
    `).join("") : `<div class="graph-empty">${activeScopes.size ? "尚未执行门禁。" : "创建并激活基线后可执行门禁。"}</div>`;
  } catch (error) {
    evalJobs.innerHTML = `<div class="graph-empty">读取失败：${escapeHtml(error.message)}</div>`;
  }
}

async function createEvaluationBaseline(jobId) {
  const name = window.prompt("请输入基线名称", `baseline-${new Date().toISOString().slice(0, 10)}`);
  if (!name || !name.trim()) return;
  const response = await fetch("/api/eval/baselines", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ job_id: jobId, name: name.trim(), activate: true }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "创建基线失败");
  showToast("评估基线已创建并激活", "success");
  await loadEvaluationCenter();
}

async function activateEvaluationBaseline(baselineId) {
  const response = await fetch(`/api/eval/baselines/${encodeURIComponent(baselineId)}/activate`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("激活基线失败");
  showToast("评估基线已激活", "success");
  await loadEvaluationCenter();
}

function openEvaluationCenter() {
  evalOverlay.hidden = false;
  document.body.classList.add("modal-open");
  loadEvaluationCenter();
}

function closeEvaluationCenter() {
  evalOverlay.hidden = true;
  document.body.classList.remove("modal-open");
}

async function loadCustomAgents() {
  customAgentList.innerHTML = '<div class="graph-empty">正在读取 Agent…</div>';
  const response = await fetch("/api/agents");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "读取 Agent 失败");
  availableAgentTools = data.tools || [];
  const agents = data.agents || [];
  customAgentList.innerHTML = agents.length ? agents.map((agent) => `
    <article class="eval-item agent-list-item">
      <div>
        <strong>${escapeHtml(agent.display_name)}</strong>
        <small>${escapeHtml(agent.name)} · ${escapeHtml(agent.intents.join(", "))}</small>
        <p>${escapeHtml(agent.description)}</p>
      </div>
      <div class="agent-item-actions">
        <button type="button" data-agent-status="${escapeHtml(agent.id)}" data-next-status="${agent.status === "active" ? "disabled" : "active"}">
          ${agent.status === "active" ? "停用" : "启用"}
        </button>
        <button type="button" data-delete-agent="${escapeHtml(agent.id)}">删除</button>
      </div>
    </article>
  `).join("") : '<div class="graph-empty">尚未创建自定义 Agent。</div>';
}

function populateAgentDraft(draft) {
  $("#agent-name").value = draft.name;
  $("#agent-display-name").value = draft.display_name;
  $("#agent-description").value = draft.description;
  $("#agent-instructions").value = draft.instructions;
  $("#agent-intents").value = draft.intents.join(", ");
  const selected = new Set(draft.allowed_tools || []);
  agentTools.innerHTML = availableAgentTools.length
    ? availableAgentTools.map((tool) => `
      <label title="${escapeHtml(tool.description)}">
        <input type="checkbox" value="${escapeHtml(tool.name)}" ${selected.has(tool.name) ? "checked" : ""}>
        <span>${escapeHtml(tool.name)}</span>
      </label>
    `).join("")
    : '<span class="graph-empty">没有可分配的低风险工具。</span>';
  agentConfirmForm.hidden = false;
}

async function generateAgentDraft(request) {
  const response = await fetch("/api/agents/draft", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ request }),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "草案生成失败");
  populateAgentDraft(data.draft);
}

async function confirmAgentDraft() {
  const draft = {
    name: $("#agent-name").value.trim(),
    display_name: $("#agent-display-name").value.trim(),
    description: $("#agent-description").value.trim(),
    instructions: $("#agent-instructions").value.trim(),
    intents: $("#agent-intents").value.split(",").map((item) => item.trim()).filter(Boolean),
    allowed_tools: Array.from(agentTools.querySelectorAll("input:checked")).map((item) => item.value),
    model_name: null,
  };
  const response = await fetch("/api/agents", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(draft),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "创建 Agent 失败");
  agentConfirmForm.hidden = true;
  agentDraftForm.reset();
  showToast(`${data.agent.display_name} 已注册`, "success");
  await loadCustomAgents();
}

function openAgentStudio() {
  agentsOverlay.hidden = false;
  document.body.classList.add("modal-open");
  loadCustomAgents().catch((error) => {
    customAgentList.innerHTML = `<div class="graph-empty">${escapeHtml(error.message)}</div>`;
  });
}

function closeAgentStudio() {
  agentsOverlay.hidden = true;
  document.body.classList.remove("modal-open");
}

function parseJsonObject(selector, label) {
  const value = $(selector).value.trim();
  if (!value) return {};
  const parsed = JSON.parse(value);
  if (!parsed || Array.isArray(parsed) || typeof parsed !== "object") {
    throw new Error(`${label} 必须是 JSON 对象`);
  }
  return parsed;
}

function renderMcpServer(server) {
  const connected = server.connection_status === "connected";
  const tools = server.tools || [];
  return `
    <article class="mcp-server">
      <div class="mcp-server__header">
        <div>
          <strong>${escapeHtml(server.name)}</strong>
          <small>${escapeHtml(server.transport)} · ${tools.length} 个工具</small>
        </div>
        <span class="mcp-status" data-status="${escapeHtml(server.connection_status)}">${escapeHtml(server.connection_status)}</span>
      </div>
      ${server.last_error ? `<div class="eval-failed">${escapeHtml(server.last_error)}</div>` : ""}
      <div class="mcp-server__actions">
        ${connected
          ? `<button type="button" data-mcp-action="refresh" data-server-id="${escapeHtml(server.id)}">刷新工具</button>
             <button type="button" data-mcp-action="disconnect" data-server-id="${escapeHtml(server.id)}">断开</button>`
          : `<button type="button" data-mcp-action="connect" data-server-id="${escapeHtml(server.id)}">连接</button>`}
        <button type="button" data-mcp-action="delete" data-server-id="${escapeHtml(server.id)}">删除</button>
      </div>
      ${tools.length ? `<div class="mcp-tools">${tools.map((tool) => `
        <label class="mcp-tool" title="${escapeHtml(tool.description)}">
          <span>
            <strong>${escapeHtml(tool.remote_name)}</strong>
            <small>${tool.requires_confirmation ? "需要人工确认" : "可直接调用"}</small>
          </span>
          <input type="checkbox"
                 data-mcp-tool="${escapeHtml(tool.remote_name)}"
                 data-server-id="${escapeHtml(server.id)}"
                 ${tool.enabled ? "checked" : ""}>
        </label>
      `).join("")}</div>` : ""}
    </article>
  `;
}

async function loadMcpServers() {
  mcpServerList.innerHTML = '<div class="graph-empty">正在读取 MCP 服务…</div>';
  const response = await fetch("/api/mcp/servers");
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "读取 MCP 服务失败");
  mcpServerList.innerHTML = data.servers.length
    ? data.servers.map(renderMcpServer).join("")
    : '<div class="graph-empty">尚未配置 MCP 服务。</div>';
}

function openMcpCenter() {
  mcpOverlay.hidden = false;
  document.body.classList.add("modal-open");
  loadMcpServers().catch((error) => {
    mcpServerList.innerHTML = `<div class="graph-empty">${escapeHtml(error.message)}</div>`;
  });
}

function closeMcpCenter() {
  mcpOverlay.hidden = true;
  document.body.classList.remove("modal-open");
}

// ══════════════════════════════════════════════════════════════
//  消息渲染
// ══════════════════════════════════════════════════════════════

function hideWelcome() {
  if (welcomeScreen && welcomeScreen.style.display !== "none") {
    welcomeScreen.style.display = "none";
  }
}

function addUserMessage(text, attachments = []) {
  hideWelcome();
  const el = document.createElement("div");
  el.className = "message user";
  const images = attachments.map((attachment) => `
    <img class="message-image" src="/api/chat/attachments/${encodeURIComponent(attachment.id)}" alt="${escapeHtml(attachment.name || "图片附件")}">
  `).join("");
  el.innerHTML = `${images}${text ? `<div>${escapeHtml(text)}</div>` : ""}`;
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
  const targetSessionId = sessionId;
  if (
    generatingSessions.has(targetSessionId)
    || (!text.trim() && pendingAttachments.length === 0)
  ) return;
  const attachments = pendingAttachments.slice();
  sessionEventIds.set(targetSessionId, 0);
  const controller = beginSessionStream(targetSessionId);

  addUserMessage(text, attachments);
  const assistantEl = addAssistantPlaceholder();
  // 思考动画
  assistantEl.innerHTML = '<div class="thinking-indicator"><span></span><span></span><span></span></div>';
  scrollToBottom();

  let accumulated = "";
  let toolLines = [];
  let throttleTimer = null;
  let finalized = false;
  let terminal = false;
  let lastSources = null;
  let lastTrace = null;
  let lastMessageId = null;

  function flushContent() {
    let display = accumulated;
    if (toolLines.length > 0) {
      display += '\n\n<div class="tool-status">' + toolLines.join("<br>") + "</div>";
    }
    assistantEl.innerHTML = renderMarkdown(display);
    bindApprovalButtons(assistantEl);
    scrollToBottom();
  }

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        session_id: targetSessionId,
        mode: agentMode,
        attachment_ids: attachments.map((attachment) => attachment.id),
      }),
      signal: controller.signal,
    });
    if (!response.ok) {
      terminal = true;
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `SSE 请求失败: ${response.status}`);
    }
    pendingAttachments = [];
    renderAttachmentTray();

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
          const eventId = parseInt(line.slice(4));
          if (eventId) sessionEventIds.set(targetSessionId, eventId);
          continue;
        }
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
            case "message_id":
              lastMessageId = event.message_id;
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
              // 区分 Supervisor 和普通工具/Agent
              if (event.name === "supervisor") {
                toolLines.push(
                  '🧠 <b>Supervisor</b> 分析意图中……'
                );
              } else if (["content_creator", "task_manager", "learning_tutor", "social_writer", "knowledge_agent"].includes(event.name)) {
                const agentLabels = {
                  content_creator: "内容创作 Agent",
                  task_manager: "任务管理 Agent",
                  learning_tutor: "学习导师 Agent",
                  social_writer: "社媒内容 Agent",
                  knowledge_agent: "知识检索 Agent",
                };
                toolLines.push(
                  '🤖 <b>' + (agentLabels[event.name] || event.name) + '</b> 工作中……'
                );
              } else {
                toolLines.push(
                  '<span class="spinner"></span> 正在调用 <b>' +
                    escapeHtml(event.name) +
                    "</b>……"
                );
              }
              flushContent();
              break;

            case "tool_end":
              toolLines = toolLines.map((l) =>
                l.replace('class="spinner"', 'class="done-icon"')
                  .replace("正在调用", "已完成")
                  .replace("分析意图中……", "分析完成")
                  .replace("工作中……", "已完成")
              );
              // 提取检索来源 (search_knowledge_base)
              if (event.name === "search_knowledge_base") {
                try {
                  const output = typeof event.output === "string"
                    ? JSON.parse(event.output) : event.output;
                  const results = output?.data?.results || [];
                  if (results.length > 0) {
                    lastSources = results.map((r) => ({
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
                lastTrace = event.trace;
              }
              flushContent();
              break;

            case "approval_required":
              toolLines.push(renderApprovalCard(event));
              flushContent();
              break;

            case "done":
              if (throttleTimer) {
                clearTimeout(throttleTimer);
                throttleTimer = null;
              }
              if (lastSources && lastSources.length > 0) {
                let html = renderMarkdown(accumulated);
                html = highlightCitations(html, lastSources);
                assistantEl.innerHTML = html;
                assistantEl.innerHTML += renderSourceCard(lastSources);
              } else {
                assistantEl.innerHTML = renderMarkdown(accumulated);
              }
              if (lastTrace) {
                assistantEl.innerHTML += renderTimingPanel(lastTrace);
              }
              addFeedbackButtons(assistantEl, lastMessageId);
              finalizeAssistant(assistantEl);
              scrollToBottom();
              finalized = true;
              terminal = true;
              break;

            case "error":
              accumulated +=
                '\n\n<span style="color:var(--danger)">' +
                escapeHtml(event.message || "未知错误") +
                "</span>";
              flushContent();
              terminal = true;
              break;
          }
        } catch (e) {
          // 忽略 JSON 解析错误
        }
      }
    }

    terminal = true;
    if (!finalized) {
      if (throttleTimer) {
        clearTimeout(throttleTimer);
        throttleTimer = null;
      }
      assistantEl.innerHTML = renderMarkdown(accumulated);
      addFeedbackButtons(assistantEl, lastMessageId);
      finalizeAssistant(assistantEl);
      scrollToBottom();
    }

    // 更新侧边栏会话列表（标题可能变化）
    loadSessionList();
  } catch (err) {
    if (err.name === "AbortError") return;

    assistantEl.innerHTML = renderMarkdown(
      accumulated +
        '\n\n<span style="color:var(--danger)">请求失败: ' +
        escapeHtml(err.message) +
        "</span>"
    );
    addFeedbackButtons(assistantEl, lastMessageId);
    finalizeAssistant(assistantEl);
  } finally {
    if (terminal) {
      finishSessionStream(targetSessionId, controller);
    } else if (streamControllers.get(targetSessionId) === controller) {
      streamControllers.delete(targetSessionId);
      updateComposerState();
    }
    if (targetSessionId === sessionId) msgInput.focus();
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

async function uploadChatImages(files) {
  const remaining = 4 - pendingAttachments.length;
  const selected = Array.from(files).slice(0, remaining);
  if (selected.length === 0) {
    showToast("每条消息最多添加 4 张图片", "error");
    return;
  }
  for (const file of selected) {
    const formData = new FormData();
    formData.append("file", file);
    const response = await fetch(
      `/api/chat/attachments?session_id=${encodeURIComponent(sessionId)}`,
      { method: "POST", body: formData }
    );
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `图片上传失败: ${file.name}`);
    }
    pendingAttachments.push(data);
    renderAttachmentTray();
  }
  msgInput.focus();
}

async function removePendingAttachment(attachmentId) {
  const response = await fetch(
    `/api/chat/attachments/${encodeURIComponent(attachmentId)}?session_id=${encodeURIComponent(sessionId)}`,
    { method: "DELETE" }
  );
  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    throw new Error(data.detail || "移除图片失败");
  }
  pendingAttachments = pendingAttachments.filter(
    (attachment) => attachment.id !== attachmentId
  );
  renderAttachmentTray();
}

async function toggleRecording() {
  if (mediaRecorder?.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
    showToast("当前浏览器不支持录音", "error");
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) recordedChunks.push(event.data);
    });
    mediaRecorder.addEventListener("stop", async () => {
      recordButton.classList.remove("recording");
      recordButton.querySelector("span").textContent = "语音";
      stream.getTracks().forEach((track) => track.stop());
      const audio = new Blob(recordedChunks, { type: mediaRecorder.mimeType });
      const formData = new FormData();
      formData.append("file", audio, "recording.webm");
      try {
        showToast("正在转写语音…", "info");
        const response = await fetch("/api/transcribe", {
          method: "POST",
          body: formData,
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "语音转写失败");
        msgInput.value = [msgInput.value.trim(), data.text].filter(Boolean).join(" ");
        msgInput.dispatchEvent(new Event("input"));
        msgInput.focus();
      } catch (error) {
        showToast(error.message, "error");
      }
    });
    mediaRecorder.start();
    recordButton.classList.add("recording");
    recordButton.querySelector("span").textContent = "停止";
  } catch (error) {
    showToast(`无法开始录音：${error.message}`, "error");
  }
}

// ══════════════════════════════════════════════════════════════
//  清空会话
// ══════════════════════════════════════════════════════════════

async function clearSession() {
  if (generatingSessions.has(sessionId)) return;
  try {
    const resp = await fetch("/api/session/clear", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    });
    const data = await resp.json();
    discardPendingAttachments();
    sessionId = data.session_id;
    localStorage.setItem("agentkb_session_id", sessionId);
    setTitle(data.title);
    chatContainer.innerHTML = "";
    chatContainer.appendChild(welcomeScreen);
    welcomeScreen.style.display = "";
    msgInput.value = "";
    updateComposerState();
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
  sidebar.classList.toggle("collapsed", !sidebarVisible);
  sidebarToggle.setAttribute("aria-expanded", String(sidebarVisible));
  localStorage.setItem("agentkb_sidebar_visible", sidebarVisible);
}

// ══════════════════════════════════════════════════════════════
//  初始化和事件绑定
// ══════════════════════════════════════════════════════════════

async function init() {
  // 恢复侧边栏状态
  const savedVisibility = localStorage.getItem("agentkb_sidebar_visible");
  sidebarVisible = savedVisibility === null
    ? !window.matchMedia("(max-width: 900px)").matches
    : savedVisibility === "true";
  sidebar.classList.toggle("collapsed", !sidebarVisible);
  sidebarToggle.setAttribute("aria-expanded", String(sidebarVisible));

  // 加载知识库文件
  loadKnowledgeFiles();

  // 初始化会话
  let hasHistory = false;
  let hasExistingSession = false;
  if (sessionId) {
    try {
      const resp = await fetch(`/api/session/${sessionId}`);
      const data = await resp.json();
      hasExistingSession = true;
      if (data.is_generating) {
        generatingSessions.add(sessionId);
      } else {
        generatingSessions.delete(sessionId);
      }
      if (data.message_count > 0 || data.is_generating) {
        setTitle(data.title);
        loadMessageHistory(sessionId, data.is_generating);
        hasHistory = true;
      }
    } catch (err) {
      sessionId = "";
    }
  }

  if (!hasHistory && !hasExistingSession) {
    sessionId = crypto.randomUUID().slice(0, 12);
    localStorage.setItem("agentkb_session_id", sessionId);
  }

  updateComposerState();
  loadSessionList();

  try {
    const response = await fetch("/api/capabilities");
    const capabilities = await response.json();
    recordButton.hidden = !capabilities.transcription?.enabled;
    chatImageInput.disabled = !capabilities.vision?.enabled;
  } catch (error) {
    recordButton.hidden = true;
  }

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
  if (!text && pendingAttachments.length === 0) return;
  msgInput.value = "";
  msgInput.style.height = "auto";
  sendMessage(text);
});

msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    const text = msgInput.value.trim();
    if (!text && pendingAttachments.length === 0) return;
    msgInput.value = "";
    msgInput.style.height = "auto";
    sendMessage(text);
  }
});

$("#btn-upload").addEventListener("click", () => uploadInput.click());
$("#btn-graph").addEventListener("click", openKnowledgeGraph);
$("#btn-eval").addEventListener("click", openEvaluationCenter);
$("#btn-agents").addEventListener("click", openAgentStudio);
$("#btn-mcp").addEventListener("click", openMcpCenter);
$("#btn-close-graph").addEventListener("click", closeKnowledgeGraph);
$("#btn-close-eval").addEventListener("click", closeEvaluationCenter);
$("#btn-close-agents").addEventListener("click", closeAgentStudio);
$("#btn-close-mcp").addEventListener("click", closeMcpCenter);
$("#graph-search").addEventListener("submit", (event) => {
  event.preventDefault();
  loadKnowledgeGraph(graphQuery.value);
});
graphOverlay.addEventListener("click", (event) => {
  if (event.target === graphOverlay) closeKnowledgeGraph();
});
evalOverlay.addEventListener("click", (event) => {
  if (event.target === evalOverlay) closeEvaluationCenter();
});
evalOverlay.addEventListener("click", async (event) => {
  const createButton = event.target.closest("[data-create-baseline]");
  const activateButton = event.target.closest("[data-activate-baseline]");
  try {
    if (createButton) {
      await createEvaluationBaseline(createButton.dataset.createBaseline);
    } else if (activateButton) {
      await activateEvaluationBaseline(activateButton.dataset.activateBaseline);
    }
  } catch (error) {
    showToast(error.message, "error");
  }
});
agentsOverlay.addEventListener("click", (event) => {
  if (event.target === agentsOverlay) closeAgentStudio();
});
mcpOverlay.addEventListener("click", (event) => {
  if (event.target === mcpOverlay) closeMcpCenter();
});
$("#mcp-transport").addEventListener("change", (event) => {
  const isStdio = event.target.value === "stdio";
  $("#mcp-command-field").hidden = !isStdio;
  $("#mcp-args-field").hidden = !isStdio;
  $("#mcp-url-field").hidden = isStdio;
});
mcpServerForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const transport = $("#mcp-transport").value;
    const payload = {
      name: $("#mcp-name").value.trim(),
      transport,
      command: transport === "stdio" ? $("#mcp-command").value.trim() : null,
      args: transport === "stdio"
        ? $("#mcp-args").value.split("\n").map((item) => item.trim()).filter(Boolean)
        : [],
      url: transport === "streamable_http" ? $("#mcp-url").value.trim() : null,
      env: parseJsonObject("#mcp-env", "环境变量"),
      headers: parseJsonObject("#mcp-headers", "请求头"),
      confirmation_policy: $("#mcp-confirmation-policy").value,
    };
    const response = await fetch("/api/mcp/servers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "创建 MCP 服务失败");
    mcpServerForm.reset();
    $("#mcp-transport").dispatchEvent(new Event("change"));
    showToast("MCP 服务配置已创建，请手动连接", "success");
    await loadMcpServers();
  } catch (error) {
    showToast(error.message, "error");
  }
});
mcpServerList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-mcp-action]");
  if (!button) return;
  const action = button.dataset.mcpAction;
  if (action === "delete" && !window.confirm("确认删除这个 MCP 服务及其工具配置？")) return;
  button.disabled = true;
  try {
    const baseUrl = `/api/mcp/servers/${encodeURIComponent(button.dataset.serverId)}`;
    const response = await fetch(
      action === "delete" ? baseUrl : `${baseUrl}/${action}`,
      { method: action === "delete" ? "DELETE" : "POST" },
    );
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || `MCP ${action} 操作失败`);
    await loadMcpServers();
  } catch (error) {
    showToast(error.message, "error");
    button.disabled = false;
  }
});
mcpServerList.addEventListener("change", async (event) => {
  const input = event.target.closest("[data-mcp-tool]");
  if (!input) return;
  try {
    const response = await fetch(
      `/api/mcp/servers/${encodeURIComponent(input.dataset.serverId)}/tools/${encodeURIComponent(input.dataset.mcpTool)}`,
      {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ enabled: input.checked }),
      },
    );
    if (!response.ok) throw new Error("更新 MCP 工具状态失败");
  } catch (error) {
    input.checked = !input.checked;
    showToast(error.message, "error");
  }
});
agentDraftForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const request = $("#agent-request").value.trim();
  if (!request) return;
  const button = agentDraftForm.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "正在生成…";
  try {
    await generateAgentDraft(request);
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "生成配置草案";
  }
});
agentConfirmForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await confirmAgentDraft();
  } catch (error) {
    showToast(error.message, "error");
  }
});
$("#btn-cancel-agent-draft").addEventListener("click", () => {
  agentConfirmForm.hidden = true;
});
customAgentList.addEventListener("click", async (event) => {
  const statusButton = event.target.closest("[data-agent-status]");
  const deleteButton = event.target.closest("[data-delete-agent]");
  try {
    if (statusButton) {
      const response = await fetch(`/api/agents/${encodeURIComponent(statusButton.dataset.agentStatus)}/status`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ status: statusButton.dataset.nextStatus }),
      });
      if (!response.ok) throw new Error("更新 Agent 状态失败");
      await loadCustomAgents();
    } else if (deleteButton && window.confirm("确认删除这个自定义 Agent？")) {
      const response = await fetch(`/api/agents/${encodeURIComponent(deleteButton.dataset.deleteAgent)}`, {
        method: "DELETE",
      });
      if (!response.ok) throw new Error("删除 Agent 失败");
      await loadCustomAgents();
    }
  } catch (error) {
    showToast(error.message, "error");
  }
});

uploadInput.addEventListener("change", () => {
  if (uploadInput.files.length > 0) {
    uploadFiles(uploadInput.files);
    uploadInput.value = "";
  }
});
chatImageInput.addEventListener("change", async () => {
  if (chatImageInput.files.length > 0) {
    try {
      await uploadChatImages(chatImageInput.files);
    } catch (error) {
      showToast(error.message, "error");
    }
    chatImageInput.value = "";
  }
});
attachmentTray.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-remove-attachment]");
  if (!button) return;
  try {
    await removePendingAttachment(button.dataset.removeAttachment);
  } catch (error) {
    showToast(error.message, "error");
  }
});
recordButton.addEventListener("click", toggleRecording);

$("#btn-clear").addEventListener("click", clearSession);
$("#btn-new-chat").addEventListener("click", createNewSession);
sidebarToggle.addEventListener("click", toggleSidebar);
sidebarBackdrop.addEventListener("click", () => {
  if (sidebarVisible) toggleSidebar();
});

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && sidebarVisible && window.matchMedia("(max-width: 900px)").matches) {
    toggleSidebar();
  }
});

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

// ── Agent 模式切换 ────────────────────────────────────────────

const modeToggle = document.getElementById("btn-mode-toggle");
const modeLabel = document.getElementById("mode-label");

if (modeToggle) {
  updateModeUI();
  modeToggle.addEventListener("click", () => {
    agentMode = agentMode === "auto" ? "simple" : "auto";
    localStorage.setItem("agentkb_agent_mode", agentMode);
    updateModeUI();
    showToast(
      agentMode === "auto" ? "已切换到多 Agent 协作模式" : "已切换到单 Agent 模式",
      "info", 2000
    );
  });
}

function updateModeUI() {
  if (modeToggle && modeLabel) {
    if (agentMode === "auto") {
      modeLabel.textContent = "多 Agent";
      modeToggle.classList.add("active");
    } else {
      modeLabel.textContent = "单 Agent";
      modeToggle.classList.remove("active");
    }
  }
}

// ── 启动 ──────────────────────────────────────────────────────
init();
