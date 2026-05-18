/* global marked */
// ── AgentKB 前端逻辑 ──────────────────────────────────────────────

// ══════════════════════════════════════════════════════════════
//  状态
// ══════════════════════════════════════════════════════════════

let sessionId = localStorage.getItem("agentkb_session_id") || "";
let isStreaming = false;
let sidebarVisible = true;

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
  // 保留 welcome screen DOM 元素，但隐藏它
  hideWelcome();

  // 移除之前的消息（保留 welcomeScreen）
  chatContainer.querySelectorAll(".message").forEach((el) => el.remove());

  messages.forEach((m) => {
    if (m.role === "human") {
      addUserMessage(m.content);
    } else if (m.role === "ai") {
      addAssistantMessage(m.content);
    }
  });

  scrollToBottom();
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
      kbFileList.innerHTML = '<div class="kb-empty">暂无文件</div>';
      return;
    }

    kbFileList.innerHTML = files
      .map((f) => {
        const name = escapeHtml(f.filename || "unknown");
        return `
          <div class="kb-file-item" title="${name}">
            <svg class="file-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
            ${name}
          </div>`;
      })
      .join("");
  } catch (err) {
    // 静默
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

  let accumulated = "";
  let toolLines = [];
  let throttleTimer = null;

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
        if (!line.startsWith("data: ")) continue;
        try {
          const event = JSON.parse(line.slice(6));
          switch (event.type) {
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
              flushContent();
              break;

            case "done":
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

    if (throttleTimer) {
      clearTimeout(throttleTimer);
      throttleTimer = null;
    }
    assistantEl.innerHTML = renderMarkdown(accumulated);
    finalizeAssistant(assistantEl);
    scrollToBottom();

    // 更新侧边栏会话列表（标题可能变化）
    loadSessionList();
  } catch (err) {
    assistantEl.innerHTML = renderMarkdown(
      accumulated +
        '\n\n<span style="color:var(--danger)">请求失败: ' +
        escapeHtml(err.message) +
        "</span>"
    );
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
