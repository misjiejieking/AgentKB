"""AgentKB REST API 路由——聊天流式传输、文件上传、会话管理。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from loguru import logger

from agentkb.api.deps import get_graph, get_multi_agent_graph, get_settings, get_session_mgr
from agentkb.knowledge.cache import get_cache
from agentkb.knowledge.embedder import get_embedder
from agentkb.knowledge.loader import FileLoader
from agentkb.knowledge.splitter import TextSplitter
from agentkb.storage.pg_database import get_db
from agentkb.storage.models import new_id

router = APIRouter()


def _clean_text(text: str) -> str:
    """文本清洗：去中文字间空格、合并空白、过滤无意义短 chunk。"""
    import re
    # 移除中文字间的空格（PDF 提取常见问题：员 工 手 册 → 员工手册）
    text = re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", text)
    # 合并连续空白（空格、制表符、全角空格）
    text = re.sub(r"[　\t ]+", " ", text)
    text = text.strip()
    # 移除纯数字/标点行
    text = re.sub(r"^\d+[\.\)、]?\s*$", "", text, flags=re.MULTILINE)
    # 合并多余换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════
#  SessionStream — 解耦「LLM 执行」与「前端连接」
# ══════════════════════════════════════════════════════════════

class SessionStream:
    """单会话的事件流——缓存事件（带 id），支持断点续传。

    - _events: 索引 list，_events[0] 为事件 1
    - _done: 生成是否完成
    - _ai_msg_id: DB 中 AI 消息的 ID
    - _subscribers: 当前连接的 SSE 队列（广播用）
    """

    MAX_EVENTS = 2000

    def __init__(self, session_id: str, ai_msg_id: str):
        self.session_id = session_id
        self.ai_msg_id = ai_msg_id
        self._events: list[dict] = []
        self._done = False
        self._subscribers: list[asyncio.Queue] = []

    def push(self, event: dict) -> None:
        if len(self._events) >= self.MAX_EVENTS:
            self._events.pop(0)
        self._events.append(event)
        # 广播给所有订阅者
        for q in self._subscribers:
            if not q.full():
                q.put_nowait(event)

    def finish(self) -> None:
        self._done = True
        for q in self._subscribers:
            if not q.full():
                q.put_nowait(None)  # 结束信号

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    def events_since(self, last_id: int) -> list[dict]:
        """返回 last_id 之后的所有事件（last_id=0 表示从头）。"""
        if last_id <= 0:
            return list(self._events)
        return self._events[last_id:]  # last_id 是 1-based 索引


# 全局流注册表
_streams: dict[str, SessionStream] = {}


def _get_or_create_stream(session_id: str, ai_msg_id: str = "") -> SessionStream:
    if session_id not in _streams or _streams[session_id]._done:
        _streams[session_id] = SessionStream(session_id, ai_msg_id)
    return _streams[session_id]


# ══════════════════════════════════════════════════════════════
#  请求模型
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default")
    mode: str = Field(default="auto", description="Agent模式: auto=多Agent协作, simple=单Agent")


class ClearSessionRequest(BaseModel):
    session_id: str


# ══════════════════════════════════════════════════════════════
#  聊天 SSE（POST 新消息 / GET 断点续传）
# ══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发起新对话，启动 LLM 后台任务。支持多 Agent 模式。"""
    session_mgr = get_session_mgr()
    graph = get_graph()
    multi_agent_graph = get_multi_agent_graph()

    session_mgr.ensure_session(req.session_id)
    session_mgr.save_message(req.session_id, "human", req.message)

    db = get_db()
    ai_msg_id = new_id()
    db.add_message(msg_id=ai_msg_id, session_id=req.session_id, role="ai", content="")

    stream = _get_or_create_stream(req.session_id, ai_msg_id)
    stream.push({"type": "message_id", "message_id": ai_msg_id})

    use_multi_agent = req.mode == "auto" and multi_agent_graph is not None

    async def run_llm():
        try:
            if use_multi_agent:
                await _run_multi_agent(req, stream, db, ai_msg_id)
            else:
                await _run_single_agent(req, graph, stream, db, ai_msg_id)
        except Exception as exc:
            logger.error(f"LLM 后台任务出错: {exc}", exc_info=True)
            stream.push({"type": "error", "message": str(exc)})
            stream.finish()

    asyncio.ensure_future(run_llm())

    return _sse_response(stream)


@router.get("/chat/stream/{session_id}")
async def chat_resume(session_id: str, request: Request):
    """断点续传：根据 Last-Event-ID 补发漏掉的事件，继续接收新事件。"""
    stream = _streams.get(session_id)
    if not stream or stream._done:
        async def done():
            yield f"id: 0\ndata: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
            if False: yield ""
        return StreamingResponse(
            done(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache"},
        )

    # 解析 Last-Event-ID
    last_id_str = request.headers.get("Last-Event-ID", "0")
    try:
        last_id = int(last_id_str)
    except ValueError:
        last_id = 0

    # last_id=0 表示首次连接，跳到最新位置，不回放历史
    if last_id == 0:
        last_id = len(stream._events)
    missed = stream.events_since(last_id)
    event_id = last_id

    q = stream.subscribe()
    try:
        async def resume_stream():
            nonlocal event_id
            # 补发漏掉的事件
            for ev in missed:
                event_id += 1
                yield f"id: {event_id}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            # 继续接收新事件
            try:
                while True:
                    ev = await q.get()
                    if ev is None:
                        break
                    event_id += 1
                    yield f"id: {event_id}\ndata: {json.dumps(ev, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                stream.unsubscribe(q)

        return StreamingResponse(
            resume_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception:
        stream.unsubscribe(q)
        raise


# ── 文件上传与向量化 ────────────────────────────────────────────


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传知识文件并完成切块→向量化→入库全流程。"""
    cfg = get_settings()
    loader = FileLoader()
    embedder = get_embedder(
        model_name=cfg.embedding_model_name,
        device=cfg.embedding_device,
        normalize=cfg.embedding_normalize,
        batch_size=cfg.embedding_batch_size,
    )
    splitter = TextSplitter(embedder=embedder)
    db = get_db()

    results = []
    for file in files:
        try:
            content = await file.read()
            ext = Path(file.filename or "").suffix.lower()
            saved_path = Path("data/uploads") / (file.filename or "uploaded_file")
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_bytes(content)

            docs = loader.load(str(saved_path))
            chunks = splitter.split(docs)
            # 文本清洗：合并多余空白、去除首尾空格、过滤纯标点/数字的无效 chunk
            for c in chunks:
                c.page_content = _clean_text(c.page_content)
            chunks = [c for c in chunks if len(c.page_content) >= 10]
            texts = [c.page_content for c in chunks]
            embeddings = embedder.embed_documents(texts)

            file_id = new_id()
            chunk_records = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_records.append({
                    "file_id": file_id,
                    "chunk_index": i,
                    "content": chunk.page_content,
                    "embedding": emb,
                    "chunk_metadata": {
                        "filename": chunk.metadata.get("source", file.filename),
                        "source": chunk.metadata.get("source", ""),
                    },
                })

            db.add_knowledge_file(
                file_id=file_id,
                filename=file.filename or "unknown",
                filepath=str(saved_path),
                file_size=len(content),
                chunk_count=len(chunks),
                file_type=ext.lstrip(".") or "unknown",
            )
            db.upsert_chunks(chunk_records)
            results.append({"status": "ok", "filename": file.filename, "chunks": len(chunks)})
        except Exception as exc:
            logger.error(f"File upload error: {exc}", exc_info=True)
            results.append({"status": "error", "filename": file.filename, "error": str(exc)})

    # 知识库变更 → 失效检索缓存
    if any(r["status"] == "ok" for r in results):
        get_cache().invalidate()

    return {"results": results}


# ── 知识文件管理 ───────────────────────────────────────────────


@router.get("/knowledge/files")
def list_files():
    """返回所有活跃知识文件的元数据。"""
    db = get_db()
    files = db.list_knowledge_files()
    return {"files": files}


@router.delete("/knowledge/files/{file_id}")
def delete_file(file_id: str):
    """软删除知识文件并从向量库移除。"""
    db = get_db()
    db.delete_knowledge_file(file_id)
    deleted_count = db.delete_chunks_by_file_id(file_id)
    get_cache().invalidate()
    return {"deleted": True, "file_id": file_id, "vectors_removed": deleted_count}


# ── 会话管理 ──────────────────────────────────────────────────


@router.get("/sessions")
def list_sessions():
    """返回所有会话列表，含消息数。"""
    db = get_db()
    sessions = db.list_sessions()
    return {"sessions": sessions}


@router.post("/sessions")
def create_session():
    """创建新会话并返回 session_id。"""
    session_mgr = get_session_mgr()
    sid = new_id()
    session_mgr.ensure_session(sid)
    return {"session_id": sid, "title": "New Chat"}


@router.get("/session/{session_id}")
def get_session(session_id: str):
    """获取会话信息。"""
    session_mgr = get_session_mgr()
    session_mgr.ensure_session(session_id)
    messages = session_mgr.load_messages(session_id)
    return {
        "session_id": session_id,
        "title": session_mgr.get_session_title(session_id),
        "message_count": len(messages),
    }


@router.get("/session/{session_id}/messages")
def get_session_messages(session_id: str):
    """返回指定会话的全部历史消息。"""
    session_mgr = get_session_mgr()
    session_mgr.ensure_session(session_id)
    messages = session_mgr.load_messages(session_id)
    return {"session_id": session_id, "messages": messages}


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    """删除指定会话及其所有消息。"""
    db = get_db()
    db.delete_session(session_id)
    return {"deleted": True, "session_id": session_id}


@router.post("/session/clear")
def clear_session(req: ClearSessionRequest):
    """清空指定会话的消息。"""
    session_mgr = get_session_mgr()
    session_mgr.clear_session(req.session_id)
    new_sid = new_id()
    return {"session_id": new_sid, "title": "New Chat"}


class FeedbackRequest(BaseModel):
    session_id: str = ""
    rating: str = ""
    reason: str = ""
    query: str = ""
    message_id: str = ""


@router.post("/feedback")
def submit_feedback(req: FeedbackRequest):
    """记录用户反馈。"""
    db = get_db()
    db.add_feedback(
        session_id=req.session_id,
        rating=req.rating,
        reason=req.reason,
        query=req.query,
        message_id=req.message_id,
    )
    return {"ok": True}


# ── 链路追踪 ────────────────────────────────────────────────────


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """返回指定 trace_id 的完整调用链路。"""
    from agentkb.utils.tracer import TraceRecord
    record = TraceRecord.load(trace_id)
    if record is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "trace not found"}, status_code=404)
    return record.to_dict()


# ── 指标 ──────────────────────────────────────────────────────────


@router.get("/metrics")
def get_metrics():
    """Prometheus 格式的指标端点。"""
    from fastapi.responses import PlainTextResponse
    try:
        from agentkb.observability.metrics import get_metrics as gm
        text = gm().to_prometheus_text()
        return PlainTextResponse(text, media_type="text/plain")
    except ImportError:
        return PlainTextResponse("# metrics module not available\n", media_type="text/plain")


# ── 健康检查 ──────────────────────────────────────────────────────


@router.get("/health")
def health_check():
    """健康检查端点——返回各组件状态。"""
    status = {"status": "ok", "components": {}}

    # PG 数据库
    try:
        from agentkb.storage.pg_database import get_db
        db = get_db()
        with db._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
        status["components"]["postgresql"] = "ok"
    except Exception as e:
        status["components"]["postgresql"] = f"error: {e}"
        status["status"] = "degraded"

    # LLM
    try:
        from agentkb.llm.factory import get_chat_model
        get_chat_model(streaming=True)
        status["components"]["llm"] = "ok"
    except Exception as e:
        status["components"]["llm"] = f"error: {e}"
        status["status"] = "degraded"

    # Embedder
    try:
        from agentkb.knowledge.embedder import get_embedder
        get_embedder()
        status["components"]["embedder"] = "ok"
    except Exception as e:
        status["components"]["embedder"] = f"error: {e}"
        status["status"] = "degraded"

    return status


# ══════════════════════════════════════════════════════════════
#  聊天 SSE 辅助函数
# ══════════════════════════════════════════════════════════════

def _sse_response(stream: SessionStream):
    """为 SessionStream 创建 SSE StreamingResponse。"""
    q = stream.subscribe()
    try:
        async def event_generator():
            event_id = 0
            try:
                while True:
                    event = await q.get()
                    if event is None:
                        break
                    event_id += 1
                    yield f"id: {event_id}\ndata: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                stream.unsubscribe(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception:
        stream.unsubscribe(q)
        raise


async def _run_single_agent(req: ChatRequest, graph, stream: SessionStream, db, ai_msg_id: str):
    """原有单 Agent 模式——LangGraph ReAct loop 流式执行。"""
    from agentkb.config.settings import Settings
    cfg = Settings.load()
    accumulated = ""
    token_count = 0
    # thread_id 加模型名后缀，切换 LLM 时自动隔离旧格式的对话状态
    thread_id = f"{req.session_id}:{cfg.llm_model_name}"
    async for event in graph.stream(
        user_input=req.message,
        session_id=req.session_id,
        thread_id=thread_id,
    ):
        stream.push(event)
        if event["type"] == "token":
            accumulated += event["content"]
            token_count += 1
            if token_count % 5 == 0:
                db.update_message_content(ai_msg_id, accumulated)
        elif event["type"] == "done":
            db.update_message_content(ai_msg_id, accumulated)
        elif event["type"] == "error":
            if accumulated:
                db.update_message_content(ai_msg_id, accumulated)
    stream.finish()


async def _run_multi_agent(req: ChatRequest, stream: SessionStream, db, ai_msg_id: str):
    """多 Agent 模式——LangGraph StateGraph 流式执行。

    通过 MultiAgentGraph.stream() 获取 astream_events v2 事件，
    直接推入 SessionStream。LangGraph checkpointer 自动管理对话历史。
    """
    from agentkb.api.deps import get_multi_agent_graph
    from agentkb.session.manager import SessionManager

    multi_graph = get_multi_agent_graph()
    if multi_graph is None:
        stream.push({"type": "error", "message": "MultiAgentGraph 未初始化"})
        stream.finish()
        return

    # 加载历史消息作为 LangChain 消息列表，注入到图的初始 state
    session_mgr = SessionManager()
    all_messages = session_mgr.load_messages(req.session_id)
    history_lc = session_mgr.dict_to_langchain(
        [m for m in all_messages if (m.get("content") or "").strip()][:-1]  # 排除当前 human
    )

    accumulated = ""
    try:
        async for event in multi_graph.stream(
            user_input=req.message,
            session_id=req.session_id,
            thread_id=req.session_id,
            history=history_lc,
        ):
            stream.push(event)
            if event["type"] == "token":
                accumulated += event["content"]
            elif event["type"] == "done":
                db.update_message_content(ai_msg_id, accumulated)
            elif event["type"] == "error":
                if accumulated:
                    db.update_message_content(ai_msg_id, accumulated)
        stream.finish()
    except Exception as exc:
        logger.error(f"Multi-Agent 执行异常: {exc}", exc_info=True)
        if accumulated:
            db.update_message_content(ai_msg_id, accumulated)
        stream.push({"type": "error", "message": str(exc)})
        stream.finish()
