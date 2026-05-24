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

from agentkb.api.deps import get_graph, get_settings, get_session_mgr
from agentkb.knowledge.embedder import get_embedder
from agentkb.knowledge.loader import FileLoader
from agentkb.knowledge.splitter import TextSplitter
from agentkb.storage.pg_database import get_db
from agentkb.storage.models import new_id

router = APIRouter()


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


class ClearSessionRequest(BaseModel):
    session_id: str


# ══════════════════════════════════════════════════════════════
#  聊天 SSE（POST 新消息 / GET 断点续传）
# ══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发起新对话，启动 LLM 后台任务。"""
    session_mgr = get_session_mgr()
    graph = get_graph()

    session_mgr.ensure_session(req.session_id)
    session_mgr.save_message(req.session_id, "human", req.message)

    db = get_db()
    ai_msg_id = new_id()
    db.add_message(msg_id=ai_msg_id, session_id=req.session_id, role="ai", content="")

    stream = _get_or_create_stream(req.session_id, ai_msg_id)

    async def run_llm():
        accumulated = ""
        token_count = 0
        try:
            async for event in graph.stream(
                user_input=req.message,
                session_id=req.session_id,
                thread_id=req.session_id,
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
        except Exception as exc:
            logger.error(f"LLM 后台任务出错: {exc}", exc_info=True)
            if accumulated:
                db.update_message_content(ai_msg_id, accumulated)
            stream.push({"type": "error", "message": str(exc)})
            stream.finish()

    asyncio.ensure_future(run_llm())

    q = stream.subscribe()
    try:
        async def new_stream():
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
            new_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    except Exception:
        stream.unsubscribe(q)
        raise


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
