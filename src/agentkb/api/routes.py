"""AgentKB REST API 路由——聊天流式传输、文件上传、会话管理。"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, UploadFile, File
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


# ── 请求模型 ──────────────────────────────────────────────────


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default")


class ClearSessionRequest(BaseModel):
    session_id: str


# ── 聊天 SSE 流式传输 ──────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """SSE 流式对话端点，逐 token 推送回答。"""
    session_mgr = get_session_mgr()
    graph = get_graph()

    session_mgr.ensure_session(req.session_id)
    session_mgr.save_message(req.session_id, "human", req.message)

    async def event_generator():
        accumulated = ""
        try:
            async for event in graph.stream(
                user_input=req.message,
                session_id=req.session_id,
                thread_id=req.session_id,
            ):
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                if event["type"] == "token":
                    accumulated += event["content"]
                elif event["type"] == "done":
                    session_mgr.save_message(req.session_id, "ai", accumulated)
                elif event["type"] == "error":
                    if accumulated:
                        session_mgr.save_message(req.session_id, "ai", accumulated)
        except Exception as exc:
            logger.error(f"SSE stream error: {exc}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
