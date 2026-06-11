"""AgentKB REST API 路由——聊天流式传输、文件上传、会话管理。"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextvars import Context
from pathlib import Path
from typing import AsyncGenerator, Literal

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, model_validator
from loguru import logger

from agentkb.api.deps import get_graph, get_multi_agent_graph, get_settings, get_session_mgr
from agentkb.knowledge.cache import get_cache
from agentkb.knowledge.embedder import get_embedder
from agentkb.knowledge.loader import FileLoader
from agentkb.knowledge.graph import (
    cancel_knowledge_graph_index,
    schedule_knowledge_graph_index,
)
from agentkb.knowledge.splitter import TextSplitter
from agentkb.memory.context import SessionSummaryService
from agentkb.multimodal.transcription import TranscriptionService
from agentkb.multimodal.vision import VisionService, detect_image_media_type
from agentkb.storage.pg_database import get_db
from agentkb.storage.models import new_id

router = APIRouter()


def _create_upload_path(filename: str | None, upload_dir: Path) -> tuple[str, Path]:
    """生成不可由客户端控制的保存路径，并保留原始文件名用于展示。"""
    original_name = Path((filename or "uploaded_file").replace("\\", "/")).name
    suffix = Path(original_name).suffix.lower()
    root = upload_dir.resolve()
    saved_path = (root / f"{uuid.uuid4().hex}{suffix}").resolve()
    saved_path.relative_to(root)
    return original_name, saved_path


def _validate_upload(
    filename: str,
    content: bytes,
    supported_extensions: list[str],
    max_file_size_mb: int,
) -> None:
    """在落盘前校验扩展名和文件大小。"""
    suffix = Path(filename).suffix.lower()
    allowed = {extension.lower() for extension in supported_extensions}
    if suffix not in allowed:
        raise ValueError(f"不支持的文件类型: {suffix or '无扩展名'}")

    max_bytes = max_file_size_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise ValueError(f"文件超过 {max_file_size_mb} MB 限制")


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


TERMINAL_RUN_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
_run_tasks: dict[str, asyncio.Task] = {}
_summary_tasks: set[asyncio.Task] = set()


def _schedule_session_summary(session_id: str) -> None:
    """后台增量摘要，不延长当前回答的 SSE 生命周期。"""
    from agentkb.config.settings import Settings

    async def refresh() -> None:
        try:
            await SessionSummaryService().refresh(
                session_id,
                max_turns=Settings.load().memory_working_max_turns,
            )
        except Exception as exc:
            logger.warning(f"会话摘要更新失败: session_id={session_id}, error={exc}")

    task = asyncio.create_task(refresh(), context=Context())
    _summary_tasks.add(task)
    task.add_done_callback(_summary_tasks.discard)


class RunEventSink:
    """合并高频 Token，并将事件顺序写入 PostgreSQL。"""

    def __init__(self, db, run_id: str) -> None:
        self._db = db
        self._run_id = run_id
        self._token_buffer = ""
        self._last_flush_at = 0.0

    async def emit(self, event: dict) -> None:
        if event.get("type") == "token":
            self._token_buffer += str(event.get("content", ""))
            now = time.monotonic()
            if len(self._token_buffer) < 64 and now - self._last_flush_at < 0.05:
                return
            await self.flush_tokens()
            return

        await self.flush_tokens()
        await asyncio.to_thread(
            self._db.append_run_event,
            self._run_id,
            event,
        )

    async def flush_tokens(self) -> None:
        if not self._token_buffer:
            return
        content = self._token_buffer
        self._token_buffer = ""
        await asyncio.to_thread(
            self._db.append_run_event,
            self._run_id,
            {"type": "token", "content": content},
        )
        self._last_flush_at = time.monotonic()


# ══════════════════════════════════════════════════════════════
#  请求模型
# ══════════════════════════════════════════════════════════════

class ChatRequest(BaseModel):
    message: str = ""
    session_id: str = Field(default="default")
    attachment_ids: list[str] = Field(default_factory=list, max_length=4)
    mode: Literal["auto", "simple"] = Field(
        default="auto",
        description="Agent模式: auto=多Agent协作, simple=单Agent",
    )

    @model_validator(mode="after")
    def validate_content(self):
        if not self.message.strip() and not self.attachment_ids:
            raise ValueError("消息和图片附件不能同时为空")
        if len(set(self.attachment_ids)) != len(self.attachment_ids):
            raise ValueError("图片附件不能重复")
        return self


class ClearSessionRequest(BaseModel):
    session_id: str


class ToolApprovalRequest(BaseModel):
    approved: bool


# ══════════════════════════════════════════════════════════════
#  聊天 SSE（POST 新消息 / GET 断点续传）
# ══════════════════════════════════════════════════════════════

@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """发起新对话，启动 LLM 后台任务。支持多 Agent 模式。"""
    db = get_db()
    active_run = await asyncio.to_thread(db.get_active_run, req.session_id)
    if active_run:
        raise HTTPException(
            status_code=409,
            detail="当前会话仍在生成回复，请等待完成后再发送",
        )

    graph = get_graph()
    multi_agent_graph = get_multi_agent_graph()

    run_id = new_id()
    human_msg_id = new_id()
    ai_msg_id = new_id()
    message = req.message.strip() or "请分析这些图片。"
    req = req.model_copy(update={"message": message})

    try:
        await asyncio.to_thread(
            db.create_chat_run,
            run_id=run_id,
            session_id=req.session_id,
            human_message_id=human_msg_id,
            ai_message_id=ai_msg_id,
            message=req.message,
            mode=req.mode,
            attachment_ids=req.attachment_ids,
        )
    except Exception:
        active_run = await asyncio.to_thread(db.get_active_run, req.session_id)
        if active_run:
            raise HTTPException(
                status_code=409,
                detail="当前会话仍在生成回复，请等待完成后再发送",
            )
        raise

    task = asyncio.create_task(
        _execute_chat_run(
            req=req,
            run_id=run_id,
            ai_msg_id=ai_msg_id,
            graph=graph,
            multi_agent_graph=multi_agent_graph,
            db=db,
        ),
        context=Context(),
    )
    _run_tasks[run_id] = task

    def clear_task(completed_task: asyncio.Task) -> None:
        if _run_tasks.get(run_id) is completed_task:
            _run_tasks.pop(run_id, None)

    task.add_done_callback(clear_task)

    return _run_stream_response(db, run_id)


@router.get("/chat/stream/{session_id}")
async def chat_resume(session_id: str, request: Request):
    """断点续传：根据 Last-Event-ID 补发漏掉的事件，继续接收新事件。"""
    db = get_db()
    run = await asyncio.to_thread(db.get_active_run, session_id)
    if run is None:
        run = await asyncio.to_thread(db.get_latest_run, session_id)
    if run is None:
        async def done():
            yield f"id: 0\ndata: {json.dumps({'type': 'done', 'session_id': session_id}, ensure_ascii=False)}\n\n"
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

    return _run_stream_response(db, run["id"], last_id)


@router.post("/tool-approvals/{approval_id}/decision")
async def decide_tool_approval(
    approval_id: str,
    req: ToolApprovalRequest,
):
    """记录人工决策并从 PostgreSQL Checkpoint 恢复暂停的图。"""
    db = get_db()
    approval = await asyncio.to_thread(db.get_tool_approval, approval_id)
    if approval is None:
        raise HTTPException(status_code=404, detail="审批记录不存在")

    run = await asyncio.to_thread(db.get_run, approval["run_id"])
    if run is None:
        raise HTTPException(status_code=404, detail="Agent Run 不存在")
    if run["status"] != "waiting_approval":
        raise HTTPException(status_code=409, detail="当前 Run 不在等待审批状态")
    if run["id"] in _run_tasks:
        raise HTTPException(status_code=409, detail="当前 Run 已在恢复执行")

    decided = await asyncio.to_thread(
        db.decide_tool_approval,
        approval_id,
        req.approved,
    )
    if decided is None:
        raise HTTPException(status_code=409, detail="审批已被其他请求处理")
    claimed = await asyncio.to_thread(db.claim_waiting_run, run["id"])
    if not claimed:
        raise HTTPException(status_code=409, detail="当前 Run 已被其他请求恢复")

    chat_request = ChatRequest(
        message=run["user_input"],
        session_id=run["session_id"],
        mode=run["mode"],
    )
    task = asyncio.create_task(
        _execute_chat_run(
            req=chat_request,
            run_id=run["id"],
            ai_msg_id=run["ai_message_id"],
            graph=get_graph(),
            multi_agent_graph=get_multi_agent_graph(),
            db=db,
            resume={
                "approved": req.approved,
                "approval_id": approval_id,
            },
        ),
        context=Context(),
    )
    _run_tasks[run["id"]] = task
    task.add_done_callback(
        lambda completed: _run_tasks.pop(run["id"], None)
        if _run_tasks.get(run["id"]) is completed
        else None
    )
    return {
        "approval_id": approval_id,
        "status": "approved" if req.approved else "rejected",
        "run_id": run["id"],
    }


# ── 文件上传与向量化 ────────────────────────────────────────────


@router.post("/chat/attachments")
async def upload_chat_attachment(
    session_id: str,
    file: UploadFile = File(...),
):
    """上传待随消息发送的图片附件。"""
    cfg = get_settings()
    max_bytes = cfg.vision_max_image_size_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"图片超过 {cfg.vision_max_image_size_mb} MB 限制",
        )
    try:
        media_type = detect_image_media_type(content)
        original_name, saved_path = _create_upload_path(
            file.filename,
            Path("data/uploads/chat"),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    saved_path.parent.mkdir(parents=True, exist_ok=True)
    saved_path.write_bytes(content)
    attachment_id = new_id()
    try:
        attachment = await asyncio.to_thread(
            get_db().create_chat_attachment,
            attachment_id=attachment_id,
            session_id=session_id,
            original_name=original_name,
            filepath=str(saved_path),
            media_type=media_type,
            file_size=len(content),
        )
    except Exception:
        saved_path.unlink(missing_ok=True)
        raise
    return {
        "id": attachment["id"],
        "name": attachment["original_name"],
        "media_type": attachment["media_type"],
        "url": f"/api/chat/attachments/{attachment['id']}",
    }


@router.get("/chat/attachments/{attachment_id}")
async def get_chat_attachment_file(attachment_id: str):
    """返回已持久化的聊天图片。"""
    attachment = await asyncio.to_thread(
        get_db().get_chat_attachment,
        attachment_id,
    )
    if not attachment:
        raise HTTPException(status_code=404, detail="图片附件不存在")
    root = Path("data/uploads/chat").resolve()
    path = Path(attachment["filepath"]).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail="图片附件路径非法") from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="图片附件文件不存在")
    return FileResponse(path, media_type=attachment["media_type"])


@router.delete("/chat/attachments/{attachment_id}")
async def delete_chat_attachment(attachment_id: str, session_id: str):
    """删除尚未随消息发送的图片附件。"""
    filepath = await asyncio.to_thread(
        get_db().delete_unclaimed_chat_attachment,
        attachment_id,
        session_id,
    )
    if filepath is None:
        raise HTTPException(status_code=404, detail="待发送附件不存在")
    Path(filepath).unlink(missing_ok=True)
    return {"deleted": True}


@router.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    """将浏览器录制的语音转写为文本。"""
    cfg = get_settings()
    max_bytes = cfg.transcription_max_audio_size_mb * 1024 * 1024
    content = await file.read(max_bytes + 1)
    try:
        text = await asyncio.to_thread(
            TranscriptionService(cfg).transcribe,
            content,
            filename=Path(file.filename or "recording.webm").name,
            media_type=file.content_type or "audio/webm",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(f"语音转写失败: {exc}", exc_info=True)
        raise HTTPException(status_code=502, detail=f"语音转写失败: {exc}") from exc
    return {"text": text}


@router.get("/capabilities")
def get_capabilities():
    """返回前端可用的多模态能力。"""
    cfg = get_settings()
    return {
        "vision": {
            "enabled": cfg.vision_enabled,
            "provider": cfg.vision_provider,
            "model": cfg.vision_model_name,
        },
        "transcription": {
            "enabled": cfg.transcription_enabled,
            "model": cfg.transcription_model_name,
        },
    }


@router.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    """上传知识文件并完成切块→向量化→入库全流程。"""
    cfg = get_settings()
    loader = FileLoader(cfg)
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
        saved_path: Path | None = None
        try:
            max_bytes = cfg.knowledge_max_file_size_mb * 1024 * 1024
            content = await file.read(max_bytes + 1)
            original_name, saved_path = _create_upload_path(
                file.filename,
                Path("data/uploads"),
            )
            _validate_upload(
                original_name,
                content,
                cfg.knowledge_supported_extensions,
                cfg.knowledge_max_file_size_mb,
            )
            ext = saved_path.suffix
            saved_path.parent.mkdir(parents=True, exist_ok=True)
            saved_path.write_bytes(content)

            docs = await asyncio.to_thread(loader.load, str(saved_path))
            chunks = await asyncio.to_thread(splitter.split, docs)
            # 文本清洗：合并多余空白、去除首尾空格、过滤纯标点/数字的无效 chunk
            for c in chunks:
                c.page_content = _clean_text(c.page_content)
            chunks = [c for c in chunks if len(c.page_content) >= 10]
            if not chunks:
                raise ValueError("文件解析后没有可入库的有效内容")
            texts = [c.page_content for c in chunks]
            embeddings = await asyncio.to_thread(embedder.embed_documents, texts)

            file_id = new_id()
            chunk_records = []
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                chunk_records.append({
                    "file_id": file_id,
                    "chunk_index": i,
                    "content": chunk.page_content,
                    "embedding": emb,
                    "chunk_metadata": {
                        "filename": original_name,
                        **chunk.metadata,
                    },
                })

            await asyncio.to_thread(
                db.add_knowledge_file,
                file_id=file_id,
                filename=original_name,
                filepath=str(saved_path),
                file_size=len(content),
                chunk_count=len(chunks),
                file_type=ext.lstrip(".") or "unknown",
            )
            await asyncio.to_thread(db.upsert_chunks, chunk_records)
            if cfg.knowledge_graph_enabled:
                schedule_knowledge_graph_index(file_id)
                graph_status = "queued"
            else:
                db.update_knowledge_graph_status(file_id, "disabled")
                graph_status = "disabled"
            results.append({
                "status": "ok",
                "filename": original_name,
                "file_id": file_id,
                "chunks": len(chunks),
                "graph_status": graph_status,
            })
        except Exception as exc:
            if saved_path is not None:
                saved_path.unlink(missing_ok=True)
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
async def delete_file(file_id: str):
    """软删除知识文件并从向量库移除。"""
    db = get_db()
    await cancel_knowledge_graph_index(file_id)
    deleted = await asyncio.to_thread(db.delete_knowledge_file, file_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="知识文件不存在")
    deleted_count = await asyncio.to_thread(
        db.delete_chunks_by_file_id,
        file_id,
    )
    get_cache().invalidate()
    return {"deleted": True, "file_id": file_id, "vectors_removed": deleted_count}


@router.get("/knowledge/graph")
def query_knowledge_graph(query: str = "", limit: int = 30):
    """查询知识图谱相邻子图和索引统计。"""
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=422, detail="limit 必须在 1 到 100 之间")
    db = get_db()
    graph = db.search_knowledge_graph(query, limit) if query.strip() else {
        "nodes": [],
        "edges": [],
    }
    return {
        **graph,
        "stats": db.get_knowledge_graph_stats(),
    }


@router.post("/knowledge/graph/reindex/{file_id}")
async def reindex_knowledge_graph(file_id: str):
    """重新构建指定文件的知识图谱索引。"""
    cfg = get_settings()
    if not cfg.knowledge_graph_enabled:
        raise HTTPException(status_code=409, detail="知识图谱功能未启用")

    db = get_db()
    file = await asyncio.to_thread(db.get_knowledge_file, file_id)
    if file is None or file["status"] != "active":
        raise HTTPException(status_code=404, detail="知识文件不存在")
    if file["graph_status"] == "processing":
        raise HTTPException(status_code=409, detail="该文件正在构建知识图谱")

    queued = await asyncio.to_thread(db.queue_knowledge_graph_index, file_id)
    if not queued:
        raise HTTPException(status_code=404, detail="知识文件不存在")
    schedule_knowledge_graph_index(file_id)
    return {"file_id": file_id, "graph_status": "queued"}


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
    run = get_db().get_active_run(session_id)
    return {
        "session_id": session_id,
        "title": session_mgr.get_session_title(session_id),
        "message_count": len(messages),
        "is_generating": run is not None,
        "run_id": run["id"] if run else None,
        "run_status": run["status"] if run else None,
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
    if db.get_active_run(session_id):
        raise HTTPException(status_code=409, detail="当前会话仍在生成回复，不能删除")
    attachments = db.get_all_session_attachments(session_id)
    db.delete_session(session_id)
    for attachment in attachments:
        Path(attachment["filepath"]).unlink(missing_ok=True)
    return {"deleted": True, "session_id": session_id}


@router.post("/session/clear")
def clear_session(req: ClearSessionRequest):
    """清空指定会话的消息。"""
    if get_db().get_active_run(req.session_id):
        raise HTTPException(status_code=409, detail="当前会话仍在生成回复，不能清空")
    db = get_db()
    attachments = db.get_all_session_attachments(req.session_id)
    session_mgr = get_session_mgr()
    session_mgr.clear_session(req.session_id)
    for attachment in attachments:
        Path(attachment["filepath"]).unlink(missing_ok=True)
    new_sid = new_id()
    return {"session_id": new_sid, "title": "New Chat"}


class FeedbackRequest(BaseModel):
    session_id: str = ""
    rating: Literal["up", "down"]
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
    from agentkb.observability.metrics import get_metrics

    get_metrics().record_feedback(req.rating)
    return {"ok": True}


# ── 链路追踪 ────────────────────────────────────────────────────


@router.get("/trace/{trace_id}")
def get_trace(trace_id: str):
    """返回指定 trace_id 的完整调用链路。"""
    from agentkb.observability.tracer import load_trace

    trace = load_trace(trace_id)
    if trace is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "trace not found"}, status_code=404)
    return trace


# ── 指标 ──────────────────────────────────────────────────────────


@router.get("/metrics")
def get_metrics():
    """Prometheus 格式的指标端点。"""
    from fastapi.responses import PlainTextResponse
    from agentkb.observability.metrics import get_metrics as gm

    text = gm().to_prometheus_text()
    return PlainTextResponse(text, media_type="text/plain")


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
#  聊天 Run 与 SSE 辅助函数
# ══════════════════════════════════════════════════════════════

def _run_stream_response(db, run_id: str, last_event_id: int = 0) -> StreamingResponse:
    """创建从 PostgreSQL 顺序回放事件的 SSE 响应。"""
    return StreamingResponse(
        _iter_persisted_events(db, run_id, last_event_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _iter_persisted_events(
    db,
    run_id: str,
    last_event_id: int = 0,
) -> AsyncGenerator[str, None]:
    """轮询持久化事件；连接断开不会影响后台 Run。"""
    cursor = max(last_event_id, 0)
    last_heartbeat = time.monotonic()

    try:
        while True:
            events = await asyncio.to_thread(
                db.get_run_events,
                run_id,
                cursor,
            )
            for row in events:
                cursor = int(row["event_id"])
                payload = row["payload"]
                if isinstance(payload, str):
                    payload = json.loads(payload)
                yield (
                    f"id: {cursor}\n"
                    f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                )

            run = await asyncio.to_thread(db.get_run, run_id)
            if run is None or (
                run["status"] in TERMINAL_RUN_STATUSES and not events
            ):
                break

            now = time.monotonic()
            if now - last_heartbeat >= 15:
                yield ": keep-alive\n\n"
                last_heartbeat = now
            await asyncio.sleep(0.05)
    except asyncio.CancelledError:
        return


async def _iter_agent_events(
    req: ChatRequest,
    run_id: str,
    graph,
    multi_agent_graph,
    resume: dict | None = None,
):
    """构建数据库历史并选择单 Agent 或多 Agent 执行。"""
    from agentkb.config.settings import Settings
    from agentkb.session.manager import SessionManager

    session_mgr = SessionManager()
    attachment_context = await _prepare_attachment_context(
        req,
        get_db(),
    )
    user_input = req.message
    if attachment_context:
        user_input = f"{user_input}\n\n[图片附件分析]\n{attachment_context}"
    all_messages = await asyncio.to_thread(
        session_mgr.load_messages,
        req.session_id,
    )
    history = session_mgr.dict_to_langchain(
        [message for message in all_messages if (message.get("content") or "").strip()][:-1]
    )
    cfg = Settings.load()
    conversation_summary = await asyncio.to_thread(
        SessionSummaryService().get_summary,
        req.session_id,
    )
    thread_id = (
        f"{req.session_id}:{cfg.llm_provider}:{cfg.llm_generator_model_name}"
    )

    selected_graph = (
        multi_agent_graph
        if req.mode == "auto" and multi_agent_graph is not None
        else graph
    )
    if resume is not None and selected_graph is multi_agent_graph:
        raise RuntimeError("多 Agent 图当前没有可恢复的高风险工具节点")
    graph_name = "multi" if selected_graph is multi_agent_graph else "single"
    thread_id = f"{thread_id}:{graph_name}"
    stream = (
        selected_graph.stream(
            user_input=user_input,
            session_id=req.session_id,
            thread_id=thread_id,
            history=history,
            conversation_summary=conversation_summary,
        )
        if selected_graph is multi_agent_graph
        else selected_graph.stream(
            user_input=user_input,
            session_id=req.session_id,
            run_id=run_id,
            thread_id=thread_id,
            history=history,
            conversation_summary=conversation_summary,
            resume=resume,
        )
    )
    async for event in stream:
        yield event


async def _prepare_attachment_context(
    req: ChatRequest,
    db,
) -> str:
    """分析当前消息附件并返回稳定的文本上下文。"""
    if not req.attachment_ids:
        return ""
    attachments = await asyncio.to_thread(
        db.get_chat_attachments,
        req.attachment_ids,
        req.session_id,
    )
    if len(attachments) != len(req.attachment_ids):
        raise ValueError("图片附件不存在或不属于当前会话")

    descriptions = []
    service = VisionService(settings=get_settings())
    for attachment in attachments:
        description = str(attachment.get("description", "")).strip()
        if attachment["status"] == "analyzed" and description:
            descriptions.append(
                f"附件《{attachment['original_name']}》：\n{description}"
            )
            continue

        try:
            path = Path(attachment["filepath"])
            analysis = await asyncio.to_thread(
                service.analyze,
                path.read_bytes(),
                prompt=(
                    f"用户上传了图片《{attachment['original_name']}》。"
                    "请提取所有可见文字，并准确描述对象、布局、关系、图表数据和异常信息。"
                    "输出将提供给后续 Agent 回答用户问题。"
                ),
            )
        except Exception as exc:
            await asyncio.to_thread(
                db.update_chat_attachment_analysis,
                attachment["id"],
                status="failed",
                error=str(exc),
            )
            raise

        await asyncio.to_thread(
            db.update_chat_attachment_analysis,
            attachment["id"],
            status="analyzed",
            description=analysis.description,
        )
        descriptions.append(
            f"附件《{attachment['original_name']}》：\n{analysis.description}"
        )
    return "\n\n".join(descriptions)


async def _execute_chat_run(
    *,
    req: ChatRequest,
    run_id: str,
    ai_msg_id: str,
    graph,
    multi_agent_graph,
    db,
    resume: dict | None = None,
) -> None:
    """执行 Agent Run，并原子推进状态与事件日志。"""
    sink = RunEventSink(db, run_id)
    started_at = time.monotonic()
    accumulated = ""
    error_message = ""
    terminal_event_seen = False
    last_message_persist_at = time.monotonic()
    first_token_ms: float | None = None
    event_count = 0
    run_status = "failed"
    waiting_approval = False

    try:
        await asyncio.to_thread(db.update_run_status, run_id, "running")

        async for event in _iter_agent_events(
            req,
            run_id,
            graph,
            multi_agent_graph,
            resume,
        ):
            event_count += 1
            event_type = event.get("type")
            if event_type == "token":
                if first_token_ms is None:
                    first_token_ms = (time.monotonic() - started_at) * 1000
                accumulated += str(event.get("content", ""))
                now = time.monotonic()
                if now - last_message_persist_at >= 0.5:
                    await asyncio.to_thread(
                        db.update_message_content,
                        ai_msg_id,
                        accumulated,
                    )
                    last_message_persist_at = now
            elif event_type == "error":
                error_message = str(event.get("message", "Agent 执行失败"))
                terminal_event_seen = True
            elif event_type == "done":
                terminal_event_seen = True
            elif event_type == "approval_required":
                waiting_approval = True

            await sink.emit(event)

        await sink.flush_tokens()
        if waiting_approval:
            await asyncio.to_thread(
                db.update_message_content,
                ai_msg_id,
                accumulated,
            )
            run_status = "waiting_approval"
            await asyncio.to_thread(
                db.update_run_status,
                run_id,
                run_status,
            )
            return

        if not terminal_event_seen:
            await sink.emit({"type": "done", "session_id": req.session_id})

        final_content = accumulated or error_message
        await asyncio.to_thread(
            db.update_message_content,
            ai_msg_id,
            final_content,
        )
        status = "failed" if error_message else "completed"
        run_status = status
        await asyncio.to_thread(
            db.update_run_status,
            run_id,
            status,
            error_message,
        )
        if status == "completed":
            _schedule_session_summary(req.session_id)
    except Exception as exc:
        logger.error(f"Agent Run 执行失败: run_id={run_id}, error={exc}", exc_info=True)
        error_message = str(exc)
        await sink.flush_tokens()
        await sink.emit({"type": "error", "message": error_message})
        await asyncio.to_thread(
            db.update_message_content,
            ai_msg_id,
            accumulated or f"处理请求时出错：{error_message}",
        )
        await asyncio.to_thread(
            db.update_run_status,
            run_id,
            "failed",
            error_message,
        )
    finally:
        from agentkb.observability.metrics import get_metrics

        get_metrics().record_agent_run(
            mode=req.mode,
            status=run_status,
            elapsed_ms=(time.monotonic() - started_at) * 1000,
            first_token_ms=first_token_ms,
            event_count=event_count,
        )
