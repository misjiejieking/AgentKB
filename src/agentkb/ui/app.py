"""Gradio UI 构建与事件绑定。"""

from __future__ import annotations

import uuid
from pathlib import Path

import gradio as gr

from agentkb.agent.graph import AgentGraph
from agentkb.knowledge.loader import FileLoader
from agentkb.knowledge.splitter import TextSplitter
from agentkb.knowledge.embedder import get_embedder
from agentkb.knowledge.vector_store import get_vector_store
from agentkb.session.manager import SessionManager
from agentkb.storage.database import get_db
from agentkb.storage.models import new_id
from agentkb.config.settings import Settings


def build_ui(graph: AgentGraph | None = None) -> gr.Blocks:
    """构建 Gradio 界面并绑定所有事件处理器。"""
    cfg = Settings.load()
    _graph = graph or AgentGraph()
    _session_mgr = SessionManager()
    _file_loader = FileLoader()
    _text_splitter = TextSplitter(
        chunk_size=cfg.knowledge_chunk_size,
        chunk_overlap=cfg.knowledge_chunk_overlap,
    )

    with gr.Blocks(title="Knowledge Agent") as app:
        session_id = gr.State(value=new_id())

        gr.Markdown("# Knowledge Agent")

        chatbot = gr.Chatbot(
            label="对话",
            height=550,
            render_markdown=True,
        )

        with gr.Row():
            msg_input = gr.Textbox(
                label="输入你的问题",
                placeholder="输入问题，或上传文件后提问……",
                lines=3,
                scale=6,
            )
            with gr.Column(scale=1, min_width=120):
                upload_btn = gr.UploadButton(
                    "上传文件",
                    file_types=[".md", ".txt"],
                    file_count="multiple",
                    variant="secondary",
                )
                clear_btn = gr.Button("清空会话", variant="stop", size="sm")

        # ── 发送消息 ──────────────────────────────────────────

        async def on_send(message: str, history: list, sid: str):
            """用户发送消息：流式调用 Agent 并逐 token 更新聊天记录。"""
            if not message.strip():
                yield history, sid
                return

            # 确保会话存在
            _session_mgr.ensure_session(sid)

            # 保存用户消息
            _session_mgr.save_message(sid, "human", message)

            # 追加用户消息到聊天记录
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            yield history, sid

            accumulated = ""
            tool_status_lines: list[str] = []

            async for event in _graph.stream(
                user_input=message,
                session_id=sid,
                thread_id=sid,
            ):
                match event["type"]:
                    case "token":
                        accumulated += event["content"]
                        history[-1]["content"] = accumulated + _render_tool_status(tool_status_lines)
                        yield history, sid

                    case "tool_start":
                        tool_status_lines.append(f"🔍 正在调用 **{event['name']}**……")
                        history[-1]["content"] = accumulated + _render_tool_status(tool_status_lines)
                        yield history, sid

                    case "tool_end":
                        tool_status_lines = [
                            line.replace("🔍", "✅") for line in tool_status_lines
                        ]
                        history[-1]["content"] = accumulated + _render_tool_status(tool_status_lines)
                        yield history, sid

                    case "done":
                        pass

                    case "error":
                        history[-1]["content"] = accumulated + f"\n\n❌ {event['message']}"
                        yield history, sid
                        return

            # 最终清理：只显示正文
            history[-1]["content"] = accumulated
            yield history, sid

            # 保存助手消息
            _session_mgr.save_message(sid, "ai", accumulated)

        msg_input.submit(
            on_send,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, session_id],
        ).then(lambda: "", outputs=[msg_input])

        # ── 清空会话 ──────────────────────────────────────────

        def on_clear(sid: str):
            _session_mgr.clear_session(sid)
            new_sid = new_id()
            return [], new_sid

        clear_btn.click(
            on_clear,
            inputs=[session_id],
            outputs=[chatbot, session_id],
        )

        # ── 上传文件 ──────────────────────────────────────────

        async def on_upload(files: list | None, sid: str):
            """处理文件上传：保存→切块→向量化→存入 Qdrant。"""
            if not files:
                return gr.update()

            _session_mgr.ensure_session(sid)

            embedder = get_embedder(
                model_name=cfg.embedding_model_name,
                device=cfg.embedding_device,
                normalize=cfg.embedding_normalize,
                batch_size=cfg.embedding_batch_size,
            )
            vector_store = get_vector_store(
                path=cfg.qdrant_path,
                collection_name=cfg.qdrant_collection_name,
                vector_size=cfg.embedding_dimension,
            )
            db = get_db()

            results = []
            for file_obj in files:
                try:
                    # 文件可能是临时路径或文件对象
                    src_path = Path(file_obj.name) if hasattr(file_obj, "name") else Path(str(file_obj))
                    saved_path = _file_loader.save(str(src_path))
                    docs = _file_loader.load(saved_path)
                    chunks = _text_splitter.split(docs)
                    texts = [c.page_content for c in chunks]
                    embeddings = embedder.embed_documents(texts)

                    file_id = new_id()
                    points = []
                    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                        points.append({
                            "id": uuid.uuid4().hex,
                            "vector": emb,
                            "payload": {
                                "file_id": file_id,
                                "filename": chunk.metadata.get("source", src_path.name),
                                "chunk_index": i,
                                "content": chunk.page_content,
                            },
                        })

                    vector_store.upsert(points)
                    db.add_knowledge_file(
                        file_id=file_id,
                        filename=saved_path.name,
                        filepath=str(saved_path),
                        file_size=saved_path.stat().st_size,
                        chunk_count=len(chunks),
                    )
                    results.append(f"✅ {saved_path.name}（{len(chunks)} 个块）")
                except Exception as e:
                    results.append(f"❌ {Path(file_obj.name).name if hasattr(file_obj, 'name') else file_obj}: {e}")

            return gr.update(value=("\n".join(results) if results else "上传完成"))

        upload_btn.upload(
            on_upload,
            inputs=[upload_btn, session_id],
            outputs=[upload_btn],
        )

    return app


def _render_tool_status(lines: list[str]) -> str:
    """将工具状态行渲染为聊天框内的状态提示。"""
    if not lines:
        return ""
    return "\n\n<div class='tool-status'>" + "<br>".join(lines) + "</div>"
