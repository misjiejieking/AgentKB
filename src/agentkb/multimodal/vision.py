"""图片格式校验与视觉内容理解。"""

from __future__ import annotations

import base64
from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from agentkb.config.settings import Settings
from agentkb.llm.factory import get_vision_chat_model
from agentkb.utils.exceptions import KnowledgeBaseError

SUPPORTED_IMAGE_TYPES = {
    "image/jpeg": (b"\xff\xd8\xff",),
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/webp": (b"RIFF",),
}


def detect_image_media_type(content: bytes) -> str:
    """根据文件签名识别允许的静态图片格式。"""
    for media_type, signatures in SUPPORTED_IMAGE_TYPES.items():
        if any(content.startswith(signature) for signature in signatures):
            if media_type != "image/webp" or content[8:12] == b"WEBP":
                return media_type
    raise ValueError("仅支持 JPEG、PNG 和 WebP 图片")


@dataclass(frozen=True)
class ImageAnalysis:
    description: str
    media_type: str


class VisionService:
    """通过独立视觉模型生成可持久化的图片描述。"""

    def __init__(self, model=None, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self.model = model

    def analyze(
        self,
        content: bytes,
        *,
        prompt: str | None = None,
    ) -> ImageAnalysis:
        media_type = detect_image_media_type(content)
        max_bytes = self.settings.vision_max_image_size_mb * 1024 * 1024
        if len(content) > max_bytes:
            raise ValueError(
                f"图片超过 {self.settings.vision_max_image_size_mb} MB 限制"
            )

        instruction = prompt or (
            "请准确描述图片中的文字、对象、关系、图表数据与关键结论。"
            "不要猜测不可见信息；输出适合知识检索的结构化中文文本。"
        )
        data_url = (
            f"data:{media_type};base64,"
            f"{base64.b64encode(content).decode('ascii')}"
        )
        model = self.model or get_vision_chat_model(self.settings)
        try:
            response = model.invoke([
                HumanMessage(content=[
                    {"type": "text", "text": instruction},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ])
            ])
        except Exception as exc:
            raise KnowledgeBaseError(f"视觉模型调用失败: {exc}") from exc
        description = _content_text(response.content)
        if not description:
            raise KnowledgeBaseError("视觉模型未返回有效描述")
        return ImageAnalysis(description=description, media_type=media_type)


def _content_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            str(block.get("text", "")).strip()
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return ""
