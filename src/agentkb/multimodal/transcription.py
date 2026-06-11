"""OpenAI 兼容语音转写服务。"""

from __future__ import annotations

import httpx

from agentkb.config.settings import Settings


class TranscriptionService:
    """调用可配置的 OpenAI 兼容 `/audio/transcriptions` 端点。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()

    def transcribe(
        self,
        content: bytes,
        *,
        filename: str,
        media_type: str,
    ) -> str:
        if not self.settings.transcription_enabled:
            raise ValueError("语音转写未启用")
        max_bytes = self.settings.transcription_max_audio_size_mb * 1024 * 1024
        if not content or len(content) > max_bytes:
            raise ValueError(
                f"音频为空或超过 {self.settings.transcription_max_audio_size_mb} MB 限制"
            )

        headers = {}
        if self.settings.transcription_api_key:
            headers["Authorization"] = (
                f"Bearer {self.settings.transcription_api_key}"
            )
        response = httpx.post(
            f"{self.settings.transcription_base_url.rstrip('/')}/audio/transcriptions",
            headers=headers,
            data={"model": self.settings.transcription_model_name},
            files={"file": (filename, content, media_type)},
            timeout=120,
        )
        response.raise_for_status()
        text = str(response.json().get("text", "")).strip()
        if not text:
            raise ValueError("转写服务未返回文本")
        return text
