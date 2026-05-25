"""DeepSeek LLM provider——OpenAI 兼容接口，支持 deepseek-chat (V3) 和 deepseek-reasoner (R1)。"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from loguru import logger

from agentkb.llm.base import LLMProvider
from agentkb.utils.exceptions import LLMConnectionError


class DeepSeekProvider(LLMProvider):
    """DeepSeek API Provider，基于 OpenAI 兼容协议。

    环境变量:
      DEEPSEEK_API_KEY: DeepSeek API Key (必需)
    """

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        request_timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self._api_key:
            logger.warning("DEEPSEEK_API_KEY 未设置，请设置环境变量或配置文件中提供")

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_chat_model(self, streaming: bool = True) -> BaseChatModel:
        return ChatOpenAI(
            model=self._model_name,
            api_key=self._api_key,
            base_url=f"{self._base_url}/v1",
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            timeout=self._request_timeout,
            streaming=streaming,
        )

    def validate_connection(self) -> bool:
        try:
            import httpx

            headers = {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            }
            r = httpx.get(
                f"{self._base_url}/v1/models",
                headers=headers,
                timeout=15.0,
            )
            if r.status_code == 200:
                data = r.json()
                models = [m.get("id", "") for m in data.get("data", [])]
                logger.info(f"DeepSeek API OK — {len(models)} models available: {models}")
                return True
            logger.error(f"DeepSeek API 返回 {r.status_code}: {r.text[:200]}")
            return False
        except Exception as e:
            raise LLMConnectionError(
                f"无法连接 DeepSeek API ({self._base_url}): {e}"
            ) from e
