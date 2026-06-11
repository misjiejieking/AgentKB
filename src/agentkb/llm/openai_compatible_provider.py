"""OpenAI 兼容协议的 LLM Provider。"""

from __future__ import annotations

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from loguru import logger
from pydantic import SecretStr

from agentkb.llm.base import LLMProvider
from agentkb.utils.exceptions import LLMConnectionError


class OpenAICompatibleProvider(LLMProvider):
    """适配 DeepSeek、OpenAI 及其他 OpenAI 兼容服务。"""

    def __init__(
        self,
        provider_name: str,
        model_name: str,
        api_key: str,
        base_url: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        request_timeout: int = 120,
    ) -> None:
        self._provider_name = provider_name
        self._model_name = model_name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_chat_model(self, streaming: bool = True) -> BaseChatModel:
        api_key = SecretStr(self._api_key)
        if self._provider_name == "openai":
            return ChatOpenAI(
                model=self._model_name,
                api_key=api_key,
                base_url=self._base_url,
                temperature=self._temperature,
                timeout=self._request_timeout,
                streaming=streaming,
                max_completion_tokens=self._max_tokens,
            )
        return BaseChatOpenAI(
            model=self._model_name,
            api_key=api_key,
            base_url=self._base_url,
            temperature=self._temperature,
            timeout=self._request_timeout,
            streaming=streaming,
            max_tokens=self._max_tokens,
        )

    def validate_connection(self) -> bool:
        if not self._api_key:
            raise LLMConnectionError(
                f"LLM Provider '{self._provider_name}' 未配置 API Key"
            )

        try:
            response = httpx.get(
                f"{self._base_url}/models",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=15.0,
            )
        except httpx.HTTPError as exc:
            raise LLMConnectionError(
                f"无法连接 LLM Provider '{self._provider_name}' "
                f"({self._base_url}): {exc}"
            ) from exc

        if response.is_success:
            logger.info(
                f"LLM Provider '{self._provider_name}' 连接正常"
            )
            return True

        logger.error(
            f"LLM Provider '{self._provider_name}' 返回 "
            f"{response.status_code}: {response.text[:200]}"
        )
        return False
