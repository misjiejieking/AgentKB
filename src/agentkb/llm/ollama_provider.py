"""Ollama LLM provider."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from loguru import logger

from agentkb.llm.base import LLMProvider
from agentkb.utils.exceptions import LLMConnectionError


class OllamaProvider(LLMProvider):
    """Ollama-backed LLM provider. Uses langchain-ollama's ChatOllama."""

    def __init__(
        self,
        model_name: str = "deepseek-chat",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        max_tokens: int = 4096,
        request_timeout: int = 120,
    ) -> None:
        self._model_name = model_name
        self._base_url = base_url
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout

    @property
    def model_name(self) -> str:
        return self._model_name

    def get_chat_model(self, streaming: bool = True) -> BaseChatModel:
        return ChatOllama(
            model=self._model_name,
            base_url=self._base_url,
            temperature=self._temperature,
            num_predict=self._max_tokens,
            client_kwargs={"timeout": self._request_timeout},
        )

    def validate_connection(self) -> bool:
        try:
            import httpx

            r = httpx.get(f"{self._base_url}/api/tags", timeout=10.0)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                logger.info(f"Ollama reachable — {len(models)} models available")
                if self._model_name not in models:
                    logger.warning(
                        f"Model '{self._model_name}' not found. Available: {models[:10]}"
                    )
                return True
            return False
        except Exception as e:
            raise LLMConnectionError(
                f"Cannot connect to Ollama at {self._base_url}: {e}"
            ) from e
