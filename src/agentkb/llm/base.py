"""Abstract LLM Provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMProvider(ABC):
    """Abstract base for LLM backends (Ollama, OpenAI, etc.)."""

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def get_chat_model(self, streaming: bool = True) -> BaseChatModel:
        """Return a LangChain-compatible ChatModel with tool-calling support."""

    @abstractmethod
    def validate_connection(self) -> bool:
        """Test whether the LLM backend is reachable."""
