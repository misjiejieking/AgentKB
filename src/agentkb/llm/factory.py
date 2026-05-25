"""LLM 工厂函数：根据配置创建 Provider 并返回 ChatModel。"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agentkb.config.settings import Settings
from agentkb.llm.base import LLMProvider
from agentkb.llm.ollama_provider import OllamaProvider
from agentkb.llm.deepseek_provider import DeepSeekProvider
from agentkb.utils.exceptions import ConfigError


def create_llm(settings: Settings | None = None) -> LLMProvider:
    """根据配置实例化 LLM Provider。"""
    cfg = settings or Settings.load()

    provider = cfg.llm_provider
    if provider == "ollama":
        return OllamaProvider(
            model_name=cfg.llm_model_name,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            request_timeout=cfg.llm_request_timeout,
        )
    if provider == "deepseek":
        return DeepSeekProvider(
            model_name=cfg.llm_model_name,
            api_key=cfg.llm_api_key or cfg.openai_api_key,
            base_url=cfg.llm_base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            request_timeout=cfg.llm_request_timeout,
        )
    raise ConfigError(f"未知的 LLM Provider: {provider}")


def get_chat_model(
    streaming: bool = True,
    settings: Settings | None = None,
) -> BaseChatModel:
    """快捷方法：返回可直接 bind_tools 的 ChatModel 实例。"""
    provider = create_llm(settings)
    return provider.get_chat_model(streaming=streaming)
