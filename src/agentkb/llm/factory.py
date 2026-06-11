"""LLM 工厂函数：根据配置创建 Provider 并返回 ChatModel。"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from agentkb.config.settings import Settings
from agentkb.llm.base import LLMProvider
from agentkb.llm.ollama_provider import OllamaProvider
from agentkb.llm.openai_compatible_provider import OpenAICompatibleProvider
from agentkb.utils.exceptions import ConfigError


def _make_provider(
    cfg: Settings,
    model_name: str,
    provider_name: str | None = None,
) -> LLMProvider:
    """根据配置档声明的协议创建 Provider 实例。"""
    provider = provider_name or cfg.llm_provider
    protocol = cfg.llm_provider_value(provider, "protocol")
    base_url = cfg.llm_provider_value(provider, "base_url")
    if protocol == "ollama":
        return OllamaProvider(
            model_name=model_name,
            base_url=base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            request_timeout=cfg.llm_request_timeout,
        )
    if protocol == "openai":
        return OpenAICompatibleProvider(
            provider_name=provider,
            model_name=model_name,
            api_key=cfg.llm_provider_api_key(provider),
            base_url=base_url,
            temperature=cfg.llm_temperature,
            max_tokens=cfg.llm_max_tokens,
            request_timeout=cfg.llm_request_timeout,
        )
    raise ConfigError(
        f"LLM Provider '{provider}' 使用了不支持的协议: {protocol}"
    )


def create_llm(settings: Settings | None = None) -> LLMProvider:
    """根据配置实例化 LLM Provider（使用 generator_model_name）。"""
    cfg = settings or Settings.load()
    return _make_provider(cfg, cfg.llm_generator_model_name)


def create_router_llm(settings: Settings | None = None) -> LLMProvider:
    """创建路由专用 LLM（轻量模型，用于意图分类）。"""
    cfg = settings or Settings.load()
    return _make_provider(cfg, cfg.llm_router_model_name)


def get_chat_model(
    streaming: bool = True,
    settings: Settings | None = None,
) -> BaseChatModel:
    """返回生成模型 ChatModel（generator_model_name）。"""
    provider = create_llm(settings)
    return provider.get_chat_model(streaming=streaming)


def get_chat_model_for(
    model_name: str,
    *,
    streaming: bool = False,
    settings: Settings | None = None,
) -> BaseChatModel:
    """使用当前 Provider 配置创建指定模型的 ChatModel。"""
    provider = _make_provider(settings or Settings.load(), model_name)
    return provider.get_chat_model(streaming=streaming)


def get_router_chat_model(
    streaming: bool = False,
    settings: Settings | None = None,
) -> BaseChatModel:
    """返回路由模型 ChatModel（router_model_name，默认不流式）。"""
    provider = create_router_llm(settings)
    return provider.get_chat_model(streaming=streaming)


def get_vision_chat_model(
    settings: Settings | None = None,
) -> BaseChatModel:
    """返回独立配置的视觉模型。"""
    cfg = settings or Settings.load()
    if not cfg.vision_enabled:
        raise ConfigError("视觉能力未启用")
    provider = _make_provider(
        cfg,
        cfg.vision_model_name,
        provider_name=cfg.vision_provider,
    )
    return provider.get_chat_model(streaming=False)
