from __future__ import annotations

from agentkb.config.settings import Settings
from agentkb.llm.factory import create_llm
from agentkb.llm.ollama_provider import OllamaProvider
from agentkb.llm.openai_compatible_provider import OpenAICompatibleProvider


def _settings(provider: str) -> Settings:
    return Settings({
        "llm": {
            "provider": provider,
            "providers": {
                "ollama": {
                    "protocol": "ollama",
                    "base_url": "http://localhost:11434",
                    "api_key": "",
                    "router_model_name": "qwen-router",
                    "generator_model_name": "qwen-generator",
                },
                "deepseek": {
                    "protocol": "openai",
                    "base_url": "https://api.deepseek.com/v1",
                    "api_key": "configured-key",
                    "router_model_name": "deepseek-chat",
                    "generator_model_name": "deepseek-reasoner",
                },
            },
            "temperature": 0.1,
            "max_tokens": 4096,
            "request_timeout": 120,
        }
    })


def test_factory_switches_provider_by_profile():
    assert isinstance(create_llm(_settings("ollama")), OllamaProvider)

    provider = create_llm(_settings("deepseek"))

    assert isinstance(provider, OpenAICompatibleProvider)
    assert provider.model_name == "deepseek-reasoner"
    assert provider._base_url == "https://api.deepseek.com/v1"


def test_provider_specific_api_key_environment_variable(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-key")

    assert _settings("deepseek").llm_api_key == "environment-key"
