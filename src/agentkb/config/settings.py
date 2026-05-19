"""全局配置：YAML 文件加载，支持环境变量覆盖。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


class Settings:
    """全局配置单例，从 YAML 加载，支持 AGENTKB_<SECTION>_<KEY> 环境变量覆盖。"""

    _instance: Settings | None = None

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    # ── 单例生命周期 ───────────────────────────────────────────

    @classmethod
    def load(cls, config_path: str | None = None) -> Settings:
        if cls._instance is not None:
            return cls._instance
        path = config_path or cls._find_config()
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cls._instance = cls(raw)
        return cls._instance

    @classmethod
    def reload(cls, config_path: str | None = None) -> Settings:
        """强制重新加载配置（测试用）。"""
        cls._instance = None
        return cls.load(config_path)

    @staticmethod
    def _find_config() -> str:
        env_path = os.getenv("AGENTKB_CONFIG")
        if env_path:
            return env_path
        for candidate in [
            "config.yaml",
            "src/agentkb/config/config.yaml",
        ]:
            if Path(candidate).exists():
                return candidate
        raise FileNotFoundError(
            "config.yaml not found. Set AGENTKB_CONFIG env var."
        )

    # ── 工具方法 ───────────────────────────────────────────────

    def _val(self, *keys: str, default: Any = None) -> Any:
        """按嵌套 key 路径从 YAML 读取值；优先返回同名环境变量。"""
        env_key = "AGENTKB_" + "_".join(keys).upper()
        if env_key in os.environ:
            raw = os.environ[env_key]
            # 根据 YAML 中的类型做适当的类型转换
            yaml_val = self._data
            for k in keys:
                yaml_val = yaml_val.get(k, {})
            if isinstance(yaml_val, bool):
                return raw.lower() in ("1", "true", "yes")
            if isinstance(yaml_val, int):
                return int(raw)
            if isinstance(yaml_val, float):
                return float(raw)
            return raw
        d = self._data
        for k in keys:
            d = d.get(k, {})
        return d if d != {} else default

    # ── app ─────────────────────────────────────────────────────

    @property
    def app_host(self) -> str:
        return self._val("app", "host")

    @property
    def app_port(self) -> int:
        return self._val("app", "port")

    @property
    def app_debug(self) -> bool:
        return self._val("app", "debug")

    @property
    def app_auto_open_browser(self) -> bool:
        return self._val("app", "auto_open_browser")

    # ── llm ─────────────────────────────────────────────────────

    @property
    def llm_provider(self) -> str:
        return self._val("llm", "provider")

    @property
    def llm_model_name(self) -> str:
        return self._val("llm", "model_name")

    @property
    def llm_base_url(self) -> str:
        return self._val("llm", "base_url")

    @property
    def llm_temperature(self) -> float:
        return self._val("llm", "temperature")

    @property
    def llm_max_tokens(self) -> int:
        return self._val("llm", "max_tokens")

    @property
    def llm_request_timeout(self) -> int:
        return self._val("llm", "request_timeout")

    @property
    def llm_streaming(self) -> bool:
        return self._val("llm", "streaming")

    @property
    def openai_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", self._val("llm", "openai_api_key"))

    @property
    def openai_base_url(self) -> str:
        return self._val("llm", "openai_base_url")

    # ── embedding ───────────────────────────────────────────────

    @property
    def embedding_model_name(self) -> str:
        return self._val("embedding", "model_name")

    @property
    def embedding_device(self) -> str:
        return self._val("embedding", "device")

    @property
    def embedding_normalize(self) -> bool:
        return self._val("embedding", "normalize")

    @property
    def embedding_batch_size(self) -> int:
        return self._val("embedding", "batch_size")

    @property
    def embedding_dimension(self) -> int:
        return self._val("embedding", "dimension")

    # ── knowledge ───────────────────────────────────────────────

    @property
    def knowledge_supported_extensions(self) -> list[str]:
        return self._val("knowledge", "supported_extensions")

    @property
    def knowledge_max_file_size_mb(self) -> int:
        return self._val("knowledge", "max_file_size_mb")

    # ── postgresql ──────────────────────────────────────────────

    @property
    def pg_host(self) -> str:
        return self._val("postgresql", "host")

    @property
    def pg_port(self) -> int:
        return self._val("postgresql", "port")

    @property
    def pg_dbname(self) -> str:
        return self._val("postgresql", "dbname")

    @property
    def pg_user(self) -> str:
        return self._val("postgresql", "user")

    @property
    def pg_password(self) -> str:
        return self._val("postgresql", "password")

    @property
    def pg_pool_min(self) -> int:
        return self._val("postgresql", "pool_min")

    @property
    def pg_pool_max(self) -> int:
        return self._val("postgresql", "pool_max")

    # ── retrieval ───────────────────────────────────────────────

    @property
    def retrieval_dense_weight(self) -> float:
        return self._val("retrieval", "dense_weight")

    @property
    def retrieval_bm25_weight(self) -> float:
        return self._val("retrieval", "bm25_weight")

    @property
    def retrieval_candidate_k(self) -> int:
        return self._val("retrieval", "candidate_k")

    @property
    def retrieval_final_k(self) -> int:
        return self._val("retrieval", "final_k")

    @property
    def retrieval_rrf_k(self) -> int:
        return self._val("retrieval", "rrf_k")

    # ── chunking ────────────────────────────────────────────────

    @property
    def chunking_semantic_threshold(self) -> float:
        return self._val("chunking", "semantic_threshold")

    @property
    def chunking_parent_size(self) -> int:
        return self._val("chunking", "parent_size")

    @property
    def chunking_child_size(self) -> int:
        return self._val("chunking", "child_size")

    @property
    def chunking_sliding_size(self) -> int:
        return self._val("chunking", "sliding_size")

    @property
    def chunking_sliding_overlap(self) -> int:
        return self._val("chunking", "sliding_overlap")

    # ── database (postgresql) ───────────────────────────────────

    @property
    def database_type(self) -> str:
        return self._val("database", "type", default=self.pg_dbname and "postgresql" or "sqlite")

    # ── web search ──────────────────────────────────────────────

    @property
    def web_search_enabled(self) -> bool:
        return self._val("web_search", "enabled")

    @property
    def web_search_engine(self) -> str:
        return self._val("web_search", "engine")

    @property
    def web_search_max_results(self) -> int:
        return self._val("web_search", "max_results")

    @property
    def web_search_timeout(self) -> int:
        return self._val("web_search", "timeout")

    # ── logging ─────────────────────────────────────────────────

    @property
    def logging_level(self) -> str:
        return self._val("logging", "level")

    @property
    def logging_file(self) -> str:
        return self._val("logging", "file")

    @property
    def logging_rotation(self) -> str:
        return self._val("logging", "rotation")

    @property
    def logging_retention(self) -> str:
        return self._val("logging", "retention")

    @property
    def logging_console(self) -> bool:
        return self._val("logging", "console")

    @property
    def logging_log_tool_calls(self) -> bool:
        return self._val("logging", "log_tool_calls")

    # ── langgraph ───────────────────────────────────────────────

    @property
    def langgraph_checkpointer_db(self) -> str:
        return self._val("langgraph", "checkpointer_db")

    @property
    def langgraph_max_recursion_limit(self) -> int:
        return self._val("langgraph", "max_recursion_limit")
