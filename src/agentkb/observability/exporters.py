"""Trace 导出器——支持 Console、File、PostgreSQL、LangFuse、LangSmith。

导出器采用可插拔设计，在 config.yaml 中配置即可启用。
每个导出器实现 export_trace(trace_dict) 接口。
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from loguru import logger

from agentkb.config.settings import Settings


class TraceExporter(ABC):
    """导出器基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """导出器名称，用于日志和配置。"""
        ...

    @abstractmethod
    def export_trace(self, trace_data: dict) -> None:
        """导出一条完整的 Trace 数据。"""
        ...

# ══════════════════════════════════════════════════════════════
#  Console 导出器 —— 开发环境使用
# ══════════════════════════════════════════════════════════════

class ConsoleExporter(TraceExporter):
    """控制台导出器——结构化打印 Trace 摘要到 stdout。"""

    @property
    def name(self) -> str:
        return "console"

    def export_trace(self, trace_data: dict) -> None:
        summary = trace_data.get("summary", {})
        spans = trace_data.get("spans", [])

        # 构建 Span 树形输出
        lines = [
            f"\n{'='*60}",
            f"Trace: {trace_data['trace_id']}  "
            f"({trace_data.get('total_elapsed_ms', 0):.0f}ms)",
            f"Query: {trace_data.get('query', '')[:80]}",
            f"{'='*60}",
        ]
        for span in spans:
            indent = "  " if span.get("parent_span_id") else ""
            status_icon = "✓" if span.get("status") == "ok" else "✗"
            events = span.get("events", [])
            lines.append(
                f"{indent}{status_icon} [{span.get('name', '?')}] "
                f"{span.get('elapsed_ms', 0):.1f}ms"
            )
            for evt in events:
                lines.append(
                    f"{indent}  └─ {evt['name']} ({evt.get('elapsed_ms', 0):.1f}ms) "
                    f"{evt.get('attributes', {})}"
                )

        lines.append(f"Tokens: {summary.get('total_tokens', 0)} | "
                      f"ToolCalls: {summary.get('total_tool_calls', 0)} | "
                      f"Errors: {summary.get('tool_call_errors', 0)}")
        lines.append(f"{'='*60}\n")

        for line in lines:
            logger.opt(depth=1).info(line)


# ══════════════════════════════════════════════════════════════
#  File 导出器 —— 生产环境持久化
# ══════════════════════════════════════════════════════════════

class FileExporter(TraceExporter):
    """文件导出器——JSON 格式写入 data/traces/ 目录。"""

    def __init__(self, output_dir: str = "data/traces") -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "file"

    def export_trace(self, trace_data: dict) -> None:
        trace_id = trace_data.get("trace_id", "unknown")
        # 按日期分目录
        date_str = time.strftime("%Y-%m-%d")
        day_dir = self._dir / date_str
        day_dir.mkdir(parents=True, exist_ok=True)

        filepath = day_dir / f"{trace_id}.json"
        filepath.write_text(
            json.dumps(trace_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.debug(f"Trace 已写入: {filepath}")


class PostgreSQLExporter(TraceExporter):
    """将 Trace 写入统一 PostgreSQL 存储。"""

    @property
    def name(self) -> str:
        return "postgresql"

    def export_trace(self, trace_data: dict) -> None:
        from agentkb.storage.pg_database import get_db

        get_db().save_trace(trace_data)


# ══════════════════════════════════════════════════════════════
#  LangFuse 导出器
# ══════════════════════════════════════════════════════════════

class LangFuseExporter(TraceExporter):
    """LangFuse 导出器——通过 HTTP API 上报 Trace。

    配置:
      observability:
        langfuse:
          public_key: "pk-..."
          secret_key: "sk-..."
          host: "https://cloud.langfuse.com"
    """

    def __init__(self) -> None:
        cfg = Settings.load()
        self._public_key = os.getenv("LANGFUSE_PUBLIC_KEY", cfg._val("observability", "langfuse", "public_key"))
        self._secret_key = os.getenv("LANGFUSE_SECRET_KEY", cfg._val("observability", "langfuse", "secret_key"))
        self._host = cfg._val("observability", "langfuse", "host", default="https://cloud.langfuse.com")
        self._enabled = bool(self._public_key and self._secret_key)
        if self._enabled:
            logger.info(f"LangFuse 导出器已启用: {self._host}")
        else:
            logger.debug("LangFuse 未配置，跳过")

    @property
    def name(self) -> str:
        return "langfuse"

    def export_trace(self, trace_data: dict) -> None:
        if not self._enabled:
            return

        import httpx

        # 将内部 Trace 格式转为 LangFuse ingestion API 格式
        langfuse_payload = self._to_langfuse_format(trace_data)

        try:
            url = f"{self._host.rstrip('/')}/api/public/ingestion"
            auth = (
                self._public_key,
                self._secret_key,
            )
            resp = httpx.post(
                url,
                json=langfuse_payload,
                auth=auth,
                timeout=10,
            )
            if resp.status_code >= 400:
                logger.warning(f"LangFuse 上报失败 ({resp.status_code}): {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"LangFuse 请求异常: {e}")

    @staticmethod
    def _to_langfuse_format(trace_data: dict) -> dict:
        """将内部 Trace 格式转为 LangFuse 批量 ingestion 格式。"""
        trace_id = trace_data["trace_id"]
        events = []
        # Trace 级别事件
        events.append({
            "type": "trace-create",
            "body": {
                "id": trace_id,
                "name": trace_data.get("query", "AgentKB Request")[:100],
                "userId": trace_data.get("session_id", "anonymous"),
                "metadata": {
                    "total_tokens": trace_data["summary"]["total_tokens"],
                    "total_tool_calls": trace_data["summary"]["total_tool_calls"],
                },
            },
        })
        # Span → LangFuse "observation"
        for span in trace_data.get("spans", []):
            events.append({
                "type": "observation-create",
                "body": {
                    "id": span["span_id"],
                    "traceId": trace_id,
                    "parentObservationId": span.get("parent_span_id") or None,
                    "name": span["name"],
                    "startTime": span.get("start_time", ""),
                    "endTime": span.get("end_time", ""),
                    "metadata": span.get("attributes", {}),
                    "level": "ERROR" if span.get("status") == "error" else "DEFAULT",
                },
            })
        return {"batch": events}


# ══════════════════════════════════════════════════════════════
#  LangSmith 导出器
# ══════════════════════════════════════════════════════════════

class LangSmithExporter(TraceExporter):
    """LangSmith 导出器——通过 langsmith SDK 上报 Trace。

    配置:
      LANGCHAIN_API_KEY=ls_...
      LANGCHAIN_PROJECT=agentkb
    """

    def __init__(self) -> None:
        self._api_key = os.getenv("LANGCHAIN_API_KEY", "")
        self._project = os.getenv("LANGCHAIN_PROJECT", "agentkb")
        self._enabled = bool(self._api_key)
        if self._enabled:
            logger.info(f"LangSmith 导出器已启用: project={self._project}")

    @property
    def name(self) -> str:
        return "langsmith"

    def export_trace(self, trace_data: dict) -> None:
        if not self._enabled:
            return

        try:
            from langsmith import Client
            client = Client(api_key=self._api_key)

            # 创建 Run
            client.create_run(
                name=trace_data.get("query", "AgentKB Request")[:100],
                run_type="chain",
                inputs={"query": trace_data.get("query", "")},
                outputs={"trace_id": trace_data["trace_id"]},
                id=trace_data["trace_id"],
                project_name=self._project,
                extra={
                    "summary": trace_data["summary"],
                    "spans": trace_data["spans"],
                },
            )
        except ImportError:
            logger.debug("langsmith SDK 未安装，跳过 LangSmith 导出")
        except Exception as e:
            logger.warning(f"LangSmith 上报失败: {e}")


# ══════════════════════════════════════════════════════════════
#  导出器注册
# ══════════════════════════════════════════════════════════════

def get_exporters() -> list[TraceExporter]:
    """根据配置创建导出器列表。"""
    cfg = Settings.load()
    enabled = cfg._val("observability", "exporters", default=["console", "file"])

    exporter_map: dict[str, type[TraceExporter]] = {
        "console": ConsoleExporter,
        "file": FileExporter,
        "postgresql": PostgreSQLExporter,
        "langfuse": LangFuseExporter,
        "langsmith": LangSmithExporter,
    }

    exporters = []
    for name in enabled:
        if name in exporter_map:
            try:
                exporters.append(exporter_map[name]())
            except Exception as e:
                logger.warning(f"导出器 {name} 初始化失败: {e}")

    logger.debug(f"已启用 {len(exporters)} 个 Trace 导出器: {[e.name for e in exporters]}")
    return exporters
