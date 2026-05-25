"""Trace 导出器——支持 Console、File、LangFuse、LangSmith、Prometheus。

导出器采用可插拔设计，在 config.yaml 中配置即可启用。
每个导出器实现 export_trace(trace_dict) 接口。
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

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

    def record_metric(self, metric_name: str, value: float, tags: dict) -> None:
        """记录指标（Prometheus 导出器实现此方法）。"""
        pass


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
            f"Trace: {trace_data['trace_id']}  ({summary.get('total_elapsed_ms', 0):.0f}ms)",
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
#  Prometheus 导出器 —— 指标聚合
# ══════════════════════════════════════════════════════════════

class PrometheusExporter(TraceExporter):
    """Prometheus 指标导出器——聚合指标供 Prometheus 抓取。

    在内存中维护 Counter/Histogram，暴露 /metrics 端点。
    不依赖 prometheus_client 库——纯内存实现，避免引入新依赖。
    如果安装了 prometheus_client，则使用官方库。
    """

    def __init__(self) -> None:
        self._metrics: dict[str, list[tuple[float, dict, float]]] = {}  # name → [(value, tags, timestamp)]
        self._use_official = False
        try:
            import prometheus_client  # noqa: F401
            self._use_official = True
            logger.info("Prometheus 导出器已启用（官方客户端）")
        except ImportError:
            logger.info("Prometheus 导出器已启用（内置实现）")

    @property
    def name(self) -> str:
        return "prometheus"

    def export_trace(self, trace_data: dict) -> None:
        """从 Trace 中提取指标聚合。"""
        summary = trace_data.get("summary", {})
        ts = time.time()
        tags = {"session_id": trace_data.get("session_id", "")}

        self._record("agentkb_request_duration_ms",
                     trace_data.get("total_elapsed_ms", 0), tags, ts)
        self._record("agentkb_request_tokens_total",
                     summary.get("total_tokens", 0), tags, ts)
        self._record("agentkb_tool_calls_total",
                     summary.get("total_tool_calls", 0), tags, ts)
        self._record("agentkb_tool_errors_total",
                     summary.get("tool_call_errors", 0), tags, ts)

        # Span 级别耗时
        for span in trace_data.get("spans", []):
            if span.get("elapsed_ms"):
                self._record(
                    f"agentkb_span_duration_ms",
                    span["elapsed_ms"],
                    {"span": span["name"], **tags},
                    ts,
                )

    def record_metric(self, metric_name: str, value: float, tags: dict) -> None:
        """外部调用的指标记录入口。"""
        self._record(metric_name, value, tags, time.time())

    def _record(self, name: str, value: float, tags: dict, ts: float) -> None:
        if self._use_official:
            # 使用 prometheus_client 库
            self._record_official(name, value, tags)
        else:
            self._metrics.setdefault(name, []).append((value, tags, ts))
            # 只保留最近 10000 条
            if len(self._metrics[name]) > 10000:
                self._metrics[name] = self._metrics[name][-5000:]

    # Prometheus 文本格式输出 —— 供 /metrics 端点使用
    def render_metrics(self) -> str:
        """以 Prometheus text format 输出当前指标快照。"""
        if self._use_official:
            import prometheus_client
            return prometheus_client.generate_latest().decode("utf-8")

        lines = []
        for name, entries in self._metrics.items():
            safe_name = name.replace("-", "_").replace(" ", "_")
            lines.append(f"# HELP {safe_name} AgentKB metric")
            lines.append(f"# TYPE {safe_name} gauge")
            for value, tags, ts in entries[-100:]:
                tag_str = ",".join(f'{k}="{v}"' for k, v in tags.items())
                tag_suffix = f"{{{tag_str}}}" if tag_str else ""
                lines.append(f"{safe_name}{tag_suffix} {value}")
        return "\n".join(lines) + "\n"

    def _record_official(self, name: str, value: float, tags: dict) -> None:
        """使用官方 prometheus_client 库记录。"""
        try:
            import prometheus_client as pc
            safe_name = "agentkb_" + name.replace("-", "_").replace(" ", "_")
            if safe_name not in _official_metrics:
                _official_metrics[safe_name] = pc.Gauge(
                    safe_name, "AgentKB metric", list(tags.keys()),
                )
            gauge = _official_metrics[safe_name]
            if tags:
                gauge.labels(**tags).set(value)
            else:
                gauge.set(value)
        except Exception:
            pass


_official_metrics: dict[str, Any] = {}


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
        "langfuse": LangFuseExporter,
        "langsmith": LangSmithExporter,
        "prometheus": PrometheusExporter,
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
