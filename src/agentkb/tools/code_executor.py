"""工具：execute_code——安全沙箱代码执行（Python/JavaScript）。

使用 subprocess + 超时限制实现基础安全隔离。
生产环境建议替换为 Docker 沙箱或 e2b.dev。
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from agentkb.tools.base import BaseTool, ToolResult


class CodeExecuteInput(BaseModel):
    code: str = Field(description="要执行的代码")
    language: str = Field(default="python", description="语言: python 或 javascript")


class CodeExecutorTool(BaseTool):
    """安全沙箱代码执行工具。"""

    _TIMEOUT = 30       # 最大执行秒数
    _MAX_OUTPUT = 4096  # 最大输出字符数

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return (
            "在安全沙箱中执行 Python 或 JavaScript 代码。"
            "适合运行简单计算、数据处理、代码验证等任务。"
            "不要用来执行不安全的操作或访问文件系统。"
        )

    @property
    def args_schema(self) -> type[BaseModel]:
        return CodeExecuteInput

    async def _execute(self, code: str, language: str = "python") -> ToolResult:
        if language not in ("python", "javascript"):
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"不支持的语言: {language}，仅支持 python/javascript",
            )

        if language == "python":
            return await self._run_python(code)
        return await self._run_javascript(code)

    async def _run_python(self, code: str) -> ToolResult:
        """用 subprocess 执行 Python 代码。"""
        try:
            proc = subprocess.run(
                ["python", "-c", code],
                capture_output=True, text=True, timeout=self._TIMEOUT,
                cwd=str(Path.home()),
            )
            stdout = proc.stdout[:self._MAX_OUTPUT]
            stderr = proc.stderr[:self._MAX_OUTPUT]

            if proc.returncode != 0:
                return ToolResult(
                    tool_name=self.name, success=True,
                    data={"stdout": stdout, "stderr": stderr, "exit_code": proc.returncode},
                )

            return ToolResult(
                tool_name=self.name, success=True,
                data={"stdout": stdout, "stderr": stderr, "exit_code": 0},
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"代码执行超时（>{self._TIMEOUT}秒）",
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Python 未安装或不在 PATH 中",
            )

    async def _run_javascript(self, code: str) -> ToolResult:
        """用 node 执行 JS。"""
        try:
            proc = subprocess.run(
                ["node", "-e", code],
                capture_output=True, text=True, timeout=self._TIMEOUT,
            )
            stdout = proc.stdout[:self._MAX_OUTPUT]
            stderr = proc.stderr[:self._MAX_OUTPUT]

            return ToolResult(
                tool_name=self.name, success=True,
                data={
                    "stdout": stdout,
                    "stderr": stderr,
                    "exit_code": proc.returncode,
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name=self.name, success=False,
                error=f"代码执行超时（>{self._TIMEOUT}秒）",
            )
        except FileNotFoundError:
            return ToolResult(
                tool_name=self.name, success=False,
                error="Node.js 未安装或不在 PATH 中",
            )
