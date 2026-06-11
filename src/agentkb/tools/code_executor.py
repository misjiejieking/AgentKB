"""工具：execute_code——经人工确认后执行 Python/JavaScript 子进程。"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import tempfile
from typing import Any

from pydantic import BaseModel, Field

from agentkb.tools.base import BaseTool, ToolResult


class CodeExecuteInput(BaseModel):
    code: str = Field(description="要执行的代码")
    language: str = Field(default="python", description="语言: python 或 javascript")


class CodeExecutorTool(BaseTool):
    """带超时和临时工作目录的代码执行工具，不构成安全沙箱。"""

    _TIMEOUT = 30       # 最大执行秒数
    _MAX_OUTPUT = 4096  # 最大输出字符数

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return (
            "在本机隔离子进程中执行 Python 或 JavaScript 代码。"
            "适合运行简单计算、数据处理、代码验证等任务。"
            "执行前必须由用户确认；该工具不是容器级安全沙箱。"
        )

    @property
    def requires_confirmation(self) -> bool:
        return True

    @property
    def confirmation_message(self) -> str:
        return "该操作将在本机执行代码，必须由你确认。"

    @property
    def args_schema(self) -> type[BaseModel]:
        return CodeExecuteInput

    async def _execute(
        self,
        code: str = "",
        language: str = "python",
        **kwargs: Any,
    ) -> ToolResult:
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
            with tempfile.TemporaryDirectory(prefix="agentkb-code-") as workdir:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    [sys.executable, "-I", "-c", code],
                    capture_output=True,
                    text=True,
                    timeout=self._TIMEOUT,
                    cwd=workdir,
                    env=self._subprocess_env(workdir),
                )
            stdout = proc.stdout[:self._MAX_OUTPUT]
            stderr = proc.stderr[:self._MAX_OUTPUT]

            if proc.returncode != 0:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=stderr or stdout or f"Python 进程退出码为 {proc.returncode}",
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
            with tempfile.TemporaryDirectory(prefix="agentkb-code-") as workdir:
                proc = await asyncio.to_thread(
                    subprocess.run,
                    ["node", "-e", code],
                    capture_output=True,
                    text=True,
                    timeout=self._TIMEOUT,
                    cwd=workdir,
                    env=self._subprocess_env(workdir),
                )
            stdout = proc.stdout[:self._MAX_OUTPUT]
            stderr = proc.stderr[:self._MAX_OUTPUT]

            if proc.returncode != 0:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=stderr or stdout or f"JavaScript 进程退出码为 {proc.returncode}",
                )

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

    @staticmethod
    def _subprocess_env(workdir: str) -> dict[str, str]:
        """仅传递进程启动必需变量，避免泄露服务密钥。"""
        allowed = ("PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC")
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env.update({"HOME": workdir, "TEMP": workdir, "TMP": workdir})
        return env
