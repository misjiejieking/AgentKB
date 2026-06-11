from __future__ import annotations

import subprocess

from agentkb.tools.code_executor import CodeExecutorTool


async def test_python_nonzero_exit_code_is_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="RuntimeError: expected failure",
        ),
    )

    result = await CodeExecutorTool()._run_python(
        "raise RuntimeError('expected failure')"
    )

    assert result.success is False
    assert "RuntimeError: expected failure" in result.error


async def test_javascript_nonzero_exit_code_is_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="",
            stderr="Error: expected failure",
        ),
    )

    result = await CodeExecutorTool()._run_javascript(
        "throw new Error('expected failure')"
    )

    assert result.success is False
    assert "expected failure" in result.error
