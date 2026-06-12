from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_direct_entrypoint_resolves_official_mcp_sdk():
    root = Path(__file__).parents[2]
    script = """
import runpy
import sys

sys.argv = ["src/agentkb/main.py"]
namespace = runpy.run_path("src/agentkb/main.py", run_name="entrypoint_test")
from mcp import ClientSession

print(ClientSession.__module__)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "mcp.client.session" in result.stdout
