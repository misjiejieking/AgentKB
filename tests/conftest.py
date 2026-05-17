"""Pytest fixtures for AgentKB tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory that cleans up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_md_file(temp_dir: Path) -> Path:
    """Create a sample markdown file for testing."""
    content = """# 公司年假制度

## 申请条件
- 入职满一年可申请年假
- 每年享有 5 天带薪年假

## 申请流程
1. 在 OA 系统提交申请
2. 直属领导审批
3. HR 确认

## 注意事项
- 年假需提前 3 天申请
- 不可跨年累计
"""
    path = temp_dir / "test_policy.md"
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def sample_txt_file(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    content = "Python 是一门解释型、面向对象的高级编程语言。"
    path = temp_dir / "test_notes.txt"
    path.write_text(content, encoding="utf-8")
    return path
