
"""基于 loguru 的日志系统配置。"""

from __future__ import annotations

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    level: str = "INFO",
    log_file: str = "data/logs/agentkb.log",
    rotation: str = "10 MB",
    retention: str = "7 days",
    console: bool = True,
) -> None:
    """配置 loguru：控制台彩色输出 + 文件滚动存储。"""
    logger.remove()

    fmt = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    if console:
        logger.add(sys.stderr, format=fmt, level=level, colorize=True)

    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        str(log_path),
        format=fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
    )

    logger.info(f"日志系统初始化完成 — 级别={level}, 文件={log_file}")
