""".md / .txt 文件加载与校验。"""

from __future__ import annotations

import shutil
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from loguru import logger

from agentkb.utils.exceptions import KnowledgeBaseError


class FileLoader:
    """将文本文件加载为 LangChain Document。支持 .md / .txt，最大 10 MB。"""

    SUPPORTED_EXTENSIONS = {".md", ".txt"}
    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

    def __init__(self, upload_dir: str = "data/uploads") -> None:
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, file_path: str | Path) -> Path:
        """校验文件扩展名与大小，不合法时抛出 KnowledgeBaseError。"""
        path = Path(file_path)
        if not path.exists():
            raise KnowledgeBaseError(f"文件不存在: {file_path}")
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise KnowledgeBaseError(
                f"不支持的文件类型 '{path.suffix}'。"
                f"支持: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )
        if path.stat().st_size > self.MAX_FILE_SIZE_BYTES:
            raise KnowledgeBaseError(
                f"文件过大: {path.stat().st_size} 字节"
                f"（上限 {self.MAX_FILE_SIZE_BYTES}）"
            )
        return path

    def save(self, file_path: str | Path) -> Path:
        """将文件复制到上传目录，若同名文件已存在则自动加序号。"""
        src = self.validate(file_path)
        dst = self._upload_dir / src.name
        counter = 1
        while dst.exists():
            dst = self._upload_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, dst)
        logger.info(f"文件已保存: {src} → {dst}")
        return dst

    def load(self, file_path: str | Path) -> list[Document]:
        """加载文本文件为 Document 列表（一个文件返回一个 Document）。"""
        path = Path(file_path)
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        logger.debug(f"已加载 {len(docs)} 个文档，来源: {path.name}")
        return docs
