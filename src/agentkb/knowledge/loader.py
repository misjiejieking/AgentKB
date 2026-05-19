""".md / .txt / .pdf / .docx / .csv / .json 文件加载与校验——统一注册机制。"""

from __future__ import annotations

import csv
import json
import shutil
from abc import ABC, abstractmethod
from io import StringIO
from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from loguru import logger

from agentkb.utils.exceptions import KnowledgeBaseError


class BaseFileLoader(ABC):
    """文件加载器基类——每个子类负责一种文件格式。"""

    @property
    @abstractmethod
    def extensions(self) -> set[str]:
        """支持的文件扩展名集合，含点号（如 {'.pdf', '.docx'}）。"""
        ...

    @abstractmethod
    def load(self, file_path: Path) -> list[Document]:
        """加载文件为 LangChain Document 列表。"""
        ...


# ══════════════════════════════════════════════════════════════
#  内置 Loader 实现
# ══════════════════════════════════════════════════════════════

class MarkdownLoader(BaseFileLoader):
    """.md / .txt 文本加载器。"""

    @property
    def extensions(self) -> set[str]:
        return {".md", ".txt"}

    def load(self, file_path: Path) -> list[Document]:
        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = file_path.name
        return docs


class PdfLoader(BaseFileLoader):
    """.pdf 加载器（PyMuPDF），逐页提取文本。"""

    @property
    def extensions(self) -> set[str]:
        return {".pdf"}

    def load(self, file_path: Path) -> list[Document]:
        import fitz  # PyMuPDF
        docs = []
        try:
            doc = fitz.open(str(file_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": file_path.name,
                            "page_number": page_num + 1,
                            "total_pages": len(doc),
                        },
                    ))
            doc.close()
        except Exception as e:
            raise KnowledgeBaseError(f"PDF 解析失败 '{file_path.name}': {e}") from e

        if not docs:
            raise KnowledgeBaseError(f"PDF 文件中无有效文本内容: {file_path.name}")
        return docs


class DocxLoader(BaseFileLoader):
    """.docx 加载器（python-docx），按段落提取，保留标题样式信息。"""

    @property
    def extensions(self) -> set[str]:
        return {".docx"}

    def load(self, file_path: Path) -> list[Document]:
        from docx import Document as DocxDocument
        from docx.enum.style import WD_STYLE_TYPE

        try:
            doc = DocxDocument(str(file_path))
        except Exception as e:
            raise KnowledgeBaseError(f"DOCX 解析失败 '{file_path.name}': {e}") from e

        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            is_heading = para.style and para.style.type == WD_STYLE_TYPE.PARAGRAPH and "heading" in (para.style.name or "").lower()
            paragraphs.append({
                "text": text,
                "is_heading": is_heading,
                "style": para.style.name if para.style else "",
            })

        if not paragraphs:
            raise KnowledgeBaseError(f"DOCX 文件中无有效文本内容: {file_path.name}")

        # 整个文档作为一个 Document，段落结构保留到 metadata
        full_text = "\n\n".join(
            (f"## {p['text']}" if p["is_heading"] else p["text"])
            for p in paragraphs
        )
        return [Document(
            page_content=full_text,
            metadata={"source": file_path.name, "paragraph_count": len(paragraphs)},
        )]


class CsvLoader(BaseFileLoader):
    """.csv 加载器，按行处理，表头作为 metadata。"""

    @property
    def extensions(self) -> set[str]:
        return {".csv"}

    def load(self, file_path: Path) -> list[Document]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            content = file_path.read_text(encoding="gbk")

        reader = csv.reader(StringIO(content))
        headers = next(reader, None)
        if not headers:
            raise KnowledgeBaseError(f"CSV 文件无表头: {file_path.name}")

        # 每 50 行合并为一个 Document
        docs = []
        batch_lines = []
        row_count = 0
        for row in reader:
            batch_lines.append(", ".join(row))
            row_count += 1
            if len(batch_lines) >= 50:
                docs.append(Document(
                    page_content="\n".join(batch_lines),
                    metadata={"source": file_path.name, "headers": headers},
                ))
                batch_lines = []

        if batch_lines:
            docs.append(Document(
                page_content="\n".join(batch_lines),
                metadata={"source": file_path.name, "headers": headers},
            ))

        if not docs:
            raise KnowledgeBaseError(f"CSV 文件中无数据行: {file_path.name}")
        return docs


class JsonLoader(BaseFileLoader):
    """.json 加载器，数组按元素展开，对象整体作为文本。"""

    @property
    def extensions(self) -> set[str]:
        return {".json"}

    def load(self, file_path: Path) -> list[Document]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise KnowledgeBaseError(f"JSON 解析失败 '{file_path.name}': {e}") from e

        if isinstance(data, list):
            docs = []
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    text = json.dumps(item, ensure_ascii=False, indent=2)
                else:
                    text = str(item)
                docs.append(Document(
                    page_content=text,
                    metadata={"source": file_path.name, "index": i},
                ))
            if not docs:
                raise KnowledgeBaseError(f"JSON 数组为空: {file_path.name}")
            return docs

        # 单个对象：格式化输出
        text = json.dumps(data, ensure_ascii=False, indent=2) if isinstance(data, dict) else str(data)
        return [Document(page_content=text, metadata={"source": file_path.name})]


# ══════════════════════════════════════════════════════════════
#  FileLoader — 统一入口
# ══════════════════════════════════════════════════════════════

class FileLoader:
    """统一文件加载入口，按扩展名分发到对应 Loader。支持 .md/.txt/.pdf/.docx/.csv/.json，最大 10 MB。"""

    MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024

    def __init__(self, upload_dir: str = "data/uploads") -> None:
        self._upload_dir = Path(upload_dir)
        self._upload_dir.mkdir(parents=True, exist_ok=True)
        self._loaders: dict[str, BaseFileLoader] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for loader_cls in [MarkdownLoader, PdfLoader, DocxLoader, CsvLoader, JsonLoader]:
            loader = loader_cls()
            for ext in loader.extensions:
                self._loaders[ext] = loader

    def register_loader(self, loader: BaseFileLoader) -> None:
        for ext in loader.extensions:
            self._loaders[ext] = loader

    @property
    def supported_extensions(self) -> set[str]:
        return set(self._loaders.keys())

    # ── 校验与保存 ────────────────────────────────────────────

    def validate(self, file_path: str | Path) -> Path:
        path = Path(file_path)
        if not path.exists():
            raise KnowledgeBaseError(f"文件不存在: {file_path}")
        ext = path.suffix.lower()
        if ext not in self._loaders:
            raise KnowledgeBaseError(
                f"不支持的文件类型 '{ext}'。支持: {', '.join(sorted(self.supported_extensions))}"
            )
        if path.stat().st_size > self.MAX_FILE_SIZE_BYTES:
            raise KnowledgeBaseError(
                f"文件过大: {path.stat().st_size} 字节（上限 {self.MAX_FILE_SIZE_BYTES}）"
            )
        return path

    def save(self, file_path: str | Path) -> Path:
        src = self.validate(file_path)
        dst = self._upload_dir / src.name
        counter = 1
        while dst.exists():
            dst = self._upload_dir / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        shutil.copy2(src, dst)
        logger.info(f"文件已保存: {src} → {dst}")
        return dst

    # ── 加载 ──────────────────────────────────────────────────

    def load(self, file_path: str | Path) -> list[Document]:
        path = Path(file_path)
        ext = path.suffix.lower()
        loader = self._loaders.get(ext)
        if not loader:
            raise KnowledgeBaseError(f"不支持的文件类型: {ext}")
        docs = loader.load(path)
        logger.debug(f"已加载 {len(docs)} 个文档，来源: {path.name}（{ext}）")
        return docs
