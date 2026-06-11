from __future__ import annotations

from agentkb.knowledge.loader import FileLoader
from agentkb.multimodal.vision import ImageAnalysis


class FakeSettings:
    vision_pdf_visual_analysis = False
    vision_pdf_max_pages = 0


class FakeVisionService:
    def analyze(self, content: bytes, *, prompt: str | None = None):
        return ImageAnalysis(
            description="图中展示 Agent 调度流程。",
            media_type="image/png",
        )


def test_image_loader_converts_image_to_searchable_document(temp_dir):
    path = temp_dir / "architecture.png"
    path.write_bytes(b"\x89PNG\r\n\x1a\npayload")
    loader = FileLoader(
        settings=FakeSettings(),
        vision_service=FakeVisionService(),
    )

    documents = loader.load(path)

    assert len(documents) == 1
    assert documents[0].page_content == "图中展示 Agent 调度流程。"
    assert documents[0].metadata["visual_analysis"] is True
