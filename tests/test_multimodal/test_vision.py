from __future__ import annotations

from types import SimpleNamespace

import pytest

from agentkb.multimodal.vision import VisionService, detect_image_media_type


class FakeSettings:
    vision_max_image_size_mb = 1


class FakeVisionModel:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content="图片包含一张流程图。")


def test_detect_image_media_type_uses_file_signature():
    assert detect_image_media_type(b"\xff\xd8\xffpayload") == "image/jpeg"
    assert detect_image_media_type(b"\x89PNG\r\n\x1a\npayload") == "image/png"
    assert detect_image_media_type(b"RIFF1234WEBPpayload") == "image/webp"


def test_detect_image_media_type_rejects_extension_spoofing():
    with pytest.raises(ValueError, match="仅支持"):
        detect_image_media_type(b"not-an-image")


def test_vision_service_builds_multimodal_message():
    model = FakeVisionModel()
    service = VisionService(model=model, settings=FakeSettings())

    result = service.analyze(b"\xff\xd8\xffpayload")

    assert result.description == "图片包含一张流程图。"
    content = model.messages[0].content
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith(
        "data:image/jpeg;base64,"
    )
