from __future__ import annotations

import pytest

from agentkb.api.routes import _create_upload_path, _validate_upload


def test_upload_path_cannot_escape_upload_directory(temp_dir):
    upload_dir = temp_dir / "uploads"

    original_name, saved_path = _create_upload_path(
        "../../outside/report.txt",
        upload_dir,
    )

    assert original_name == "report.txt"
    assert saved_path.parent == upload_dir.resolve()
    assert saved_path.name != original_name
    assert saved_path.suffix == ".txt"


def test_upload_path_normalizes_windows_client_path(temp_dir):
    original_name, saved_path = _create_upload_path(
        r"C:\Users\user\secret.pdf",
        temp_dir,
    )

    assert original_name == "secret.pdf"
    assert saved_path.parent == temp_dir.resolve()


def test_upload_validation_rejects_unsupported_extension() -> None:
    with pytest.raises(ValueError, match="不支持的文件类型"):
        _validate_upload("payload.exe", b"data", [".txt"], 1)


def test_upload_validation_rejects_oversized_file() -> None:
    with pytest.raises(ValueError, match="文件超过"):
        _validate_upload("notes.txt", b"x", [".txt"], 0)
