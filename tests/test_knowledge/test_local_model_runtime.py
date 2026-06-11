from __future__ import annotations

import importlib
import sys
import types


def test_embedder_disables_safetensors_network_conversion(monkeypatch):
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)
    import agentkb.knowledge.embedder as embedder

    importlib.reload(embedder)

    assert embedder.os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"


def test_local_reranker_disables_safetensors_network_conversion(monkeypatch):
    monkeypatch.delenv("DISABLE_SAFETENSORS_CONVERSION", raising=False)

    class FakeCrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

    fake_module = types.SimpleNamespace(CrossEncoder=FakeCrossEncoder)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    from agentkb.knowledge.reranker import LocalReranker

    LocalReranker(model_name="local-model")

    import os

    assert os.environ["DISABLE_SAFETENSORS_CONVERSION"] == "1"
