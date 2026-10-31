"""Provider selection is explicit and never silently falls back to another runtime."""
from dataclasses import replace
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from support_copilot.config import Settings
from support_copilot.providers import create_generator


@pytest.mark.parametrize("backend", ["stream", "resident", "llamacpp"])
@pytest.mark.parametrize("adapter", [False, True])
def test_factory_uses_requested_backend_paths_and_adapter_flag(monkeypatch, tmp_path, backend, adapter):
    factories = {name: Mock(name=name) for name in ("stream", "resident", "llamacpp")}
    import support_copilot.generation as generation
    monkeypatch.setattr(generation, "LocalGenerator", factories["stream"])
    monkeypatch.setitem(sys.modules, "support_copilot.resident", SimpleNamespace(ResidentGenerator=factories["resident"]))
    monkeypatch.setitem(sys.modules, "support_copilot.llamacpp", SimpleNamespace(LlamaCppGenerator=factories["llamacpp"]))
    settings = replace(Settings(), inference_backend=backend, use_adapter=adapter,
                       model_path=tmp_path / "hf", adapter_path=tmp_path / "adapter",
                       gguf_path=tmp_path / "model.gguf", gguf_adapter_path=tmp_path / "adapter.gguf",
                       llama_binary_path=tmp_path / "llama-server.exe", llama_port=18081,
                       runtime_dir=tmp_path / "runtime", max_new_tokens=99, generation_timeout_seconds=25)
    result = create_generator(settings)
    assert result is factories[backend].return_value
    expected = {"model_path": settings.model_path, "adapter_path": settings.adapter_path if adapter else None,
                "runtime_dir": settings.runtime_dir, "max_new_tokens": 99,
                "timeout_seconds": 25, "device": settings.device, "dtype": settings.model_dtype}
    if backend == "llamacpp":
        expected.update(gguf_path=settings.gguf_path, gguf_adapter_path=settings.gguf_adapter_path,
                        binary_path=settings.llama_binary_path, port=18081)
    factories[backend].assert_called_once_with(**expected)
    for name in factories:
        if name != backend:
            factories[name].assert_not_called()


def test_unknown_backend_fails_instead_of_falling_back():
    with pytest.raises(ValueError, match="Unknown inference backend"):
        create_generator(replace(Settings(), inference_backend="typo"))


@pytest.mark.parametrize("backend", ["stream", "resident", "llamacpp"])
def test_environment_selects_only_supported_backend(monkeypatch, backend):
    monkeypatch.setenv("COPILOT_BACKEND", backend)
    assert Settings.from_env().inference_backend == backend


def test_unknown_backend_environment_is_rejected(monkeypatch):
    monkeypatch.setenv("COPILOT_BACKEND", "typo")
    with pytest.raises(ValueError, match="COPILOT_BACKEND"):
        Settings.from_env()
