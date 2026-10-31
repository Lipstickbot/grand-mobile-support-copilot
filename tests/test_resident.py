"""Resident provider behavior using CPU-only doubles; no model, torch or GPU load."""
from contextlib import nullcontext
from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from support_copilot.generation import GenerationError
from support_copilot.resident import ResidentGenerator


class Tokens:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values), len(values[0]))

    def __getitem__(self, key):
        row, columns = key
        return SimpleNamespace(tolist=lambda: self.values[row][columns])


class Batch(dict):
    def to(self, device):
        return self


class Tokenizer:
    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, **kwargs):
        return "rendered synthetic prompt"

    def __call__(self, prompt, **kwargs):
        return Batch(input_ids=Tokens([[10] * 20]))

    def decode(self, ids, skip_special_tokens):
        return "Уточните время и сервер для проверки человеком."


class QuantizedHead:
    pass


class Parameter:
    def __init__(self, device="cuda"):
        self.device = SimpleNamespace(type=device)
        self.trainable = True

    def requires_grad_(self, enabled):
        self.trainable = enabled


class RawModel:
    """Deliberately has no PEFT disable_adapter method."""
    def __init__(self, *, quantized=True, parameter_device="cuda"):
        self.config = SimpleNamespace(use_cache=False)
        self.generation_config = SimpleNamespace(eos_token_id=2)
        self.params = [Parameter(parameter_device)]
        self.head = QuantizedHead() if quantized else object()

    def get_output_embeddings(self):
        return self.head

    def parameters(self):
        return iter(self.params)

    def eval(self):
        return self

    def get_memory_footprint(self):
        return 123456

    def generate(self, **kwargs):
        self.generation_kwargs = kwargs
        return Tokens([[10] * kwargs["input_ids"].shape[-1] + [50, 2]])


def fake_torch():
    return SimpleNamespace(
        bfloat16="fake-bfloat16", OutOfMemoryError=type("FakeOutOfMemory", (RuntimeError,), {}),
        inference_mode=nullcontext, autocast=lambda *args, **kwargs: nullcontext(),
        manual_seed=lambda seed: None, isfinite=lambda value: SimpleNamespace(all=lambda: True),
        equal=lambda left, right: left == right,
        cuda=SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                             empty_cache=Mock(), memory_allocated=lambda: 100,
                             memory_reserved=lambda: 200, max_memory_allocated=lambda: 300))


def install_load_doubles(monkeypatch, tmp_path, *, quantized=True, parameter_device="cuda", adapter=False):
    model = RawModel(quantized=quantized, parameter_device=parameter_device)
    torch = fake_torch()
    config = Mock(side_effect=lambda **kwargs: kwargs)
    model_factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=Tokenizer())),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_factory), BitsAndBytesConfig=config,
        StoppingCriteriaList=list))
    monkeypatch.setitem(sys.modules, "bitsandbytes", SimpleNamespace(nn=SimpleNamespace(Linear4bit=QuantizedHead)))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    (tmp_path / "config.json").write_text("{}", encoding="utf8")
    g = ResidentGenerator(tmp_path, tmp_path / "adapter" if adapter else None)
    return g, model, torch, config, model_factory


def test_resident_raw_hf_no_adapter_uses_kv_without_peft_context(monkeypatch, tmp_path):
    g, model, _, quant, factory = install_load_doubles(monkeypatch, tmp_path)
    g.load()
    assert g.generate("Synthetic question", "ru", [])
    assert not hasattr(model, "disable_adapter")
    assert model.generation_kwargs["use_cache"] is True
    assert g.last_generation["use_cache"] is True
    assert g.last_generation["adapter_enabled"] is False
    assert g.last_generation["finish_reason"] == "eos"
    assert model.config.use_cache is True
    assert all(not parameter.trainable for parameter in model.params)
    options = quant.call_args.kwargs
    assert options["load_in_4bit"] and options["bnb_4bit_use_double_quant"]
    assert options["bnb_4bit_quant_type"] == "nf4" and options["llm_int8_skip_modules"] == []
    load = factory.call_args.kwargs
    assert load["device_map"] == {"": 0}
    assert load["attn_implementation"] == "sdpa"
    assert load["local_files_only"] is True and load["trust_remote_code"] is False
    assert load["use_safetensors"] is True
    assert g.stats()["backend"] == "transformers-bnb-resident-nf4"
    assert g.stats()["cpu_or_disk_offload"] is False
    g.close()
    assert g.ready is False


@pytest.mark.parametrize("kwargs", [{"device": "cpu"}, {"dtype": "float16"}])
def test_unsupported_resident_device_dtype_fail_before_torch_import(kwargs):
    g = ResidentGenerator("unused", **kwargs)
    with pytest.raises(GenerationError) as exc:
        g.load()
    assert exc.value.code == "resident_requires_cuda_bfloat16"
    assert not g.ready


@pytest.mark.parametrize("quantized,device,expected", [
    (False, "cuda", "resident_output_head_not_quantized"),
    (True, "cpu", "resident_unexpected_offload"),
])
def test_resident_rejects_unquantized_head_or_silent_cpu_dispatch(monkeypatch, tmp_path, quantized, device, expected):
    g, _, torch, _, _ = install_load_doubles(monkeypatch, tmp_path, quantized=quantized, parameter_device=device)
    with pytest.raises(GenerationError) as exc:
        g.load()
    assert exc.value.code == expected
    assert g.ready is False and g._model is None
    torch.cuda.empty_cache.assert_called_once()


def test_adapter_state_mismatch_fails_closed(monkeypatch, tmp_path):
    g, model, _, _, _ = install_load_doubles(monkeypatch, tmp_path, adapter=True)
    peft_load = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=peft_load),
        get_peft_model_state_dict=lambda value: {"unexpected_weight": 1}))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=lambda *args, **kwargs: {"expected_weight": 1}))
    with pytest.raises(GenerationError) as exc:
        g.load()
    assert exc.value.code == "adapter_reload_mismatch"
    assert not g.ready and g._model is None
    assert peft_load.call_args.kwargs == {
        "local_files_only": True, "is_trainable": False, "autocast_adapter_dtype": False}


def test_nonfinite_adapter_is_rejected_before_peft_load(monkeypatch, tmp_path):
    g, _, torch, _, _ = install_load_doubles(monkeypatch, tmp_path, adapter=True)
    torch.isfinite = lambda value: SimpleNamespace(all=lambda: False)
    peft_load = Mock()
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=peft_load), get_peft_model_state_dict=Mock()))
    monkeypatch.setitem(sys.modules, "safetensors.torch", SimpleNamespace(load_file=lambda *args, **kwargs: {"weight": 1}))
    with pytest.raises(GenerationError) as exc:
        g.load()
    assert exc.value.code == "invalid_adapter_tensors"
    peft_load.assert_not_called()
    assert not g.ready


def test_resident_cuda_allocation_error_is_sanitized(monkeypatch, tmp_path):
    g, _, torch, _, factory = install_load_doubles(monkeypatch, tmp_path)
    factory.side_effect = torch.OutOfMemoryError("PRIVATE provider detail")
    with pytest.raises(GenerationError) as exc:
        g.load()
    assert exc.value.code == "resident_out_of_memory"
    assert "PRIVATE" not in str(exc.value)
    assert not g.ready
