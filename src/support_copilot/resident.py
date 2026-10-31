"""Offline resident NF4 inference; public Transformers/bitsandbytes/PEFT APIs only.

The output head is explicitly quantized to fit 8B on 6 GiB; embeddings stay BF16.
No CPU/disk offload, network download, retraining or writes to model/adapter files.
"""
from __future__ import annotations

import os
import time

from .generation import LocalGenerator, GenerationError, _GPU_LOCK


class ResidentGenerator(LocalGenerator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_cache = True
        self._footprint = None

    def load(self):
        with _GPU_LOCK:
            if self.ready:
                return
            started = time.monotonic()
            os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                              HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                              DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false",
                              HF_ENABLE_PARALLEL_LOADING="false")
            if self.device != "cuda" or self.dtype != "bfloat16":
                raise GenerationError("resident_requires_cuda_bfloat16")
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                raise GenerationError("resident_cuda_bfloat16_unavailable")
            if not (self.model_path / "config.json").is_file():
                raise GenerationError("local_model_missing")
            self._torch = torch
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True,
                                                          trust_remote_code=False)
                quantization = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    # Explicit [] bypasses the default lm_head exclusion in Transformers 5.22.
                    llm_int8_skip_modules=[])
                model = AutoModelForCausalLM.from_pretrained(
                    str(self.model_path), quantization_config=quantization,
                    device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa",
                    local_files_only=True, trust_remote_code=False, use_safetensors=True)
                self._model, self._tokenizer = model, tokenizer
                import bitsandbytes as bnb
                if not isinstance(model.get_output_embeddings(), bnb.nn.Linear4bit):
                    raise GenerationError("resident_output_head_not_quantized")
                if any(p.device.type != "cuda" for p in model.parameters()):
                    raise GenerationError("resident_unexpected_offload")
                if self.adapter_path:
                    from peft import PeftModel, get_peft_model_state_dict
                    from safetensors.torch import load_file
                    saved = load_file(str(self.adapter_path / "adapter_model.safetensors"), device="cpu")
                    if not saved or any(not bool(torch.isfinite(value).all()) for value in saved.values()):
                        raise GenerationError("invalid_adapter_tensors")
                    model = PeftModel.from_pretrained(model, str(self.adapter_path),
                                                      local_files_only=True, is_trainable=False,
                                                      autocast_adapter_dtype=False)
                    self._model = model
                    actual = get_peft_model_state_dict(model)
                    if set(actual) != set(saved) or any(
                            not torch.equal(actual[key].detach().cpu().to(value.dtype), value)
                            for key, value in saved.items()):
                        raise GenerationError("adapter_reload_mismatch")
                model.eval()
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
                model.config.use_cache = True
                self._footprint = model.get_memory_footprint()
                self._load_seconds = time.monotonic() - started
                self.ready = True
            except Exception as exc:
                self._close_unlocked()
                if isinstance(exc, GenerationError):
                    raise
                code = "resident_out_of_memory" if isinstance(exc, torch.OutOfMemoryError) else "resident_load_failure"
                raise GenerationError(code) from exc

    def stats(self):
        data = super().stats()
        data.update(backend="transformers-bnb-resident-nf4", use_cache=True,
                    quantized_lm_head=True, cpu_or_disk_offload=False,
                    model_memory_bytes=self._footprint)
        if self._torch is not None and self.ready:
            data.update(cuda_allocated_bytes=self._torch.cuda.memory_allocated(),
                        cuda_reserved_bytes=self._torch.cuda.memory_reserved(),
                        cuda_peak_allocated_bytes=self._torch.cuda.max_memory_allocated())
        return data
