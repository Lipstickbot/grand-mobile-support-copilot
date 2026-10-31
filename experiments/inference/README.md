# Local inference optimization

This is serving work, not additional model training. Original HF weights, Soup
cache, adapter and previous evidence are retained. Zero paid calls or data uploads.

Implemented explicit runtime choices:

- `stream`: original Soup0.77 NF4/NVMe; no KV cache, preserved for reproduction.
- `resident`: Transformers5.22/bitsandbytes NF4; CUDA-only placement, BF16 embedding,
  explicitly quantized output head, KV cache and exact PEFT adapter reload checks.
- `llamacpp`: managed official b11960 CUDA12.4 process; local Q4_K_M base and BF16
  LoRA, native CUDA Flash Attention, KV cache, one1536-token slot, loopback auth.

## Artifact preparation

Current project `.venv-soup` is used. Do not reinstall or retrain to serve existing
artifacts. Official converter needs sentencepiece<0.3 and protobuf>=4.21,<5; dependencies
and actual tested versions are recorded with the results. Converter uses its
verified source gguf-py package through PYTHONPATH, no global package installation.

```powershell
# Verify/install pinned runtime/source; reuse the existing PyTorch CUDA12 DLLs.
.\.venv-soup\Scripts\python.exe -X utf8 scripts/prepare_llamacpp.py --use-torch-cuda --skip-model
# CPU conversion of existing HF base → new BF16 GGUF → new Q4_K_M GGUF and LoRA.
.\.venv-soup\Scripts\python.exe -X utf8 scripts/convert_local_gguf.py
```

Conversion needs at least30GiB free and refuses to overwrite existing outputs.
`--resume-base` can validate/reuse a completed399-tensor BF16 intermediate after
a later-stage failure, while still refusing to overwrite Q4/LoRA outputs.
Existing training weights/adapter are not edited. A failed converter may leave
partial outputs: inspect logs and preserve them before choosing new destinations.
The BF16 intermediate is retained for reproducibility. No automatic cleanup.

Runtime zip SHA256 comes from GitHub release metadata; extracted source files
are checked against Git blob hashes at commit
`8642541b255bfb5dc39cd351a178be09013a55d6`. CUDA DLL hashes from the installed
project PyTorch are recorded. Native `--list-devices` verified the RTX3060 device.
Full-model offload and actual speed must additionally be verified in runtime logs.

GGUF is generated from the **same pinned local HF base** as training. Format,
quantization and kernels differ from Soup, so numerical equivalence/quality is
not assumed. The official converter maps LoRA tensors; loaded adapter influence
and complete answers must be checked separately.

Private manifests/logs: `.private/llamacpp-setup/manifest.json`,
`local-conversion.json`, `convert-*.log`, `.private/runtime/llamacpp-startup-*.log`.
They are ignored by Git. Neither HF weights nor raw ticket texts are published.

## Use and reproduction

```powershell
.\scripts\run-rag.ps1 -Action serve -Backend llamacpp -UseAdapter
.\scripts\run-rag.ps1 -Action serve -Backend resident -UseAdapter
.\scripts\run-rag.ps1 -Action serve -Backend stream -UseAdapter

.\.venv-soup\Scripts\python.exe scripts/compare_rag_answers.py --backend llamacpp --output .private/fast-ab-new.jsonl --case-ids ru-chat-01 kk-chat-04 uz-payment-07
```

RAG API remains `http://127.0.0.1:8000/docs`, backend uses private port8081 with
ephemeral authentication. Run one model process at a time. The wrapper starts
and stops its own native child; occupied port means failure, never attachment to
someone else's server. Local auth key is not included in command arguments or
public health; its temporary file is deleted on normal managed shutdown.

`COPILOT_BACKEND=stream|resident|llamacpp`, `COPILOT_GGUF_PATH`,
`COPILOT_GGUF_ADAPTER_PATH`, `COPILOT_LLAMA_BINARY_PATH`, `COPILOT_LLAMA_PORT`
configure selection/paths. Backend failures do not silently change runtime/model.
`-Backend` overrides the environment for that launcher invocation only.

Raw prompt uses original HF tokenizer/template, nonthinking mode. Context remains
≤1024 tokens and output≤160; omitted passages are recorded, player text is never
silently truncated. Complete EOS required; native context/token limits remain
failures. Base/adapted modes select adapters per request, no global endpoint race.
Prompt reuse is disabled in timing runs (`cache_prompt=false`); ordinary active
KV caching remains enabled. Timeouts are reported, never counted as full answers.

No labeled category/priority test set, calibrated confidence, multilingual human
quality score or automation acceptance is established by a latency experiment.
All drafts still need human review. Full pipeline timing includes retrieval and
guarding; isolated generation timing does not establish HTTP p95 under load.

## Primary sources

- [Pinned llama.cpp release and checksums](https://github.com/ggml-org/llama.cpp/releases/tag/b11960)
- [Pinned official server documentation](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/README.md)
- [Official LoRA converter](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/convert_lora_to_gguf.py)
- [Transformers bitsandbytes](https://huggingface.co/docs/transformers/quantization/bitsandbytes)
