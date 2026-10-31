# Verified inference optimization — 2026-10-30

Default: llama.cpp b11960 CUDA12.4, local Q4_K_M, KV cache, native Flash Attention.
No retraining, paid calls, cloud input uploads or global config changes.
Original HF weights, LoRA, Soup cache and all earlier measurements preserved.

```powershell
.\scripts\run-rag.ps1 -Action serve -UseAdapter
```

Then open `http://127.0.0.1:8000/docs`. Stop with Ctrl+C. Without `-UseAdapter`,
the original base is used. [Full startup instructions](../../RAG.md).

## Measured full answers

Same authored synthetic RU/KK/UZ cases, original HF template, nonthinking,
max1024 input/160 output tokens, fixed base-first order, no prompt reuse.
Cold loads excluded. Quantization and kernels differ from the old Soup runtime.

| Runtime / mode | Samples | Full EOS | Median attempt | Nearest-rank p95 |
|---|---:|---:|---:|---:|
| Soup NF4/stream, LoRA | 3 | 3 | 59.047 s | 95.703 s |
| Transformers resident NF4/KV, LoRA | 3 | 3 | 15.407 s | 19.078 s |
| llama.cpp Q4_K_M/KV, base | 8 | 8 | 3.125 s | 5.219 s |
| llama.cpp Q4_K_M/KV, LoRA | 8 | 8 | 3.071 s | 4.047 s |

Corrected native run: exit0, no generation failures. Adapted range **2.282–4.047s**.
On the three identical historical prompts,52.875/95.703/59.047s became
3.125/3.734/2.875s: observed **16.9–25.6×** speedup. Different sample sizes mean
cross-row medians are not controlled comparisons. Eight samples do not establish
production p95; base had one response above5s. One injection/payment draft per
native mode was rejected by the guard. A guard pass is not human quality approval.

## Adapter verified and control corrected

All144 BF16 GGUF tensors match original safetensors byte-for-byte:72pairs, alpha16.
Base GGUF was converted from the same pinned local HF checkpoint used for training.
Explicit0→1 probe changed common-token logprobs by1.1162945; repeated0 drift0.0.
All eight corrected base/adapted texts differed; this proves influence, not quality.

Found b11960 discrepancy: `lora:[]` falls back to global scale1 even with
`--lora-init-without-apply`. First native run used LoRA in both requested modes:
**its base control is invalid**, records retained. Fixed requests explicitly send
scale0 or1 and record/check provider acknowledgement. Regression test added.
Source: [empty-map fallback](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/tools/server/server-context.cpp#L1794),
[initial-apply flag](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/common/common.cpp#L1404).

## Integrated checks

- Real E5 + topic gate + LoRA + guards + HTTP ASGI: four requests **3.844 /2.000
  /3.671 /1.219s**, all EOS, no fallback. Cold engine load separately22.235s.
- Actual TCP through PowerShell launcher/FastAPI: health200, triage200,
  **3.95s** synthetic complaint. Temporary server stopped cleanly with Ctrl+C.
- Real batch/E5/LoRA: four ordered outputs,3.781/1.984/3.625/1.203s engine times,
  no generation errors/fallbacks; atomic output completed.
- Terminating only owned native child changed live HTTP health to503. Fixed the
  security review's cached-ready defect and covered it with a regression test.
- **124 tests +17 subtests PASS**, exit0; pip checkPASS; native12-role strict
  configuration verificationPASS. No native agent config changed.
- GPU observed4883MiB/98%, decoded approximately29–30tokens/s. All GPU layers
  requested, auto-fit disabled. Default logs omit exact layer count;37/37 log
  confirmation not claimed. HF/native prompt counts matched16/16 cases.

## Preparation and evidence

Official runtime/source commit8642541b255bfb5dc39cd351a178be09013a55d6.
CUDA12.4 zip SHA2568d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684.
Source Git blobs verified. Existing project PyTorch CUDA12 DLLs reused only in
child PATH; --list-devices verified GPU compatibility. No global installation.
Local Q4 GGUF5,027,783,712bytes; SHA256e65ebd5f4c5e8c9f4aa169273a62c56b64f29f3f5210209da24535abeb19103d.
LoRA GGUF7,677,984bytes; SHA256e3f028279ba5aef0535734355b399339f657053f3daaa7501613ea174df8d21c.
BF16 intermediate retained. SentencePiece0.2.2/Protobuf4.25.9 installed locally.
Qwen conversion verified with existing Transformers5.22/NumPy2.5; upstream
generic converter pins older versions, so other architectures are not asserted.
Observed environment: requirements-lock.txt; clean-machine install not tested.

First conversion failed on missing SentencePiece; fixed dependency. Quantizer
then failed on Cyrillic absolute paths; fixed native ASCII-relative arguments.
Validated/reused completed BF16 intermediate, preserved originals and failed logs.
CUDA/model partial downloads stopped/preserved, never used; base converted locally.

Private artifacts, ignored by Git:

- `.private/llamacpp-setup/{manifest,local-conversion,adapter-verification}.json`
- `.private/rag-resident-ab-01.jsonl`
- `.private/rag-llamacpp-ab-01.jsonl` — invalid base control, retained
- `.private/native-adapter-probe-01.json` — empty-list diagnosis
- `.private/native-adapter-probe-02.json` — validated explicit0/1
- `.private/rag-llamacpp-ab-02.jsonl` — corrected16 full answers
- `.private/http-llamacpp-final.json`, `.private/rag-tcp-llamacpp-final.json`
- `.private/rag-batch-llamacpp-final.jsonl`, `.private/runtime/llamacpp-startup-*.log`

All drafts still require human review, automation0%. Quality, grounding and
RU/KK/UZ correctness require human-labeled evaluation. No load/soak test,
production p95, clean-machine reinstall, dependency vulnerability scan, UI or
deployment. Native task cancellation after client timeout remains unverified;
one slot/max160 output tokens bound work. Private auth key is deleted on normal
shutdown; default logging is not intended to include request text.
