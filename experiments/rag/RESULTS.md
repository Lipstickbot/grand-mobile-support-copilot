# Local RAG verification — 2026-10-30

Historical Soup baseline. Serving was subsequently optimized; current runtime,
startup and measured results: [inference/RESULTS.md](../inference/RESULTS.md).

Status: working local research baseline; **hackathon product acceptance is not met**.
No paid calls, cloud uploads, deployment or additional training were performed.
Exact startup commands and remaining limits: [RAG.md](../../RAG.md).

## Data and retrieval

All six pinned E5 author files verified by SHA-256; safetensors loaded with remote
code disabled. The actual corpus produced 288 documents: 266 points, 14 terms,
eight context records, 11 sections. CPU E5 + BM25 + RRF; cached vectors reused.
Six authored synthetic queries found the expected top result: insults, cheats,
green zone, family event, business battle and van event. This is not a representative
retrieval benchmark. Opposite armor rules remain in their correct sections.

A/B retrieval inspection exposed unrelated candidates for payments: the handout
has no payment FAQ. Final TriageEngine adds a conservative topic whitelist and
passes no game rules for payments/account recovery. Its recall is not estimated.
The A/B experiment below evaluated generation with raw retrieval before that
engine-only filter; it is not a full engine benchmark.

## Actual Qwen3-8B / LoRA full-answer comparison

RTX 3060 Laptop 6 GB, local NVMe; Soup CLI 0.77.0 NF4 disk streaming, two buffers,
no KV cache; deterministic decoding, max 160 new tokens, input ≤1024. Original
64-step smoke adapter reloaded without retraining. Three new synthetic inputs,
one per RU/KK/UZ, identical passages for both modes; fixed base-first order,
uncontrolled warmup. Diagnostic deadline 180 s, separate from server deadline 45 s.

| Mode | RU seconds | KK seconds | UZ seconds | Complete EOS outputs |
|---|---:|---:|---:|---:|
| Base | 68.203 | 184.406, timeout | 56.235 | 2 / 3 |
| LoRA | 52.875 | 95.703 | 59.047 | 3 / 3 |

Median attempt: base 68.203 s, adapted 59.047 s. Nearest-rank p95 of three attempts
is their maximum: 184.406 / 95.703 s. Includes the failed base attempt; **not a
production p95 estimate**. Five outputs reached EOS and passed current pattern
checks, which is not human approval or proof of grounding/language correctness.
Successful raw drafts contained no validated rule citations. No model/citation
quality score or established LoRA improvement is claimed.

Earlier 45-second probe: three base timeouts, 45.937, 45.891, 45.281 s. Deliberately
stopped before remaining cases; partial records preserved. Full comparison exited
**1** because one generation failed. Deadlines are cooperative between tokens;
an in-flight CUDA forward may exceed them. Neither run meets five-second latency.

## Integration and checks

- Whole regression suite: **85 passed, 17 subtests passed**, exit 0.
- `pip check`: exit 0, no broken installed requirements.
- `scripts/verify-team.ps1`: exit 0; all 12 native roles and strict config verified.
  Initial sandbox attempt failed on Codex temp permissions; required local-access
  rerun passed. Native configuration unchanged.
- Real E5/rules + HTTP adapter (ASGI TestClient), explicit template mode: four
  requests, 0.031–0.297 s excluding model load. No LLM speed/quality inference.
- Actual TCP `127.0.0.1:8001`: health and synthetic triage returned 200, template/E5
  request 0.21 s. Server stopped cleanly after verification.
- Actual 8B + E5 + HTTP adapter: one request **45.531 s**, generation timeout,
  HTTP 200 with Russian fallback, `needs_human=true`, `generation_error=timeout`.
  This passes fallback integration and **fails full generation / five-second latency**.
- Actual batch CLI: four synthetic records, ordered output, RU/UZ/KK language
  routes, insult/payment/account/help categories, payment/account p0, all human.
  Explicit templates used; actual batch LLM generation not run.
- Final rerun after topic gating/nickname fix: four real E5 HTTP template requests,
  0.015–0.954 s; four batch rows, nickname punctuation fixed, no payment/account
  rule context. Targeted policy/engine tests: 22 passed, exit 0. No repeat of the
  entire suite after this isolated punctuation fix.

Windows sandbox blocked TestClient at socketpair before app code; reran checks
as local processes with loopback access and workspace TEMP/TMP. Initial downloader
tests failed on sandbox system-temp permissions; final suite passed with local temp.

## Evidence and open work

Ignored private local evidence, potentially containing full model input/output:

- `.private/rag-retrieval-smoke.json`
- `.private/rag-ab-01.jsonl` — intentionally interrupted 45 s probe
- `.private/rag-ab-full-01.jsonl` — six-attempt comparison, manifest, summary
- `.private/http-template-01.json`, `.private/rag-tcp-template-01.json`
- `.private/http-llm-01.json`, `.private/rag-batch-template-01.jsonl`
- `.private/http-template-final.json`, `.private/rag-batch-template-final.jsonl`
- `.team-verification/verification.json`

Model processes closed and GPU released after checks. Base weights, LoRA and
original training artifacts retained. Environment snapshot:
`experiments/rag/requirements-lock.txt`. Clean-machine installation, vulnerability
database audit, independent final QA/security review, load tests and deployment
not performed. Root integrated specialist code and ran final checks after specialist
quota errors. A passed regex guard is not a security or semantic correctness proof.

Next: optimize serving separately from disk-stream training; create genuinely
labeled leakage-free RU/KK/UZ evaluation; obtain approved support FAQ; improve
classification and grounding; then measure automation thresholds. Current human
coverage is 100%, automation coverage 0%; category F1, calibration, 60% automation
and p95<5 s unproven. UI, persistent multi-process dedup and deployment are outside
this research baseline.
