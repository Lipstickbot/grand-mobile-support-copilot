# RAG, LoRA and HTTP iteration — 2026-10-30

The new adapter is a research candidate. It did not earn replacement of the
existing working adapter: lower silver loss and better oracle citation coverage
coexist with more incomplete answers. Default base/explicit old adapter selection
is retained. All cases require a support employee; no automation is enabled.

## Training and artifacts

Windows, existing RTX3060Laptop6GB, local Soup0.77 disk streaming, Qwen3-8B frozen
base and rank8q/v LoRA.108balanced synthetic rows based on supplied rules;
RU/KK/UZ,12topics,3examples per topic/language. No historical admin targets;
no cloud, paid service, new download or global configuration change.
Exact formatter checks confirmed full assistant targets and masked system/user
labels, maximum510tokens at sequence512. Separate36silver references are retained.

All108optimizer steps completed. The first wrapper exited1 after saving because
it attempted a forward after Soup had released the streaming runtime. No training
was repeated. A fresh runtime verified108checkpoint/rewind records, finite logged
losses/gradient norms,144finite changed tensors, final/checkpoint equality, and
exact old/new reload. The fixed six silver probes, two per language, have mean
teacher-forced loss1.99912→1.49221. This does not establish language or support quality.

Candidate HF adapter SHA256:
`21324491c22168b9f56a657c0df0edcb1efd00d8d0ae393038582cc07eacc026`.
Official pinned llama.cpp converter produced a7.7MB BF16 adapter; all144mapped
tensor byte sequences match HF. GGUF SHA256:
`33cae1e0be841a271a0271771852df2e34425d07f6160ba2909439bf2db39d00`.
Base GGUF and old adapter are unchanged.

## Frozen first comparison

The same36AI-authored scenarios, corpus, base, inference code and case order were
verified by hashes.24cases require a source;12explicitly allow no citation.
`pipeline` uses real retrieval and topic filtering. `oracle` supplies authoritative
source IDs plus expected language/category hints; it isolates generation and is
not a deployable end-to-end result. Full means EOS plus successful normalization,
not factual correctness or a passed draft guard.

| Mode / adapter | Full generation /36 | Required raw citations /24 | Required effective citations /24 | All observable checks /36 | Median / sample p95, seconds |
| --- | ---: | ---: | ---: | ---: | --- |
| Pipeline / old |31|4|4|14|1.55 /3.30|
| Pipeline / candidate |29|4|4|13|1.48 /3.31|
| Oracle / old |34|17|17|27|1.80 /3.09|
| Oracle / candidate |32|20|19|28|1.46 /3.41|

No unknown or wrong-scope canonical citations were observed in this small set;
that is not proof of semantic grounding or safety. Missing context accounts for
many citation omissions. The required source appeared in top10retrieval in14/24
cases and survived the conservative topic gate in5/24. Event ambiguity is kept
unresolved rather than combining opposite armor rules. Keyword gating also
misses natural paraphrases: this remains an important recall limitation.

## Service corrections after comparison

Observed failures identified two service bugs: legitimate drafts beginning with a
quoted phrase were treated as malformed JSON; common Uzbek vocabulary was missing
from language detection. Quoted prose now remains text while incomplete objects,
lists, complete JSON strings and chat control tokens remain rejected. Language
checks are still heuristic. Frozen data and trained weights were not edited.
Because these scenarios informed diagnosis, reruns are development/regression
checks, not an untouched final test or a human quality benchmark.

The corrected old pipeline gives33/36full generations,4/24required citations,
16/36all observable constraints,36/36heuristic language matches and zero draft
guard rejections. Three token-limited generations remain. Timing: median1.53s,
sample p95 3.31s including failed attempts. These are not production/load SLOs.
The same corrected code with the candidate gives31/36full generations,
4/24required citations,15/36all constraints and36/36language matches;
five token-limited outputs remain. Median1.43s, sample p95 3.33s. Both versions
have zero observed unknown/wrong-scope canonical citations. The candidate is
rejected again; its shorter median does not compensate for missing full answers.

## HTTP and retrieval

New `/live`, `/ready`, `/v1/meta`, `/v1/rag/search` and namespaced rule lookup routes
are documented in [RAG.md](../../RAG.md). The existing `/health` and `/triage` remain.
Every response has a server-generated request ID; validation/internal errors do
not echo inputs. Retrieval and inference have separate single-request admission
gates; excess same-kind work returns503instead of queuing. Source lookup validates
section/ref, and unavailable sources return404. Missing account/payment FAQ is
reported explicitly; unrelated game rules are not substituted.

Actual E5+working8B adapter through HTTP TestClient: new routes200, source miss404,
RU/KK/UZ insult queries retrieve the expected general rule; payment query returns
no applicable rule. Search plus `/live` completed in0.062s during inference, while
the second `/triage` returned503busy. Four synthetic triages took
2.078/1.219/2.203/0.750s; cold load15.937s separately. Terminating only the owned
native child produced health503. This is an actual in-process ASGI HTTP check,
not a TCP load test or a quality benchmark.

Final actual TCP check ran the exact launcher command
`scripts/run-rag.ps1 -Action serve -AdapterVersion smoke -Port 8001`:
live/ready/meta/source/search/triage all200, redacted invalid search422,
request IDs present; synthetic triage2.782s. Ctrl+C completed app shutdown and
stopped its owned native child. This verifies local startup, not a load SLO.

Final full regression:185tests and33subtests passed, exit0. `pip check` passed.
Static native12-role configuration verification passed; its first sandbox temp
access failure was resolved by rerunning the same read-only verifier with local
runtime access. No additional Live model-quota test was requested or run.

Private evidence under `.private/rag-training-v2`: frozen `prepared-reviewed/`,
`training-preflight.json`, `saved-adapter-verification.json`,
`gguf-verification.json`, `comparison.json`, `final-comparison.json`,
six versioned evaluation JSONLs, `http-v2-working-adapter.json` and
`http-v2-tcp.json`. Raw inputs, source excerpts and model drafts
are kept private; public tests use invented fixtures.

The next quality work needs improved semantic retrieval/topic selection and
independent human annotations. No gold category/priority F1, calibrated confidence,
native-speaker language review, production security audit or60%automation is claimed.
