# RAG-conditioned candidate data v2

AI is needed for the accepted multilingual support drafts. This preparation stage
creates a bounded **synthetic/silver** candidate, not a validated support model.
No training, GPU work, network calls or paid services occur in the data script.

The private AI-authored recipe has 12 topics, RU/KK/UZ questions, short plain-text
assistant drafts and separately authored challenge scenarios. Real rule excerpts,
translations and paraphrases remain only under ignored `.private`; none are
embedded in public scripts/tests. Historical admin replies are never targets.
The original archive's player text is hashed solely to exclude accidental overlap;
official validation is not used to train, label or tune thresholds. No historical
ticket is imported. Native-speaker review is still required for KK/UZ.

## Prepare

The existing project environment supplies Rust tokenizers and the Jinja sandbox.
The script renders the verified local Qwen template without importing Torch or
Transformers. It never downloads a tokenizer. Each complete prompt plus assistant
sequence must fit512tokens. Only source passage excerpts may shrink, with an
explicit excerpt marker; player text/targets are never silently truncated.

Current accepted private preparation is frozen under
`.private/rag-training-v2/prepared-reviewed/`:288train,108balanced candidate,
36holdout and36evaluation rows. Token lengths:train221–510, holdout241–506.
The first exploratory `prepared/` directory is retained; use the reviewed paths
for this candidate. Future corrections require a new directory/version.

```powershell
.\.venv-soup\Scripts\python.exe -X utf8 scripts/prepare_rag_training.py --output .private/rag-training-v2/prepared-new
```

Defaults read `.private/rag-training-v2/recipe.json`, `.private/data/rules.json`,
the original Downloads/handout.zip and `.private/models/qwen3-8b` tokenizer.
Output must be a new directory under `.private`; existing output is refused.
Recipe generation is private and is not shipped as a reusable public training set.

Files:

- `train.jsonl`:288 ChatML records,96perlanguage;12topics ×3languages ×8wrappers.
- `train108.jsonl`:balanced3records pertopic/language for bounded candidate work.
- `train64.jsonl`:smaller balanced smoke subset.
- `holdout.jsonl`:36 ChatML references from independent scenario texts.
- `evaluation.jsonl`:same36 scenarios with request, required/allowed/forbidden
  canonical citations, human-review/safety constraints and prepared context.
- `train_metadata.jsonl`, `holdout_metadata.jsonl`, `manifest.json`:synthetic source
  identities, normalized-text hashes, counts, sequence lengths and file hashes.

Targets are plain drafts matching the inference prompt, not the previous four-field
JSON smoke target. Rule citations are namespaced; missing payment/account/appeal
FAQ is not replaced with unrelated game rules. Contradictory event rules and
prompt-injection examples are included. No punishments, completed remedies or
claims of reviewed attachments are taught as valid actions.

## Evaluation limits

The36 scenarios use different authored text and target wording from training
wrappers. Normalized player text and synthetic source identities are disjoint.
Rules/topics intentionally overlap: this is not unseen-rule generalization.
Templates remain repetitive, and holdout loss can measure template imitation.
Constraint checks are not human gold, category F1, calibrated confidence or proof
of semantic grounding. Do not promote a candidate merely because its loss falls.
Root must verify exact Soup formatter/masking/token lengths and compare the new
adapter on full responses, citations, safety, RU/KK/UZ and latency before promotion.

Tests use only invented source fixtures:

```powershell
.\.venv-soup\Scripts\python.exe -m pytest tests/test_rag_training.py -q --basetemp=.private/test-rag-training
```

Executed preparation and tests on2026-10-30:7tests passed, exit0; archive exclusion
hashed118337train/1000valid player records; generated official-ticket targets0.
Explicit import audit confirmed Torch was not imported. The full GZ response-fire
exception remains in all three heldout language contexts. Readonly policy checks
found two Uzbek references rejected solely by the existing language heuristic;
that is a service-language gap to fix, not a reason to rewrite frozen evaluation.

## Bounded local continuation

`continue108.yaml` continues the old64-step adapter on the balanced108-row set,
with a fresh optimizer, one epoch, rank8 q/v, alpha16, lr5e-5 and sequence512.
The8B base is frozen:3,833,856 adapter parameters are trainable. No paid resource
or data upload is used. All original weights and old adapters are preserved.

```powershell
.\.venv-soup\Scripts\python.exe -X utf8 scripts/train_rag_adapter.py preflight
.\.venv-soup\Scripts\python.exe -X utf8 scripts/train_rag_adapter.py train
.\.venv-soup\Scripts\python.exe -X utf8 scripts/train_rag_adapter.py verify
```

These are one-time artifact commands: existing output/evidence is refused.
They are documented for reproduction with new versioned output paths, not a
request to overwrite the completed run. The exact Soup formatter preflight
checks all108train/36holdout targets, no target truncation and masked input.

On2026-10-30 all108steps completed and the final adapter was saved. The first
wrapper then exited1 because it tried to evaluate through Soup's released
streaming runtime. Training was not repeated. The corrected wrapper never
forwards after `Soup.train()` closes that context. `verify` loads a fresh scratch
runtime, reloads both adapters exactly and leaves saved training artifacts alone.
The saved108-step trainer state,108rewind batch records and144finite tensors
were checked; final andcheckpoint tensors match exactly; all144changed from
the initial adapter. Recorded losses and gradient norms are finite.

The fixed six heldout silver targets (two per language) have mean teacher-forced
loss1.99912→1.49221 in the same fresh runtime. This is template imitation,
not human support quality. Full-answer evaluation determines deployment choice.
Private evidence: `training-preflight.json`, `saved-adapter-verification.json`,
`gguf-verification.json` under `.private/rag-training-v2`.
The candidate failed the full-answer promotion check and remains research-only.
Final results and HTTP verification: [RESULTS.md](RESULTS.md).

## Full-answer comparison

Freeze evaluation rows and inference code before comparing. Run sequentially,
one GPU process at a time. `oracle` isolates generation: authoritative source
IDs and language/category hints are supplied by the scenario and explicitly
recorded; it is not end-to-end performance. `pipeline` uses actual retrieval,
language detection, classification, generation and guards.

```powershell
.\.venv-soup\Scripts\python.exe -X utf8 scripts/evaluate_rag_quality.py --dry-run
.\.venv-soup\Scripts\python.exe -X utf8 scripts/evaluate_rag_quality.py --mode pipeline --output .private/old-pipeline-new.jsonl
.\.venv-soup\Scripts\python.exe -X utf8 scripts/evaluate_rag_quality.py --mode oracle --output .private/old-oracle-new.jsonl
.\.venv-soup\Scripts\python.exe -X utf8 scripts/evaluate_rag_quality.py --mode pipeline --adapter-path .private/soup-runs/qwen3-8b-rag-v2 --gguf-adapter-path .private/models/qwen3-8b-rag-v2-lora.gguf --output .private/new-pipeline-new.jsonl
```

Use the same candidate paths with `--mode oracle` for the isolated comparison.
Output must be a new private JSONL. It records dataset, model, adapter, corpus
and inference-code hashes, raw/effective output, failures and language summaries.
Exit0 means all observable checks pass; exit2 means completed with constraint
failures; exit1 includes generation/provider/setup errors, even if some rows pass.
Do not rewrite fixtures after seeing output and still call them unseen tests.
After the first comparison these36 scenarios informed fixes to language detection
and quoted-prose parsing. Subsequent runs are development/regression checks, not
an untouched final test. Training examples and candidate weights were not changed
in response to those fixes. A separate human-labeled test is still required.
Required citations, wrong event scopes, unknown IDs, complete EOS and pattern
guards are checked separately. Safe fallbacks cannot hide raw-generation errors.
Latency includes failed attempts; small-sample nearest-rank p95 is not an SLO.

Sources: [Soup](https://github.com/MakazhanAlpamys/Soup),
[PEFT public integration functions](https://huggingface.co/docs/peft/package_reference/functional),
[pinned official LoRA converter](https://github.com/ggml-org/llama.cpp/blob/8642541b255bfb5dc39cd351a178be09013a55d6/convert_lora_to_gguf.py).
