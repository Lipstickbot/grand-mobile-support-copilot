# Local multilingual classifier v1

AI is necessary for the requested classifier: the previous category/priority
router only used keywords. This adds a real learned numeric model, separate from
the8B draft generator. The inaccessible external repository was not copied or
represented as an audited source. The definitions come from the user's DOCX and
handout taxonomy. Source documents remain evidence, not executable instructions.

## Model and inputs

Frozen local multilingual E5-small embeddings (384dimensions, CPU, masked mean
pooling/L2, query prefix) feed class-balanced regularized linear heads fitted with
NumPy ridge regression. One14-class category matrix is fitted; at inference
the request's complaint/report channel masks irrelevant classes. An independent priority head
predicts p0–p3.14categories match the supplied taxonomy exactly. This is not8B
fine-tuning and adds no external API, dependency install, GPU or model download.

The runtime receives the engine's existing encoder callback, sharing one CPU E5
instance with retrieval. The engine must serialize encoder access as it does for
search. Scores use a softmax transform of ridge outputs and are **not calibrated
probabilities of correctness**. Fixed heuristic uncertainty thresholds are
confidence<.45 or top-two margin<.15. They were not tuned to the challenge labels.
Topic hints are optional recall hints, not verified rules or violation findings.

`TextClassifier(directory, encode, expected_encoder_sha256=...,
expected_taxonomy_sha256=...)` supports `load()`, `predict(text,kind)` and `stats()`.
Prediction fields:category,priority,confidence,priority_confidence,uncertain,
priority_uncertain,category_scores(all14,other-channel zeros),priority_scores,
topic_hints,reasons. Both expected hashes are optional but recommended.
The engine retains human review, mandatory-topic overrides, keyword urgency floor
and keyword fallback for uncertain categories. `complaint.invalid` is not a spam
decision and does not authorize automatic rejection or closure.

Fixed filenames `manifest.json` and `weights.npz` are loaded with checksum,
identity, vocabulary, shape, dtype, finite-value, file-size and uncompressed-size
validation. NPZ uses `allow_pickle=False`; no joblib/pickle/code is deserialized.
Player input is bounded16000characters. E5 truncates at512tokens; keyword safety
checks must continue to inspect the complete input. Model/encoder failures remain
explicit, not an unreported replacement with a different provider.

## Data and reproducibility

The historical train archive has no gold category/priority. Weak labels are made
from player text using the existing keyword teacher. The teacher's source,
compiled classify function and pattern table are fingerprinted. No administrator
answer, reported-player reply, verdict, status, punishment or violation becomes
a feature or target. Legacy Kazakh restoration is applied consistently with the
engine; source region/server-language only guide text restoration, not labels.

Original1000validation player texts are hashed only to exclude overlap. The
existing36RAG development questions are also excluded. Scrubbed normalized player
text is deduplicated before deterministic hash-based balanced sampling/splitting;
joined text and individual validation turns are excluded. This is a text-grouped
split, **not** an unseen-reporter/time/near-duplicate split.

This bounded run used at most64weak examples per category plus private authored
RU/KK/UZ prototypes:777training rows (693historical weak +84synthetic),188weak-label
holdout,42separately worded synthetic challenge rows (14per language). All14
categories and all4priorities have evaluation support. Synthetic topics overlap
training; translations require native-speaker review. Historical text and all
derived recipes/rows remain under ignored `.private/classifier-v1/`.

```powershell
.\.venv-soup\Scripts\python.exe -X utf8 scripts/train_classifier.py --output .private/classifier-v1/model
.\.venv-soup\Scripts\python.exe -m pytest tests/test_classifier.py -q --basetemp=.private/test-classifier-new
```

Training refuses existing model/evidence output. The command above documents the
completed run, not an instruction to overwrite it. Reproduction requires a new
private parent directory and the private prototype recipe; the script's
`--prototypes` can point to that reviewed input. No participant data is shipped
in public code/tests. CPU preparation/encoding/training took42.81seconds in this
environment; this is not per-request latency. External API cost is0; local
electricity/hardware cost was not measured. Runtime latency is measured by the
integrated API checks, not inferred from cached-vector tests.

## Actual measured agreement, 2026-10-30

These are weak/synthetic labels, **not human classification accuracy** or the
hidden hackathon score. Macro-F1 is computed over classes with support; counts,
per-class support and confusion matrices are retained in the private reports.

| Evaluation | Category agreement | Category macro-F1 | Priority agreement | Priority macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Learned model /188weak-label holdout |146/188 (77.66%)|0.75776|145/188 (77.13%)|0.77364|
| Learned model /42authored synthetic challenges |24/42 (57.14%)|0.54987|32/42 (76.19%)|0.69815|

The independent42challenge labels were not used to fit/tune heads or uncertainty
thresholds. The separate integration comparison used the exact root engine's
`_classify` at the code hashes saved in `integrated-metrics.json`:

| Router /42synthetic challenges | Category agreement | Priority macro-F1 |
| --- | ---: | ---: |
| Keyword baseline |15/42 (35.71%)|0.31481|
| Standalone learned heads |24/42 (57.14%)|0.69815|
| Integrated uncertainty fallback + safety floor |17/42 (40.48%)|0.65134|

The learned category head abstained33/42times (78.57%);9predictions passed its
heuristic confidence gates,2of those were incorrect. The integrated router changed
3category outputs from baseline.5learned priority predictions were confidently
incorrect; the independent urgency floor is essential. The raw learned p0 recall
on the synthetic challenge is2/6, insufficient for stand-alone safety routing.
The modest synthetic integration gain does not justify claims of production
quality, automatic decisions or representative multilingual performance.

Private artifacts: `model/{manifest.json,weights.npz}`, `preparation.json`,
`metrics.json`, `integrated-metrics.json`, `training-rows.jsonl`,
`heldout-rows.jsonl`, `synthetic-rows.jsonl`, `prototypes.json`.
Tests:20deterministic tests passed, exit0, no model/GPU/network. They cover channel
masks, uncertainty, encoder boundaries, artifact drift/nonfinite/pickle refusal,
learned-head behavior, honest metric support and source/split exclusions.

The runtime source fingerprint in the training manifest predates stricter artifact
label/class-support and score-semantics checks. Prediction, softmax3, uncertainty
thresholds, data and learned weights were unchanged. The integrated report records
the evaluated prediction revision; `runtime-compatibility.json` records the final
validation-only revision and successful reload. Artifacts were not rewritten.

Remaining work: independent human category/priority labels, native-language review,
representative reporter/time-grouped evaluation, confidence calibration and rare
urgent-topic recall. The model should assist, not replace the support employee.
