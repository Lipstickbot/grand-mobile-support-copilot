"""Train bounded local CPU classifier from weak player labels + private prototypes.

No historical outcome labels, GPU, downloads, unsafe model serialization or paid API.
The original validation player text is read solely to exclude overlap.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import heapq
import json
import marshal
import os
from pathlib import Path
import re
import sys
import time
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
from support_copilot.classifier import TextClassifier, metrics_for, regularized_head, sha256_file
from support_copilot.text import restore_kazakh
from prepare_soup_data import normalize, scrub

SEED = "classifier-v1-20261030"
LABEL_KIND = "weak_and_synthetic_not_human_gold"


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def player_text(row):
    # Construct from an explicit allowlist. No admin/reported-player/outcome reads.
    messages = row.get("messages", [])
    text = "\n".join(m["text"] for m in messages if isinstance(m, dict)
                     and m.get("role") == "player" and isinstance(m.get("text"), str))
    text, _ = restore_kazakh(text, row.get("server_lang"), row.get("region"))
    return scrub(text)


def exclusion_keys(row):
    keys = {digest(normalize(player_text(row)))}
    for message in row.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "player" and isinstance(message.get("text"), str):
            turn = {"messages": [message], "server_lang": row.get("server_lang"), "region": row.get("region")}
            keys.add(digest(normalize(player_text(turn))))
    return keys


def prepare_examples(archive, evaluation, prototypes, taxonomy, teacher, max_per_class):
    labels = [item["code"] for item in taxonomy["categories"]]
    excluded = set()
    source_counts = Counter()
    with ZipFile(archive) as zipped:
        with zipped.open("handout/tickets_valid.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                excluded.update(exclusion_keys(row))
                source_counts["original_validation_hashed_only"] += 1
        if evaluation.exists():
            for line in evaluation.read_text(encoding="utf8").splitlines():
                case = json.loads(line)
                excluded.update(exclusion_keys(case["request"]))
                source_counts["rag_development_hashed_only"] += 1
        seen, pools = set(), {label: [] for label in labels}
        with zipped.open("handout/tickets_train.jsonl.gz") as compressed, gzip.GzipFile(fileobj=compressed) as stream:
            for line in stream:
                row = json.loads(line)
                source_counts["original_train_scanned"] += 1
                kind = row.get("kind")
                if kind not in {"complaint", "report"}:
                    continue
                text = player_text(row)
                key = digest(normalize(text))
                if not 2 <= len(text) <= 1800 or not re.search(r"[^\W\d_]", text) or key in seen:
                    continue
                seen.add(key)
                if exclusion_keys(row) & excluded:
                    source_counts["overlap_excluded"] += 1
                    continue
                category, priority, _ = teacher(text, kind)
                if category not in labels or priority not in {"p0", "p1", "p2", "p3"}:
                    raise ValueError("Weak teacher does not match taxonomy")
                rank = int(digest(SEED + key), 16)
                item = {"text": text, "kind": kind, "category": category, "priority": priority,
                        "text_sha256": key, "source": "historical_player_weak_label", "human_gold": False}
                pool = pools[category]
                candidate = (-rank, key, item)
                if len(pool) < max_per_class:
                    heapq.heappush(pool, candidate)
                elif candidate > pool[0]:
                    heapq.heapreplace(pool, candidate)
    train, heldout, synthetic = [], [], []
    for pool in pools.values():
        for _, key, item in sorted(pool, reverse=True):
            (heldout if int(digest("holdout|" + key)[:8], 16) % 5 == 0 else train).append(item)
    if set(prototypes["classes"]) != set(labels):
        raise ValueError("Private prototypes must cover the exact taxonomy")
    all_keys = {item["text_sha256"] for item in train + heldout}
    for category, recipe in prototypes["classes"].items():
        kind = category.split(".", 1)[0]
        for lang in ("ru", "kk", "uz"):
            texts = recipe[lang]
            if len(texts) < 3:
                raise ValueError("Each class/language needs two train and one challenge text")
            for index, text in enumerate(texts):
                key = digest(normalize(text))
                if key in excluded or key in all_keys:
                    source_counts["prototypes_excluded_overlap"] += 1
                    continue
                all_keys.add(key)
                item = {"text": text, "kind": kind, "category": category, "priority": recipe["priority"],
                        "text_sha256": key, "source": "authored_multilingual_synthetic", "lang": lang,
                        "human_gold": False}
                (synthetic if index == len(texts) - 1 else train).append(item)
    assert not {item["text_sha256"] for item in train} & {item["text_sha256"] for item in heldout + synthetic}
    return train, heldout, synthetic, dict(source_counts)


def write_json(path, value):
    with path.open("x", encoding="utf8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def evaluate(rows, vectors, classifier, category_labels, priority_labels):
    if not rows:
        return {"n": 0, "label_kind": LABEL_KIND, "not_measured": "no examples"}
    predictions = []
    for row, vector in zip(rows, vectors):
        classifier.encode = lambda texts, value=vector: np.repeat(value[None, :], len(texts), axis=0)
        predictions.append(classifier.predict(row["text"], row["kind"]))
    return {"label_kind": LABEL_KIND, "human_gold": False,
            "category": metrics_for([r["category"] for r in rows], [p["category"] for p in predictions], category_labels),
            "priority": metrics_for([r["priority"] for r in rows], [p["priority"] for p in predictions], priority_labels),
            "category_uncertain": sum(p["uncertain"] for p in predictions),
            "by_language": {lang: {"n": sum(r.get("lang") == lang for r in rows),
                "category_correct": sum(r["category"] == p["category"] for r, p in zip(rows, predictions) if r.get("lang") == lang),
                "priority_correct": sum(r["priority"] == p["priority"] for r, p in zip(rows, predictions) if r.get("lang") == lang)}
                for lang in ("ru", "kk", "uz")}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".private/classifier-v1/model")
    parser.add_argument("--archive", type=Path, default=Path.home() / "Downloads/handout.zip")
    parser.add_argument("--prototypes", type=Path, default=ROOT / ".private/classifier-v1/prototypes.json")
    parser.add_argument("--max-per-class", type=int, default=64)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    evidence = output.parent
    if not output.is_relative_to((ROOT / ".private").resolve()) or output.exists() or not 16 <= args.max_per_class <= 80:
        parser.error("Use a new private model directory and16..80examples per weak class")
    evidence.mkdir(parents=True, exist_ok=True)
    for name in ("training-rows.jsonl", "heldout-rows.jsonl", "synthetic-rows.jsonl", "preparation.json", "metrics.json"):
        if (evidence / name).exists():
            parser.error("Evidence exists; preserve it and choose a new private parent/output")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", HF_HUB_DISABLE_IMPLICIT_TOKEN="1",
                      TOKENIZERS_PARALLELISM="false", DO_NOT_TRACK="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    from support_copilot import policy
    import torch
    from support_copilot.knowledge import HybridRetriever
    torch.set_num_threads(4)
    teacher_identity = {"policy_source_sha256_at_import": sha256_file(ROOT / "src/support_copilot/policy.py"),
                        "classify_code_sha256": hashlib.sha256(marshal.dumps(policy.classify.__code__)).hexdigest(),
                        "patterns_sha256": digest(json.dumps(policy.PATTERNS, ensure_ascii=False, sort_keys=True))}
    taxonomy_path = ROOT / ".private/data/taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf8"))
    prototypes = json.loads(args.prototypes.read_text(encoding="utf8"))
    started = time.monotonic()
    train, heldout, synthetic, source_counts = prepare_examples(args.archive,
        ROOT / ".private/rag-training-v2/prepared-reviewed/evaluation.jsonl", prototypes, taxonomy, policy.classify, args.max_per_class)
    category_labels = [row["code"] for row in taxonomy["categories"]]
    priority_labels = [row["code"] for row in taxonomy["priorities"]]
    for filename, rows in (("training-rows.jsonl", train), ("heldout-rows.jsonl", heldout), ("synthetic-rows.jsonl", synthetic)):
        with (evidence / filename).open("x", encoding="utf8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    preparation = {"label_kind": LABEL_KIND, "teacher": teacher_identity, "source_counts": source_counts,
                   "archive_sha256": sha256_file(args.archive), "taxonomy_sha256": sha256_file(taxonomy_path),
                   "private_prototypes_sha256": sha256_file(args.prototypes),
                   "counts": {"train": len(train), "weak_holdout": len(heldout), "synthetic_challenge": len(synthetic)},
                   "train_category_counts": dict(Counter(row["category"] for row in train)),
                   "holdout_category_counts": dict(Counter(row["category"] for row in heldout)),
                   "dedup": "normalized scrubbed player text hash before sampling and split; valid/dev joined and individual turns excluded",
                   "historical_administrator_targets": 0, "historical_outcome_features": 0, "official_valid_usage": "exclusion_hashes_only"}
    write_json(evidence / "preparation.json", preparation)
    print(json.dumps({"stage": "prepared", "counts": preparation["counts"], "teacher": teacher_identity}), flush=True)
    encoder = HybridRetriever(ROOT / ".private/data/rules.json", ROOT / ".private/models/multilingual-e5-small",
                              ROOT / ".private/index", dense_enabled=True)
    encoder.load()
    rows = train + heldout + synthetic
    vectors = encoder.encode_queries([row["text"] for row in rows])
    train_vectors = vectors[:len(train)]
    weights = np.zeros((vectors.shape[1] + 1, len(category_labels)), dtype=np.float32)
    for kind in ("complaint", "report"):
        indices = [i for i, row in enumerate(train) if row["kind"] == kind]
        labels = [label for label in category_labels if label.startswith(kind + ".")]
        head = regularized_head(train_vectors[indices], [train[i]["category"] for i in indices], labels)
        weights[:, [category_labels.index(label) for label in labels]] = head
    priority = regularized_head(train_vectors, [row["priority"] for row in train], priority_labels)
    output.mkdir()
    np.savez_compressed(output / "weights.npz", category_weights=weights, priority_weights=priority)
    manifest = {"format_version": 1, "label_kind": LABEL_KIND, "human_gold": False,
                "category_labels": category_labels, "priority_labels": priority_labels, "dimension": int(vectors.shape[1]),
                "encoder_sha256": encoder._model_identity(), "taxonomy_sha256": sha256_file(taxonomy_path),
                "weights_sha256": sha256_file(output / "weights.npz"),
                "training_rows_sha256": sha256_file(evidence / "training-rows.jsonl"),
                "preparation_sha256": sha256_file(evidence / "preparation.json"),
                "runtime_source_sha256": sha256_file(ROOT / "src/support_copilot/classifier.py"),
                "teacher": teacher_identity, "method": "separate-kind class-balanced ridge; independent priority ridge",
                "regularization": .05, "softmax_scale": 3., "scores_calibrated": False,
                "seed": SEED, "counts": preparation["counts"], "no_gpu": True, "external_api_cost_usd": 0,
                "limitations": ["weak_labels_not_human_gold", "synthetic_translations_need_native_review",
                                "encoder_input_truncated_at512tokens", "no_official_valid_accuracy",
                                "kind_supplied_by_caller", "root_must_apply_mandatory_safety_overrides"]}
    write_json(output / "manifest.json", manifest)
    classifier = TextClassifier(output, encoder.encode_queries)
    classifier.load()
    heldout_vectors = vectors[len(train):len(train) + len(heldout)]
    metrics = {"schema_version": 1, "label_kind": LABEL_KIND, "human_gold": False,
               "weak_label_holdout_agreement": evaluate(heldout, heldout_vectors, classifier, category_labels, priority_labels),
               "authored_synthetic_challenge": evaluate(synthetic, vectors[len(train) + len(heldout):], classifier, category_labels, priority_labels),
               "all_cases_require_human": True, "confidence_calibrated": False,
               "training_seconds": time.monotonic() - started, "model_manifest_sha256": sha256_file(output / "manifest.json"),
               "not_measured": ["human_category_accuracy", "human_priority_macroF1", "calibration", "production_quality"]}
    write_json(evidence / "metrics.json", metrics)
    print(json.dumps({"stage": "complete", "output": str(output), "train": len(train),
        "weak_holdout_n": len(heldout), "synthetic_n": len(synthetic),
        "weak_category_agreement": metrics["weak_label_holdout_agreement"]["category"]["accuracy"],
        "synthetic_category_agreement": metrics["authored_synthetic_challenge"]["category"]["accuracy"],
        "training_seconds": metrics["training_seconds"], "human_gold": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
