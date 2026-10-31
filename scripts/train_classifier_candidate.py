"""Fit a frozen-E5 head candidate on private authored intent labels.

No administrator outcomes or freeform answer targets. Uncertainty thresholds,
encoder, dimensionality and priority safety rules remain unchanged.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.classifier import TextClassifier, regularized_head, metrics_for, sha256_file
from support_copilot.config import Settings
from support_copilot.engine import TriageEngine
from support_copilot.knowledge import HybridRetriever
from support_copilot.schemas import TriageRequest
from support_copilot.text import analyze_language, normalize_retrieval_query


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]


def view(row):
    # Expected/gold language is an evaluation target, never a feature. Only
    # metadata actually supplied to the service may scope legacy restoration.
    return normalize_retrieval_query(analyze_language(row["text"], row.get("server_lang"), row.get("region"))["normalized_text"])[0]


def validate_rows(rows, labels):
    if not rows:
        raise ValueError("Empty training corpus")
    seen = set()
    for row in rows:
        if (set(row) != {"text", "kind", "category", "priority", "lang", "source"}
                or row["kind"] not in {"report", "complaint"} or row["category"] not in labels
                or not row["category"].startswith(row["kind"]+".")
                or row["priority"] not in {"p0", "p1", "p2", "p3"}
                or row["lang"] not in {"ru", "kk", "uz", "en"}
                or row["source"] not in {"previous_authored_training_not_weak_teacher", "authored_intent_or_general_rule_question", "new_authored_english_intent"}
                or not isinstance(row["text"], str) or not row["text"].strip() or len(row["text"]) > 16000):
            raise ValueError("Only explicitly authored intent examples are accepted")
        key = " ".join(view(row).casefold().split())
        if key in seen:
            raise ValueError("Duplicate normalized training text")
        seen.add(key)
    return seen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    directory = args.output.resolve()
    if not directory.is_relative_to((ROOT/".private").resolve()) or directory.exists():
        raise ValueError("Use a new private output directory")
    taxonomy_path = ROOT/".private/data/taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_text(encoding="utf8"))
    labels = [r["code"] for r in taxonomy["categories"]]
    priorities = ["p0", "p1", "p2", "p3"]
    rows = read(args.training)
    training_keys = validate_rows(rows, labels)
    challenge = read(ROOT/".private/classifier-v1/synthetic-rows.jsonl")
    weak = read(ROOT/".private/classifier-v1/heldout-rows.jsonl")
    fresh = read(ROOT/".private/quality-v5/critical16-fresh.jsonl")
    fresh_rows = [{"text": "\n".join(m["text"] for m in r["request"]["messages"] if m["role"] == "player"),
        "kind": r["request"]["kind"], "category": r["expected_category"], "priority": "p0",
        "lang": r["expected_language"]} for r in fresh]
    for row in challenge + weak + fresh_rows:
        if " ".join(view(row).casefold().split()) in training_keys:
            raise ValueError("Evaluation/training text overlap")
    encoder = HybridRetriever(ROOT/".private/data/rules.json", ROOT/".private/models/multilingual-e5-small", ROOT/".private/index", dense_enabled=True)
    encoder.load()
    vectors = encoder.encode_queries([view(r) for r in rows])
    category_weights = np.zeros((vectors.shape[1]+1, len(labels)), dtype=np.float32)
    for kind in ("report", "complaint"):
        indices = [i for i,r in enumerate(rows) if r["kind"] == kind]
        sublabels = [label for label in labels if label.startswith(kind+".")]
        head = regularized_head(vectors[indices], [rows[i]["category"] for i in indices], sublabels)
        category_weights[:, [labels.index(label) for label in sublabels]] = head
    priority_weights = regularized_head(vectors, [r["priority"] for r in rows], priorities)
    model_dir = directory/"model"
    model_dir.mkdir(parents=True)
    np.savez_compressed(model_dir/"weights.npz", category_weights=category_weights, priority_weights=priority_weights)
    manifest = {"format_version": 1, "label_kind": "weak_and_synthetic_not_human_gold", "human_gold": False,
        "category_labels": labels, "priority_labels": priorities, "dimension": int(vectors.shape[1]),
        "encoder_sha256": encoder._model_identity(), "taxonomy_sha256": sha256_file(taxonomy_path),
        "weights_sha256": sha256_file(model_dir/"weights.npz"), "training_rows_sha256": sha256_file(args.training),
        "method": "class_balanced_ridge_authored_intents_only_no_historical_keyword_targets", "regularization": .05,
        "softmax_scale": 3., "scores_calibrated": False, "counts": {"train": len(rows)},
        "historical_outcome_targets": 0, "weak_keyword_training_targets": 0, "encoder_fine_tuned": False,
        "runtime_source_sha256": sha256_file(ROOT/"src/support_copilot/classifier.py")}
    (model_dir/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    models = {"current": TextClassifier(ROOT/".private/classifier-v1/model", encoder.encode_queries),
              "candidate": TextClassifier(model_dir, encoder.encode_queries)}
    for classifier in models.values():
        classifier.load()
    def head_metrics(classifier, values):
        predicted = [classifier.predict(view(r), r["kind"]) for r in values]
        return {"category": metrics_for([r["category"] for r in values], [r["category"] for r in predicted], labels),
                "priority": metrics_for([r["priority"] for r in values], [r["priority"] for r in predicted], priorities)}
    def integrated(classifier, values):
        engine = TriageEngine(replace(Settings(), generation_enabled=False))
        engine.ready = True
        engine.retriever = encoder
        engine.classifier = classifier
        predicted = [engine.classify_request(TriageRequest.model_validate({"kind": r["kind"],
            "messages": [{"role": "player", "text": r["text"]}]})) for r in values]
        return {"category": metrics_for([r["category"] for r in values], [r["category"] for r in predicted], labels),
                "priority": metrics_for([r["priority"] for r in values], [r["priority"] for r in predicted], priorities),
                "critical_correct": sum(p["category"] == r["category"] and p["priority"] == "p0" for p,r in zip(predicted, values) if r["priority"] == "p0"),
                "critical_cases": sum(r["priority"] == "p0" for r in values),
                "predictions": [{"expected": r["category"], "actual": p["category"], "priority": p["priority"], "uncertain": p["uncertain"]} for r,p in zip(values, predicted)]}
    result = {"human_gold": False, "training_rows": len(rows), "encoder_unchanged": True,
        "thresholds_unchanged": True, "models": {name: {"challenge42": integrated(model, challenge),
            "fresh16_known_stress": integrated(model, fresh_rows)} for name,model in models.items()},
        "limitations": "AI intent labels; known stress cases, no human representative accuracy or latency comparison"}
    (directory/"comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    report = {"schema_version": 1, "label_kind": "weak_and_synthetic_not_human_gold", "human_gold": False,
        "weak_label_holdout_agreement": head_metrics(models["candidate"], weak),
        "authored_synthetic_challenge": head_metrics(models["candidate"], challenge)}
    (directory/"metrics.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf8")
    print(json.dumps({name: {dataset: {"category_accuracy": m["category"]["accuracy"],
        "priority_macro_f1": m["priority"]["macro_f1_supported_classes"], "critical_correct": m["critical_correct"],
        "critical_cases": m["critical_cases"]} for dataset,m in scores.items()} for name,scores in result["models"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
