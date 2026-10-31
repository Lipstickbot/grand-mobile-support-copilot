"""Invented deterministic classifier fixtures; no official data, encoder or GPU."""
import gzip
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from support_copilot.classifier import TextClassifier, metrics_for, regularized_head, sha256_file

SPEC = importlib.util.spec_from_file_location("classifier_training_tested", Path(__file__).parents[1] / "scripts/train_classifier.py")
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
LABELS = ["complaint.alpha", "complaint.beta", "report.gamma", "report.delta"]


def save_model(path, category=None, priority=None, **manifest_changes):
    path.mkdir(exist_ok=True)
    category = np.array([[3, -3, -3, 3], [-3, 3, 3, -3], [0, 0, 0, 0]], dtype=np.float32) if category is None else category
    priority = np.array([[2, -1, 0, 0], [-1, 2, 0, 0], [0, 0, 0, 0]], dtype=np.float32) if priority is None else priority
    np.savez_compressed(path / "weights.npz", category_weights=category, priority_weights=priority)
    manifest = {"format_version": 1, "label_kind": "weak_and_synthetic_not_human_gold",
                "dimension": 2, "category_labels": LABELS, "priority_labels": ["p0", "p1", "p2", "p3"],
                "softmax_scale": 3.0, "scores_calibrated": False,
                "encoder_sha256": "encoder-fixture", "taxonomy_sha256": "taxonomy-fixture",
                "weights_sha256": sha256_file(path / "weights.npz"), **manifest_changes}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return TextClassifier(path, lambda texts: np.tile([[1., 0.]], (len(texts), 1)))


def test_separate_kind_masks_cannot_predict_another_channel(tmp_path):
    classifier = save_model(tmp_path)
    classifier.load()
    assert classifier.predict("Synthetic input", "complaint")["category"] == "complaint.alpha"
    actual = classifier.predict("Synthetic input", "report")
    assert actual["category"] == "report.delta"
    assert set(actual["category_scores"]) == set(LABELS)
    assert actual["category_scores"]["complaint.alpha"] == 0
    assert not classifier.stats()["calibrated"]
    assert classifier.stats()["label_kind"] == "weak_and_synthetic_not_human_gold"


def test_ambiguous_scores_are_explicit_not_automatic_confidence(tmp_path):
    classifier = save_model(tmp_path, category=np.zeros((3, 4), dtype=np.float32))
    classifier.load()
    result = classifier.predict("Invented text", "complaint")
    assert result["uncertain"] and result["topic_hints"] == []
    assert result["confidence"] == .5


def test_classifier_reuses_encoder_and_only_receives_supplied_text(tmp_path):
    classifier = save_model(tmp_path)
    captured = []
    classifier.encode = lambda texts: captured.extend(texts) or np.tile([[1., 0.]], (len(texts), 1))
    classifier.load()
    classifier.predict("Supplied player content", "report")
    assert captured == ["Supplied player content"]


@pytest.mark.parametrize("features", [np.array([[np.nan, 1.]]), np.zeros((1, 2)), np.ones((1, 3))])
def test_invalid_encoder_features_fail_closed(tmp_path, features):
    classifier = save_model(tmp_path)
    classifier.encode = lambda texts: features
    classifier.load()
    with pytest.raises(ValueError):
        classifier.predict("Synthetic input", "report")


@pytest.mark.parametrize("text,kind", [("", "report"), ("text", "admin"), ("x" * 16001, "report")])
def test_input_limits(tmp_path, text, kind):
    classifier = save_model(tmp_path)
    classifier.load()
    with pytest.raises(ValueError):
        classifier.predict(text, kind)


def test_missing_load_and_identity_drift_are_errors(tmp_path):
    classifier = save_model(tmp_path)
    with pytest.raises(RuntimeError):
        classifier.predict("Text", "report")
    classifier.expected_encoder_sha256 = "different-encoder"
    with pytest.raises(ValueError, match="identity"):
        classifier.load()
    assert not classifier.ready


def test_weight_hash_prevents_silent_replacement(tmp_path):
    classifier = save_model(tmp_path)
    with (tmp_path / "weights.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        classifier.load()


@pytest.mark.parametrize("change", [{"softmax_scale": 4.0}, {"scores_calibrated": True}])
def test_score_semantics_cannot_silently_drift_in_manifest(tmp_path, change):
    classifier = save_model(tmp_path, **change)
    with pytest.raises(ValueError, match="score semantics"):
        classifier.load()


@pytest.mark.parametrize("category", [np.zeros((2, 4), dtype=np.float32),
                                      np.full((3, 4), np.nan, dtype=np.float32),
                                      np.zeros((3, 4), dtype=object)])
def test_invalid_shape_nonfinite_or_pickle_payload_rejected(tmp_path, category):
    classifier = save_model(tmp_path, category=category)
    with pytest.raises(ValueError):
        classifier.load()


def test_learned_head_generalizes_intermediate_vectors_without_keyword_rules():
    x = np.array([[1., 0.], [.9, .1], [0., 1.], [.1, .9]])
    labels = ["left", "right"]
    learned = regularized_head(x, ["left", "left", "right", "right"], labels)
    assert int(np.argmax(np.array([.8, .2, 1.]) @ learned)) == 0
    assert int(np.argmax(np.array([.2, .8, 1.]) @ learned)) == 1
    with pytest.raises(ValueError):
        regularized_head(x, ["left"] * 4, labels)


def test_metrics_keep_unobserved_classes_visible_and_do_not_invent_support():
    actual = metrics_for(["left", "left", "right"], ["left", "right", "right"], ["left", "right", "absent"])
    assert actual["n"] == 3 and actual["accuracy"] == 2 / 3
    assert actual["classes_with_support"] == 2 and actual["total_classes"] == 3
    assert actual["per_class"]["absent"]["support"] == 0


def test_training_allowlist_excludes_administrator_and_outcome_data():
    row = {"messages": [{"role": "player", "text": "Player description"},
                        {"role": "admin", "text": "PRIVATE_ADMIN"},
                        {"role": "reported_player", "text": "PRIVATE_REPORTED"}],
           "verdict": "PRIVATE_VERDICT", "punishment": "PRIVATE_PUNISHMENT", "status": "PRIVATE_STATUS"}
    assert TRAIN.player_text(row) == "Player description"
    assert len(TRAIN.exclusion_keys(row)) == 1


def test_preparation_excludes_valid_turns_duplicates_and_keeps_splits_disjoint(tmp_path):
    def ticket(text, kind="report"):
        return {"kind": kind, "messages": [{"role": "player", "text": text}, {"role": "admin", "text": "PRIVATE"}]}
    original = [ticket(f"Invented historical example number {i}") for i in range(100)]
    original += [ticket("Player excluded validation turn"), ticket("Invented historical example number 1")]
    valid = ticket("Player excluded validation turn")
    archive = tmp_path / "fixture.zip"
    with ZipFile(archive, "w") as zipped:
        zipped.writestr("handout/tickets_valid.jsonl", json.dumps(valid))
        zipped.writestr("handout/tickets_train.jsonl.gz", gzip.compress("\n".join(map(json.dumps, original)).encode()))
    labels = ["report.fixture", "complaint.fixture"]
    taxonomy = {"categories": [{"code": label} for label in labels]}
    prototypes = {"classes": {label: {"priority": "p3", **{lang: [f"Synthetic {label} {lang} variant{j}" for j in range(3)]
                      for lang in ("ru", "kk", "uz")}} for label in labels}}
    train, heldout, synthetic, counts = TRAIN.prepare_examples(archive, tmp_path / "absent.jsonl", prototypes, taxonomy,
        lambda text, kind: (kind + ".fixture", "p3", []), 40)
    assert counts["original_validation_hashed_only"] == 1
    assert counts["original_train_scanned"] == len(original)
    assert counts["overlap_excluded"] >= 1
    assert not {r["text_sha256"] for r in train} & {r["text_sha256"] for r in heldout + synthetic}
    assert all("PRIVATE" not in r["text"] for r in train + heldout + synthetic)
    assert all(r["human_gold"] is False for r in train + heldout + synthetic)
