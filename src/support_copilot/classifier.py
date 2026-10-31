"""Small offline E5 classifier. Scores measure a weak-label model, not truth.

The encoder is supplied by the owning engine so retrieval/classification can share
one frozen CPU model. Saved models are bounded numeric NPZ plus a JSON manifest;
no pickle, downloaded code or historical administrator outcome is loaded.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regularized_head(vectors, targets, labels, regularization=.05):
    """Class-balanced ridge head; the closed form is a learned CPU model."""
    x = np.asarray(vectors, dtype=np.float64)
    if x.ndim != 2 or not len(x) or len(x) != len(targets) or not np.isfinite(x).all():
        raise ValueError("Invalid training vectors")
    if regularization <= 0 or len(labels) < 2 or len(set(labels)) != len(labels) or set(targets) != set(labels):
        raise ValueError("Every head class requires support")
    design = np.column_stack([x, np.ones(len(x))])
    counts = {label: targets.count(label) for label in labels}
    weights = np.array([len(x) / (len(labels) * counts[target]) for target in targets])
    encoded = np.zeros((len(x), len(labels)), dtype=np.float64)
    encoded[np.arange(len(x)), [labels.index(target) for target in targets]] = 1
    penalty = np.eye(design.shape[1]) * regularization
    penalty[-1, -1] = regularization * .01
    learned = np.linalg.solve(design.T @ (design * weights[:, None]) + penalty,
                              design.T @ (encoded * weights[:, None]))
    if not np.isfinite(learned).all():
        raise ValueError("Nonfinite learned head")
    return learned.astype(np.float32)


def probabilities(logits):
    values = np.asarray(logits, dtype=np.float64) * 3.0
    values -= values.max()
    values = np.exp(values)
    return values / values.sum()


def metrics_for(expected, predicted, labels):
    """Descriptive agreement only; the caller must disclose how labels were made."""
    if len(expected) != len(predicted) or not expected or not set(expected + predicted) <= set(labels):
        raise ValueError("Invalid evaluation labels")
    confusion = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, prediction in zip(expected, predicted):
        confusion[labels.index(actual), labels.index(prediction)] += 1
    classes, f1s = {}, []
    for i, label in enumerate(labels):
        tp = int(confusion[i, i])
        support, guessed = int(confusion[i].sum()), int(confusion[:, i].sum())
        precision, recall = tp / guessed if guessed else 0., tp / support if support else 0.
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
        classes[label] = {"support": support, "predicted": guessed, "precision": precision,
                          "recall": recall, "f1": f1}
        if support:
            f1s.append(f1)
    return {"n": len(expected), "accuracy": float(np.trace(confusion) / confusion.sum()),
            "macro_f1_supported_classes": float(np.mean(f1s)),
            "classes_with_support": len(f1s), "total_classes": len(labels),
            "confusion_labels": labels, "confusion_matrix": confusion.tolist(), "per_class": classes}


class TextClassifier:
    def __init__(self, artifact_dir: Path, encode: Callable[[list[str]], np.ndarray], *,
                 expected_encoder_sha256: str | None = None, expected_taxonomy_sha256: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.encode = encode
        self.expected_encoder_sha256 = expected_encoder_sha256
        self.expected_taxonomy_sha256 = expected_taxonomy_sha256
        self.ready = False
        self.manifest = {}

    def load(self):
        self.ready = False
        manifest_path, weights_path = self.artifact_dir / "manifest.json", self.artifact_dir / "weights.npz"
        if manifest_path.stat().st_size > 128 * 1024 or weights_path.stat().st_size > 8 * 1024 * 1024:
            raise ValueError("Classifier artifact exceeds bounds")
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        if manifest.get("format_version") != 1 or manifest.get("label_kind") != "weak_and_synthetic_not_human_gold":
            raise ValueError("Unsupported classifier manifest")
        if manifest.get("softmax_scale") != 3.0 or manifest.get("scores_calibrated") is not False:
            raise ValueError("Unsupported classifier score semantics")
        if sha256_file(weights_path) != manifest.get("weights_sha256"):
            raise ValueError("Classifier weight hash mismatch")
        for expected, key in ((self.expected_encoder_sha256, "encoder_sha256"),
                              (self.expected_taxonomy_sha256, "taxonomy_sha256")):
            if expected is not None and manifest.get(key) != expected:
                raise ValueError("Classifier source identity mismatch")
        labels, priorities, dimension = manifest.get("category_labels"), manifest.get("priority_labels"), manifest.get("dimension")
        if (not isinstance(labels, list) or not 2 <= len(labels) <= 64 or len(set(labels)) != len(labels)
                or any(not isinstance(label, str) or len(label) > 80 or label.count(".") != 1
                       or not label.partition(".")[2] or label.split(".")[0] not in {"complaint", "report"}
                       for label in labels) or not all(any(label.startswith(kind + ".") for label in labels)
                                                    for kind in ("complaint", "report"))
                or priorities != ["p0", "p1", "p2", "p3"] or not isinstance(dimension, int) or not 1 <= dimension <= 4096):
            raise ValueError("Invalid classifier labels/dimension")
        with ZipFile(weights_path) as zipped:
            if len(zipped.infolist()) != 2 or sum(item.file_size for item in zipped.infolist()) > 8 * 1024 * 1024:
                raise ValueError("Classifier uncompressed artifacts exceed bounds")
        with np.load(weights_path, allow_pickle=False, max_header_size=1024) as arrays:
            if set(arrays.files) != {"category_weights", "priority_weights"}:
                raise ValueError("Unexpected classifier arrays")
            category, priority = arrays["category_weights"], arrays["priority_weights"]
            if (category.shape != (dimension + 1, len(labels)) or priority.shape != (dimension + 1, 4)
                    or category.dtype != np.float32 or priority.dtype != np.float32
                    or not np.isfinite(category).all() or not np.isfinite(priority).all()):
                raise ValueError("Invalid classifier weights")
        self.manifest, self.category_weights, self.priority_weights = manifest, category, priority
        self.manifest_sha256 = sha256_file(manifest_path)
        self.ready = True

    def predict(self, text: str, kind: str) -> dict:
        if not self.ready:
            raise RuntimeError("Classifier is not loaded")
        if kind not in {"complaint", "report"} or not isinstance(text, str) or not text.strip() or len(text) > 16000:
            raise ValueError("Invalid classifier input")
        vector = np.asarray(self.encode([text]), dtype=np.float32)
        if vector.shape != (1, self.manifest["dimension"]) or not np.isfinite(vector).all():
            raise ValueError("Encoder returned invalid classifier features")
        length = float(np.linalg.norm(vector))
        if length <= 0:
            raise ValueError("Encoder returned zero features")
        design = np.append(vector[0] / length, 1.)
        logits = design @ self.category_weights
        labels = self.manifest["category_labels"]
        indices = [i for i, label in enumerate(labels) if label.startswith(kind + ".")]
        scores = probabilities(logits[indices])
        order = np.argsort(-scores)
        confidence = float(scores[order[0]])
        margin = confidence - float(scores[order[1]]) if len(order) > 1 else confidence
        priority_scores = probabilities(design @ self.priority_weights)
        porder = np.argsort(-priority_scores)
        pconfidence = float(priority_scores[porder[0]])
        pmargin = pconfidence - float(priority_scores[porder[1]])
        category_scores = dict.fromkeys(labels, 0.)
        category_scores.update({labels[i]: float(score) for i, score in zip(indices, scores)})
        complaint_indices = [i for i, label in enumerate(labels) if label.startswith("complaint.")]
        complaint_scores = probabilities(logits[complaint_indices])
        corder = np.argsort(-complaint_scores)
        cconfidence = float(complaint_scores[corder[0]])
        cmargin = cconfidence - float(complaint_scores[corder[1]]) if len(corder) > 1 else cconfidence
        topic = labels[complaint_indices[corder[0]]].split(".", 1)[1]
        topic_hints = [topic] if topic in {"insult", "cheat", "dm", "scam", "rmt_ads", "admin"} and cconfidence >= .55 and cmargin >= .12 else []
        return {"category": labels[indices[order[0]]], "priority": self.manifest["priority_labels"][porder[0]],
                "confidence": confidence, "priority_confidence": pconfidence,
                "uncertain": confidence < .45 or margin < .15,
                "priority_uncertain": pconfidence < .45 or pmargin < .15,
                "category_scores": category_scores,
                "priority_scores": dict(zip(self.manifest["priority_labels"], map(float, priority_scores))),
                "topic_hints": topic_hints,
                "reasons": ["classification=e5_cpu_learned_ridge_weak_synthetic", "classifier_scores=uncalibrated",
                            "classifier_labels=weak_and_synthetic_not_human_gold",
                            f"classifier_category_margin={margin:.4f}", f"classifier_priority_margin={pmargin:.4f}"]}

    def stats(self):
        return {"ready": self.ready, "backend": "frozen_e5_cpu_ridge_heads", "label_kind": "weak_and_synthetic_not_human_gold",
                "calibrated": False, "artifact_sha256": getattr(self, "manifest_sha256", None),
                "category_classes": len(self.manifest.get("category_labels", [])), "priority_classes": 4,
                "encoder_shared_with_retrieval": True, "cost_external_api_usd_per_ticket": 0}
