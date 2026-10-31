"""Training + loading of the v3 classical models (CPU, ~3 min to train, ~1 ms to predict).

* category heads: TF-IDF (char 2-5 + word 1-2) -> LinearSVC, one per ticket kind,
  trained on weak labels distilled from admin outcomes (see ``labels.py``);
* rule head: same features -> LinearSVC over rule points that admins actually cited
  (learned retrieval, "which rule does this text usually end up under");
* answer memory: nearest historical tickets + cleaned admin replies, used only as
  style exemplars for the optional LLM.

Admin replies/outcomes are used ONLY to build training targets and exemplars;
at inference the models see the player's text, rule_refs_raw and kind.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .labels import weak_category
from .rulekb import admin_rule_ids, load_rule_ids

FORMAT = 3
NICK = re.compile(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b")


def feature_text(kind: str, player_text: str, rule_refs_raw: str | None) -> str:
    return f"[{kind}] [refs: {rule_refs_raw or ''}] {player_text}"


def player_text(ticket: dict) -> str:
    return " \n ".join(m.get("text", "") for m in ticket.get("messages", []) if m.get("role") == "player")


def clean_admin(text: str) -> str:
    text = NICK.sub("{nick}", text)
    text = re.sub(r"#?<digits>", "#{id}", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_jsonl(path: Path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def train(data_dir: Path, out_dir: Path, *, svm_c: float = 0.5, memory_size: int = 12000, seed: int = 13, log=print,
          include_ids: set[str] | None = None):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    from sklearn.svm import LinearSVC

    data_dir, out_dir = Path(data_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tickets = _read_jsonl(data_dir / "tickets_train.jsonl.gz")
    if include_ids is not None:  # group split: train only on the given tickets (test/dev groups excluded)
        tickets = [t for t in tickets if t["ticket_id"] in include_ids]
    known = load_rule_ids(data_dir / "rules.json")
    texts = [feature_text(t["kind"], player_text(t), t.get("rule_refs_raw")) for t in tickets]
    labels = [weak_category(t)[0] for t in tickets]
    log(f"tickets={len(tickets)} label_dist={dict(Counter(labels))}")

    vec = make_union(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=200000, sublinear_tf=True, dtype=np.float32),
        TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=120000, sublinear_tf=True,
                        token_pattern=r"(?u)\b[\w'.]+\b", dtype=np.float32))
    X = vec.fit_transform(texts)
    heads = {}
    for kind in ("complaint", "report"):
        idx = [i for i, t in enumerate(tickets) if t["kind"] == kind]
        # complaints: unweighted (+1-2 pt on valid); reports: balanced to keep recall of rare p0 classes
        clf = LinearSVC(C=svm_c, class_weight=None if kind == "complaint" else "balanced").fit(X[idx], [labels[i] for i in idx])
        clf.coef_ = clf.coef_.astype(np.float32)
        heads[kind] = clf
        log(f"category head {kind}: classes={list(clf.classes_)}")

    rule_rows, rule_targets = [], []
    for i, t in enumerate(tickets):
        if t["kind"] == "complaint":
            ids = admin_rule_ids(t, known)
            if ids:
                rule_rows.append(i); rule_targets.append(ids[0])
    counts = Counter(rule_targets)
    keep = [j for j, g in enumerate(rule_targets) if counts[g] >= 15]
    rule_head = LinearSVC(C=0.3, class_weight="balanced").fit(X[[rule_rows[j] for j in keep]], [rule_targets[j] for j in keep])
    rule_head.coef_ = rule_head.coef_.astype(np.float32)
    log(f"rule head: {len(rule_head.classes_)} rule points from {len(keep)} admin-cited complaints")

    # answer memory (style exemplars for the optional LLM)
    rng = np.random.default_rng(seed)
    with_admin = [i for i, t in enumerate(tickets) if any(m.get("role") == "admin" for m in t["messages"])]
    pick = sorted(rng.choice(with_admin, size=min(memory_size, len(with_admin)), replace=False).tolist())
    mem_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), min_df=3, max_features=60000, sublinear_tf=True, dtype=np.float32)
    mem_X = mem_vec.fit_transform([texts[i] for i in pick])
    mem_meta = [{"category": labels[i], "server_lang": tickets[i].get("server_lang"),
                 "player": player_text(tickets[i])[:300],
                 "admin": clean_admin(next(m["text"] for m in tickets[i]["messages"] if m.get("role") == "admin"))[:400]}
                for i in pick]

    bundle = {"format": FORMAT, "vectorizer": vec, "heads": heads, "rule_head": rule_head,
              "memory": {"vectorizer": mem_vec, "matrix": mem_X, "meta": mem_meta},
              "label_kind": "weak_labels_from_admin_outcomes"}
    path = out_dir / "model_v3.pkl.gz"
    with gzip.open(path, "wb", compresslevel=6) as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"format": FORMAT, "sha256": digest, "trained_on": "tickets_train.jsonl.gz",
                "n_train": len(tickets), "svm_c": svm_c, "subset": include_ids is not None, "seconds": round(time.time() - started, 1),
                "label_kind": bundle["label_kind"], "rule_head_classes": list(rule_head.classes_)}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf8")
    log(f"saved {path} ({path.stat().st_size / 1e6:.1f} MB) in {manifest['seconds']} s")
    return path


def load(model_dir: Path) -> dict:
    """Load a self-built artifact after verifying its SHA-256 against the manifest."""
    model_dir = Path(model_dir)
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf8"))
    path = model_dir / "model_v3.pkl.gz"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest.get("sha256") or manifest.get("format") != FORMAT:
        raise ValueError("v3 model artifact hash/format mismatch: retrain with `python -m support_copilot.v3 train`")
    bundle = pickle.loads(gzip.decompress(raw))  # trusted: produced locally by train() and hash-pinned above
    bundle["manifest"] = manifest
    return bundle


def softmax(scores, scale: float = 2.0):
    z = np.asarray(scores, dtype=np.float64) * scale
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()
