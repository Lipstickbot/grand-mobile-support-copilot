"""Group-split comparison of category / priority / retrieval (no human gold; weak labels are diagnostics).

  python scripts/compare_v8.py train          # v3 heads on split TRAIN only -> .private/v8/model_split
  python scripts/compare_v8.py eval [--e5 DIR] # all candidates on DEV (=valid) and TEST -> .private/v8/results.json

Labels: weak labels from labels.py, each with its source. "outcome" sources come from admin outcome fields
(violation / verdict / admin reply), "player_text" sources come from keyword rules over the player's own text;
the latter overlap with the pipeline's own overrides, so they are reported separately and are NOT an
independent check. Human gold is not available: human accuracy is blocked (see human_label_packet).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.v3 import model as modelmod  # noqa: E402
from support_copilot.v3.labels import label_complaint, label_report, weak_priority  # noqa: E402
from support_copilot.v3.rulekb import admin_rule_ids, load_rule_ids  # noqa: E402

DATA, PRIV = ROOT / ".private/data", ROOT / ".private"
OUT = PRIV / "v8"
OUTCOME_SOURCES = {"admin_appeal_outcome", "admin_invalid_format", "admin_rule_code", "admin_invalid_or_no_evidence", "admin_help_reply"}
ABLATIONS = ["multilang_keywords", "payment_account", "appeal", "invalid_to_topic", "bare_help", "full_question",
             "not_a_complaint", "keyword_backoff", "empty_description"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows():
    split = json.loads((PRIV / "split/split.json").read_text(encoding="utf8"))
    rows = [json.loads(l) for l in gzip.open(DATA / "tickets_train.jsonl.gz", "rt", encoding="utf8")]
    rows += [json.loads(l) for l in open(DATA / "tickets_valid.jsonl", encoding="utf8")]
    for r in rows:
        r["_split"] = split[r["ticket_id"]]["split"]
        r["_src_file"] = split[r["ticket_id"]]["source"]
        cat, src = (label_complaint if r["kind"] == "complaint" else label_report)(r)
        r["_weak"], r["_weak_src"] = cat, src
        r["_weak_kind"] = "outcome" if src in OUTCOME_SOURCES else "player_text"
        r["_weak_p"] = weak_priority(r, cat)
    return rows


def request(r):
    return {k: r.get(k) for k in ("ticket_id", "kind", "created_at", "region", "server_lang", "lang", "reporter_hash",
                                  "reported_hash", "reported_nick", "rule_refs_raw", "images_count")} | {
        "messages": [m for m in r["messages"] if m.get("role") == "player"]}


def macro_f1(y, p, labels=None):
    labels = sorted(set(y) | set(p)) if labels is None else labels
    f = []
    for c in labels:
        tp = sum(a == c and b == c for a, b in zip(y, p)); fp = sum(a != c and b == c for a, b in zip(y, p))
        fn = sum(a == c and b != c for a, b in zip(y, p))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f)) if f else 0.0


def per_class_f1(y, p, labels):
    return {c: round(macro_f1(y, p, [c]), 3) for c in labels}


def confusion(y, p):
    m = defaultdict(Counter)
    for a, b in zip(y, p):
        m[a][b] += 1
    return {a: dict(c) for a, c in sorted(m.items())}


def cat_report(rows, preds, labels):
    out = {}
    def block(idx):
        y = [rows[i]["_weak"] for i in idx]; p = [preds[i] for i in idx]
        return {"n": len(idx), "acc": round(float(np.mean([a == b for a, b in zip(y, p)])), 4) if idx else None,
                "macro_f1": round(macro_f1(y, p, sorted({l for l in labels if any(l == v for v in y)})), 4) if idx else None}
    all_idx = list(range(len(rows)))
    out["all"] = block(all_idx)
    for kind in ("complaint", "report"):
        out[kind] = block([i for i in all_idx if rows[i]["kind"] == kind])
    out["outcome_labels_only"] = block([i for i in all_idx if rows[i]["_weak_kind"] == "outcome"])
    out["player_text_labels_only"] = block([i for i in all_idx if rows[i]["_weak_kind"] == "player_text"])
    by_lang = defaultdict(list)
    for i in all_idx:
        by_lang[rows[i].get("_lang") or "?"].append(i)
    out["by_detected_lang"] = {k: block(v) for k, v in sorted(by_lang.items())}
    return out


def prio_report(y, p):
    tp = sum(a == "p0" and b == "p0" for a, b in zip(y, p)); fp = sum(a != "p0" and b == "p0" for a, b in zip(y, p))
    fn = sum(a == "p0" and b != "p0" for a, b in zip(y, p)); neg = sum(a != "p0" for a in y)
    return {"n": len(y), "p0_support": tp + fn, "p0_tp": tp, "p0_fp": fp, "p0_fn": fn,
            "p0_recall": round(tp / (tp + fn), 4) if tp + fn else None, "p0_precision": round(tp / (tp + fp), 4) if tp + fp else None,
            "p0_fpr": round(fp / neg, 5) if neg else None,
            "f1": per_class_f1(y, p, ["p0", "p1", "p2", "p3"]),
            "macro_f1_all": round(macro_f1(y, p, ["p0", "p1", "p2", "p3"]), 4),
            "macro_f1_p1_p3": round(macro_f1(y, p, ["p1", "p2", "p3"]), 4)}


def cmd_train(_):
    rows = load_rows()
    ids = {r["ticket_id"] for r in rows if r["_split"] == "train"}
    started = time.time()
    modelmod.train(DATA, OUT / "model_split", include_ids=ids, log=lambda m: print(m, flush=True))
    print("trained on", len(ids), "tickets in", round(time.time() - started), "s")


def plain_tfidf_svm(train_rows):
    """Baseline 3: standard TF-IDF word(1-2)+char_wb(3-5) -> LinearSVC(balanced), player text only."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import make_union
    from sklearn.svm import LinearSVC
    vec = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000, dtype=np.float32),
                     TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, max_features=30000, dtype=np.float32))
    X = vec.fit_transform([modelmod.player_text(r) for r in train_rows])
    heads = {}
    for kind in ("complaint", "report"):
        idx = [i for i, r in enumerate(train_rows) if r["kind"] == kind]
        heads[kind] = LinearSVC(C=1.0, class_weight="balanced", random_state=42).fit(X[idx], [train_rows[i]["_weak"] for i in idx])
    return vec, heads


def cmd_eval(a):
    from support_copilot.v3.pipeline import TriageV3
    rows = load_rows()
    labels = [c["code"] for c in json.loads((DATA / "taxonomy.json").read_text(encoding="utf8"))["categories"]]
    train_rows = [r for r in rows if r["_split"] == "train"]
    parts = {"dev_valid": [r for r in rows if r["_split"] == "dev" and r["_src_file"] == "valid"],
             "test": [r for r in rows if r["_split"] == "test"]}
    eng = TriageV3(DATA, OUT / "model_split").load()
    t0 = time.time(); vec3, heads3 = plain_tfidf_svm(train_rows); fit_s = time.time() - t0
    e5 = None
    if a.e5:
        e5 = load_e5(Path(a.e5), rows, train_rows)
    known = load_rule_ids(DATA / "rules.json")
    res = {"provenance": {
        "split_sha256": sha(PRIV / "split/split.json"), "model_sha256": sha(OUT / "model_split/model_v3.pkl.gz"),
        "data_sha256": {p: sha(DATA / p) for p in ("tickets_train.jsonl.gz", "tickets_valid.jsonl", "taxonomy.json", "rules.json")},
        "code_sha256": {p: sha(ROOT / p) for p in ("src/support_copilot/v3/pipeline.py", "src/support_copilot/v3/model.py",
                                                    "src/support_copilot/v3/labels.py", "scripts/compare_v8.py")},
        "n_train": len(train_rows), "label_source": "weak labels (labels.py); no human gold",
        "train_label_sources": dict(Counter(r["_weak_src"] for r in train_rows)),
        "seeds": {"split": 7, "memory": 13, "baseline_svm": 42}, "plain_tfidf_fit_seconds": round(fit_s, 1)}}
    for name, part in parts.items():
        R = {}
        reqs = [request(r) for r in part]
        full, head, lat_full, lat_head, prios, cited_ok, rule_hits = [], [], [], [], [], [], []
        for r, q in zip(part, reqs):
            t = time.perf_counter()
            text = modelmod.player_text(r)
            _, c, _, _ = eng.classify(r["kind"], text, r.get("rule_refs_raw"))
            lat_head.append((time.perf_counter() - t) * 1000); head.append(c)
            t = time.perf_counter()
            eng.duplicates = type(eng.duplicates)()  # dedup state must not leak across evaluation rows
            o = eng.triage(q)
            lat_full.append((time.perf_counter() - t) * 1000)
            full.append(o["category"]); prios.append(o["priority"])
            r["_lang"] = o["lang"]
            if r["kind"] == "complaint":
                gold = admin_rule_ids(r, known)
                if gold:
                    X, cat, _, _ = eng.classify(r["kind"], text, r.get("rule_refs_raw"))
                    hits = [d.canonical_id for d, _, _ in eng.retrieve(X, o["category"], text, r.get("rule_refs_raw"), k=5)]
                    cited = [x.split("=", 1)[1] for x in o["reasons"] if x.startswith("cited_id=")]
                    rule_hits.append((gold[0], hits, cited))
        t = time.perf_counter()
        Xb = vec3.transform([modelmod.player_text(r) for r in part])
        plain = [str(heads3[r["kind"]].predict(Xb[i])[0]) for i, r in enumerate(part)]
        plain_ms = (time.perf_counter() - t) * 1000 / max(1, len(part))
        R["category"] = {"v3_full_pipeline": cat_report(part, full, labels), "v3_ml_head_only": cat_report(part, head, labels),
                         "plain_tfidf_svm": cat_report(part, plain, labels)}
        if e5 is not None:
            R["category"]["e5_logreg"] = cat_report(part, e5["predict"](part), labels)
        R["confusion_v3_full"] = confusion([r["_weak"] for r in part], full)
        R["per_class_f1_v3_full"] = per_class_f1([r["_weak"] for r in part], full, labels)
        R["per_class_f1_head"] = per_class_f1([r["_weak"] for r in part], head, labels)
        R["latency_ms_cpu"] = {"v3_full_triage_incl_retrieval_and_draft": {"p50": round(float(np.percentile(lat_full, 50)), 2), "p95": round(float(np.percentile(lat_full, 95)), 2)},
                               "v3_head_incl_features": {"p50": round(float(np.percentile(lat_head, 50)), 2), "p95": round(float(np.percentile(lat_head, 95)), 2)},
                               "plain_tfidf_batch_mean_per_ticket": round(plain_ms, 3)}
        R["priority_v3_full_vs_weak"] = prio_report([r["_weak_p"] for r in part], prios)
        R["priority_v3_full_outcome_labels_only"] = prio_report([r["_weak_p"] for r in part if r["_weak_kind"] == "outcome"],
                                                                [p for r, p in zip(part, prios) if r["_weak_kind"] == "outcome"])
        # ablations: one override disabled at a time (category only)
        abl = {}
        for sw in ABLATIONS:
            eng.disabled = {sw}
            preds = []
            for q in reqs:
                eng.duplicates = type(eng.duplicates)()
                preds.append(eng.triage(q)["category"])
            eng.disabled = set()
            rep = cat_report(part, preds, labels)
            abl[sw] = {"acc_all": rep["all"]["acc"], "acc_outcome": rep["outcome_labels_only"]["acc"],
                       "acc_player_text": rep["player_text_labels_only"]["acc"], "changed_rows": sum(a != b for a, b in zip(preds, full))}
        eng.disabled = set(ABLATIONS)
        preds = []
        for q in reqs:
            eng.duplicates = type(eng.duplicates)()
            preds.append(eng.triage(q)["category"])
        eng.disabled = set()
        rep = cat_report(part, preds, labels)
        abl["ALL_OVERRIDES_OFF"] = {"acc_all": rep["all"]["acc"], "acc_outcome": rep["outcome_labels_only"]["acc"],
                                    "acc_player_text": rep["player_text_labels_only"]["acc"], "changed_rows": sum(a != b for a, b in zip(preds, full))}
        R["ablation_v3_overrides"] = abl
        # retrieval stages vs the rule the admin cited (canonical id incl. section)
        n = len(rule_hits)
        R["retrieval_vs_admin_rule"] = {
            "n": n, "label_source": "admin-cited rule (outcome), canonical id with section",
            "R@1": round(sum(h[:1] == [g] for g, h, _ in rule_hits) / n, 4) if n else None,
            "R@5": round(sum(g in h for g, h, _ in rule_hits) / n, 4) if n else None,
            "draft_cites_gold": round(sum(g in c for g, _, c in rule_hits) / n, 4) if n else None,
            "stage_miss": dict(Counter("not_in_top5" if g not in h else ("gated_out_no_citation" if not c else
                                                                         ("cited_other_rule" if g not in c else "ok")) for g, h, c in rule_hits))}
        res[name] = R
        print(name, json.dumps({k: v["all"] for k, v in R["category"].items()}, ensure_ascii=False), flush=True)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print("->", OUT / "results.json")


def load_e5(e5_dir: Path, rows, train_rows):
    """E5 + LogisticRegression on embeddings computed on the user's GPU (scripts/embed_e5.py)."""
    from sklearn.linear_model import LogisticRegression
    ids = json.loads((e5_dir / "ids.json").read_text(encoding="utf8"))
    emb = np.load(e5_dir / "emb.npy").astype(np.float32)
    pos = {t: i for i, t in enumerate(ids)}
    heads = {}
    for kind in ("complaint", "report"):
        tr = [r for r in train_rows if r["kind"] == kind]
        heads[kind] = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000, random_state=42).fit(
            emb[[pos[r["ticket_id"]] for r in tr]], [r["_weak"] for r in tr])

    def predict(part):
        return [str(heads[r["kind"]].predict(emb[[pos[r["ticket_id"]]]])[0]) for r in part]
    return {"predict": predict, "sha256": sha(e5_dir / "emb.npy")}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("train")
    e = sp.add_parser("eval"); e.add_argument("--e5", default=None)
    a = ap.parse_args()
    {"train": cmd_train, "eval": cmd_eval}[a.cmd](a)
