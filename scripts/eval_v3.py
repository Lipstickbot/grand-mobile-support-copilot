"""Evaluate v3 vs the previous pipeline pieces on tickets_valid.jsonl.

Usage: python -X utf8 scripts/eval_v3.py [--out .private/v3/metrics.json]
Labels: (1) weak labels from admin outcomes (agreement, not truth); (2) 160 manual
labels (.private/v3/hand_labels.json); retrieval gold = rule points admins cited.
"""
import argparse, json, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from support_copilot.v3.pipeline import TriageV3
from support_copilot.v3.labels import weak_category, weak_priority
from support_copilot.v3.rulekb import admin_rule_ids
from support_copilot.policy import classify as legacy_classify, select_relevant_rules
from support_copilot.text import analyze_language, normalize_retrieval_query

def f1_macro(y, p):
    labels = sorted(set(y) | set(p)); out = []
    for l in labels:
        tp = sum(a == b == l for a, b in zip(y, p)); fp = sum(b == l != a for a, b in zip(y, p)); fn = sum(a == l != b for a, b in zip(y, p))
        if tp + fn == 0: continue
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn)
        out.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(out))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default=str(ROOT / ".private/data")); ap.add_argument("--model", default=str(ROOT / ".private/v3"))
    ap.add_argument("--out", default=str(ROOT / ".private/v3/metrics.json")); a = ap.parse_args()
    eng = TriageV3(Path(a.data), Path(a.model)).load()
    valid = [json.loads(l) for l in open(Path(a.data) / "tickets_valid.jsonl", encoding="utf8")]
    hand = json.loads((Path(a.model) / "hand_labels.json").read_text(encoding="utf8")) if (Path(a.model) / "hand_labels.json").exists() else {}
    order = sorted(range(len(valid)), key=lambda i: valid[i].get("created_at") or "")
    preds, lat = [None] * len(valid), []
    for i in order:
        t = time.monotonic(); preds[i] = eng.triage(valid[i]); lat.append(time.monotonic() - t)
    legacy = []
    for r in valid:
        pl = "\n".join(m["text"] for m in r["messages"] if m["role"] == "player")
        an = analyze_language(pl, r.get("server_lang"), r.get("region"))
        q = normalize_retrieval_query(an["normalized_text"])[0]
        c, p, _ = legacy_classify(q, r["kind"]); legacy.append((c, p, an["language"], q))
    weak = [weak_category(r)[0] for r in valid]; weakp = [weak_priority(r, c) for r, c in zip(valid, weak)]
    M = {"n_valid": len(valid)}
    M["category_vs_weak_labels"] = {"v3_acc": round(np.mean([p["category"] == w for p, w in zip(preds, weak)]), 3),
                                    "legacy_keyword_acc": round(np.mean([l[0] == w for l, w in zip(legacy, weak)]), 3)}
    M["priority_macroF1_vs_weak"] = {"v3": round(f1_macro(weakp, [p["priority"] for p in preds]), 3),
                                     "legacy_keyword": round(f1_macro(weakp, [l[1] for l in legacy]), 3)}
    idx = {r["ticket_id"]: i for i, r in enumerate(valid)}
    for split in ("dev", "fresh"):
        if split in hand:
            items = [(idx[t], h) for t, h in hand[split].items() if t in idx]
            M[f"category_vs_manual_{split}"] = {"n": len(items), "v3": sum(preds[i]["category"] == h for i, h in items),
                                                "legacy_keyword": sum(legacy[i][0] == h for i, h in items),
                                                "weak_labeler_upper_ref": sum(weak[i] == h for i, h in items)}
    # retrieval vs admin-cited rules
    ev = [(i, admin_rule_ids(r, eng.known)) for i, r in enumerate(valid) if r["kind"] == "complaint"]
    ev = [(i, g) for i, g in ev if g]
    rec = defaultdict(lambda: Counter())
    for i, g in ev:
        r = valid[i]; text = "\n".join(m["text"] for m in r["messages"] if m["role"] == "player")
        X, cat, _, _ = eng.classify(r["kind"], text, r.get("rule_refs_raw"))
        new = [d.canonical_id for d, _, _ in eng.retrieve(X, preds[i]["category"], text, r.get("rule_refs_raw"))]
        refs = sorted(set(re.findall(r"\b\d+(?:\.\d+){1,3}\b", r.get("rule_refs_raw") or "")))
        d = eng.retriever.search_with_diagnostics(legacy[i][3], rule_refs=refs, limit=10)
        old = [x.canonical_id for x in d["results"]]
        gated = [x.canonical_id for x in select_relevant_rules(legacy[i][3], d["results"], cues=d.get("query_cues"), events=d.get("event_sections"))]
        cited_ids = [x.split("=", 1)[1] for x in preds[i]["reasons"] if x.startswith("cited_id=")]
        for name, res in (("v3", new), ("legacy_bm25", old), ("legacy_bm25_gate", gated)):
            for k in (1, 3, 5): rec[name][f"R@{k}"] += bool(set(res[:k]) & set(g))
        rec["v3_draft"]["cited_matches_admin_rule"] += bool(set(cited_ids) & set(g))
        rec["v3_draft"]["has_citation"] += bool(cited_ids)
    M["retrieval_vs_admin_cited_rule"] = {"n": len(ev), **{n: {k: round(v / len(ev), 3) for k, v in c.items()} for n, c in rec.items()}}
    # hallucination / safety
    all_refs = {c.split(":")[2] for c in eng.known}
    cited = [c for p in preds for c in p["draft_reply"]["cited"]]
    M["citations"] = {"drafts_with_citation": round(np.mean([bool(p["draft_reply"]["cited"]) for p in preds]), 3),
                      "cited_refs_exist_in_rules": f"{sum(c in all_refs for c in cited)}/{len(cited)}",
                      "complaint_drafts_with_citation": round(np.mean([bool(p["draft_reply"]["cited"]) for p, r in zip(preds, valid) if r["kind"] == "complaint"]), 3)}
    from support_copilot.v3.pipeline import PROMISE
    M["promise_violations"] = sum(bool(PROMISE.search(p["draft_reply"]["text"])) for p in preds)
    M["needs_human_share"] = round(np.mean([p["needs_human"] for p in preds]), 3)
    M["needs_human_share_by_kind"] = {k: round(np.mean([p["needs_human"] for p, r in zip(preds, valid) if r["kind"] == k]), 3) for k in ("complaint", "report")}
    M["spam_or_duplicate_flagged"] = sum(p["is_spam_or_duplicate"] for p in preds)
    M["language"] = {"v3": dict(Counter(p["lang"] for p in preds)), "legacy_detector": dict(Counter(l[2] for l in legacy))}
    lat.sort(); M["latency_ms_no_llm"] = {"p50": round(lat[len(lat) // 2] * 1e3, 2), "p95": round(lat[int(.95 * (len(lat) - 1))] * 1e3, 2)}
    by_lang = defaultdict(list)
    for p, w in zip(preds, weak):
        by_lang[p["lang"]].append(p["category"] == w)
    M["category_vs_weak_by_reply_lang"] = {k: {"n": len(v), "acc": round(float(np.mean(v)), 3)} for k, v in sorted(by_lang.items())}
    M["draft_chars_mean"] = round(float(np.mean([len(p["draft_reply"]["text"]) for p in preds])), 1)
    M["drafts_with_sanction_info"] = round(float(np.mean([("предусмотрено" in p["draft_reply"]["text"] or "nazarda" in p["draft_reply"]["text"]
                                                          or "көзделген" in p["draft_reply"]["text"] or "provide" in p["draft_reply"]["text"]) for p in preds])), 3)
    M["category_distribution"] = dict(Counter(p["category"] for p in preds).most_common())
    M["priority_distribution"] = dict(Counter(p["priority"] for p in preds).most_common())
    Path(a.out).write_text(json.dumps(M, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps(M, ensure_ascii=False, indent=1))

if __name__ == "__main__":
    main()
