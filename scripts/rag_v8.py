"""Retriever comparison on group-split TEST complaints, one component at a time.

Gold = rule the admin applied (canonical id WITH section, e.g. rules:30:2.1). Same queries, same rules.json,
same split model (.private/v8/model_split). Player-cited numbers are hints (a separate variant), not evidence.

  python scripts/rag_v8.py [--e5 .private/v8/e5 --rule-index ../.private/index]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from compare_v8 import DATA, OUT, load_rows, sha  # noqa: E402
from support_copilot.text import normalize_retrieval_query  # noqa: E402
from support_copilot.v3 import model as modelmod  # noqa: E402
from support_copilot.v3.pipeline import TriageV3  # noqa: E402
from support_copilot.v3.rulekb import admin_rule_ids, load_rule_ids, parse_refs  # noqa: E402


def rank_from(scores: dict) -> list[str]:
    return [k for k, _ in sorted(scores.items(), key=lambda x: -x[1])]


def rrf(*rankings, k=60):
    s = defaultdict(float)
    for r in rankings:
        for i, c in enumerate(r):
            s[c] += 1 / (k + i + 1)
    return rank_from(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e5", default=None)
    ap.add_argument("--rule-index", default=None)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = [r for r in load_rows() if r["_split"] == a.split and r["kind"] == "complaint"]
    eng = TriageV3(DATA, OUT / "model_split").load()
    known = load_rule_ids(DATA / "rules.json")
    dense = None
    if a.e5 and a.rule_index:
        ids = json.loads((Path(a.e5) / "ids.json").read_text(encoding="utf8"))
        emb = np.load(Path(a.e5) / "emb.npy").astype(np.float32)
        pos = {t: i for i, t in enumerate(ids)}
        rv = np.load(Path(a.rule_index) / "vectors.npy").astype(np.float32)
        meta = json.loads((Path(a.rule_index) / "metadata.json").read_text(encoding="utf8"))
        rule_ids = meta["ids"]
        dense = (emb, pos, rv, rule_ids)
    # ambiguity of plain numbers: how many gold refs exist in more than one section
    plain = Counter(c.split(":")[2] for c in known)
    V = defaultdict(list)
    n = 0
    for r in rows:
        gold = admin_rule_ids(r, known)
        if not gold:
            continue
        n += 1
        g = gold[0]
        text = modelmod.player_text(r)
        X, cat, _, _ = eng.classify("complaint", text, r.get("rule_refs_raw"))
        head = eng.bundle["rule_head"]
        p = modelmod.softmax(head.decision_function(X)[0], scale=3.0)
        head_s = {str(c): float(x) for c, x in zip(head.classes_, p)}
        q, _ = normalize_retrieval_query(text)
        bm = [d.canonical_id for d in eng.retriever.search_with_diagnostics(q, limit=10)["results"]]
        bm_s = {c: .15 / (i + 1) for i, c in enumerate(bm)}
        refs = parse_refs((r.get("rule_refs_raw") or "") + " " + text, known)
        prior = eng.prior.get(cat, [])

        def comb(*parts):
            s = defaultdict(float)
            for part in parts:
                for c, v in part.items():
                    s[c] += v
            return rank_from(s)
        ref_s = {c: .25 for c in refs}
        prior_s = {c: .15 for c in prior}
        V["bm25"].append((g, bm))
        V["exact_player_refs_only"].append((g, refs))
        V["learned_rule_head"].append((g, rank_from(head_s)))
        V["head+bm25"].append((g, comb(head_s, bm_s)))
        V["head+bm25+player_refs"].append((g, comb(head_s, bm_s, ref_s)))
        V["v3_full (head+bm25+refs+taxonomy_prior)"].append((g, comb(head_s, bm_s, ref_s, prior_s)))
        V["head+refs+prior (no bm25)"].append((g, comb(head_s, ref_s, prior_s)))
        if dense is not None and r["ticket_id"] in dense[1]:
            emb, pos, rv, rule_ids = dense
            sims = rv @ emb[pos[r["ticket_id"]]]
            dn = [rule_ids[i] for i in np.argsort(-sims)[:10]]
            V["dense_e5"].append((g, dn))
            V["hybrid_rrf(bm25,dense)"].append((g, rrf(bm, dn)))
            V["hybrid_rrf+player_refs"].append((g, list(dict.fromkeys(refs + rrf(bm, dn)))))
            V["hybrid_rrf+refs, reranked by learned head"].append(
                (g, sorted(dict.fromkeys(refs + rrf(bm, dn))[:10], key=lambda c: -(head_s.get(c, 0) + (.25 if c in refs else 0)))))
    res = {"n_complaints_with_admin_rule": n, "split": a.split, "model_sha256": sha(OUT / "model_split/model_v3.pkl.gz"),
           "gold_refs_ambiguous_without_section": sum(1 for v in V["bm25"] if plain[v[0].split(":")[2]] > 1),
           "note": "reranking = learned rule head re-orders hybrid top-10 (no cross-encoder available offline)"}
    for name, lst in V.items():
        res[name] = {"n": len(lst), "R@1": round(sum(h[:1] == [g] for g, h in lst) / len(lst), 4),
                     "R@5": round(sum(g in h[:5] for g, h in lst) / len(lst), 4),
                     "R@10": round(sum(g in h[:10] for g, h in lst) / len(lst), 4)}
    (OUT / f"rag_{a.split}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
