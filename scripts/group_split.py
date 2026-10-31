"""Group-aware train/dev/test split over tickets_train + tickets_valid (no labels, no model scores).

Tickets are linked into connected components (union-find) when they share
  * reporter_hash;
  * the same normalized joined player text (>= MIN_TEXT chars);
  * the same normalized individual player message (>= MIN_TEXT chars);
  * a near-copy of the joined text: char-5 shingle Jaccard >= NEAR_J (MinHash LSH candidates,
    exact Jaccard verification), texts >= MIN_NEAR chars.
Normalization masks nicknames (Name_Surname99), numbers, <url>/<digits> placeholders and punctuation,
so "repeat with another nick / number / punctuation" collapses to the same key.

Assignment: every component that touches tickets_valid (already used for development) is DEV.
Of the remaining components a seeded random subset (largest components excluded) becomes TEST,
the rest TRAIN. Output: .private/split/split.json + split_manifest.json (hashes, sizes, limits).

Limitations (also written to the manifest): semantic paraphrases and very short texts
(< MIN_TEXT chars, e.g. "помогите") are not linked unless they share a reporter.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MIN_TEXT, MIN_NEAR, NEAR_J = 15, 20, 0.85
NICK = re.compile(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b")


def norm(text: str) -> str:
    text = NICK.sub(" nick ", text or "")
    text = text.casefold().replace("ё", "е")
    text = re.sub(r"<\w+>", " ", text)
    text = re.sub(r"\d+", "0", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def player_msgs(t):
    return [m.get("text", "") for m in t.get("messages", []) if m.get("role") == "player"]


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)


def shingles(s: str, k: int = 5) -> np.ndarray:
    return np.unique(np.fromiter((zlib.crc32(s[i:i + k].encode()) for i in range(max(1, len(s) - k + 1))), dtype=np.uint64))


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.intersect1d(a, b, assume_unique=True).size
    return inter / (a.size + b.size - inter)


def near_pairs(texts: list[str], perms: int = 64, bands: int = 16, seed: int = 1):
    rng = np.random.default_rng(seed)
    P = np.uint64((1 << 61) - 1)
    A = rng.integers(1, 1 << 31, perms, dtype=np.uint64)
    B = rng.integers(0, 1 << 31, perms, dtype=np.uint64)
    sh = [shingles(t) for t in texts]
    rows = perms // bands
    buckets = defaultdict(list)
    for i, s in enumerate(sh):
        sig = ((np.outer(s, A) + B) % P).min(axis=0)
        for b in range(bands):
            buckets[(b, sig[b * rows:(b + 1) * rows].tobytes())].append(i)
    seen, out, compared = set(), [], 0
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, min(len(members), x + 200)):  # bounded per bucket
                i, j = members[x], members[y]
                if (i, j) in seen:
                    continue
                seen.add((i, j))
                compared += 1
                if jaccard(sh[i], sh[j]) >= NEAR_J:
                    out.append((i, j))
    return out, compared


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=".private/data")
    ap.add_argument("--out", default=".private/split")
    ap.add_argument("--test-share", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    train_path, valid_path = data / "tickets_train.jsonl.gz", data / "tickets_valid.jsonl"
    rows = [(json.loads(l), "train") for l in gzip.open(train_path, "rt", encoding="utf8") if l.strip()]
    rows += [(json.loads(l), "valid") for l in open(valid_path, encoding="utf8") if l.strip()]
    n = len(rows)
    dsu = DSU(n)
    links = Counter()

    def link_by(keyfn, name):
        first = {}
        for i, (t, _) in enumerate(rows):
            for key in keyfn(t):
                if key in first:
                    if dsu.find(first[key]) != dsu.find(i):
                        links[name] += 1
                    dsu.union(first[key], i)
                else:
                    first[key] = i

    link_by(lambda t: [t.get("reporter_hash")] if t.get("reporter_hash") else [], "reporter")
    joined = [norm(" ".join(player_msgs(t))) for t, _ in rows]
    first = {}
    for i, s in enumerate(joined):
        if len(s) >= MIN_TEXT:
            if s in first:
                if dsu.find(first[s]) != dsu.find(i):
                    links["joined_text"] += 1
                dsu.union(first[s], i)
            else:
                first[s] = i
    link_by(lambda t: [m for m in (norm(x) for x in player_msgs(t)) if len(m) >= MIN_TEXT], "message")

    uniq = {}
    for i, s in enumerate(joined):
        if len(s) >= MIN_NEAR:
            uniq.setdefault(s, i)
    utexts, uidx = list(uniq.keys()), list(uniq.values())
    pairs, compared = near_pairs(utexts)
    for x, y in pairs:
        i, j = uidx[x], uidx[y]
        if dsu.find(i) != dsu.find(j):
            links["near_copy"] += 1
        dsu.union(i, j)

    comp = defaultdict(list)
    for i in range(n):
        comp[dsu.find(i)].append(i)
    sizes = sorted((len(v) for v in comp.values()), reverse=True)
    dev_roots = {dsu.find(i) for i, (_, src) in enumerate(rows) if src == "valid"}
    rest = [r for r in comp if r not in dev_roots]
    n_rest = sum(len(comp[r]) for r in rest)
    cap = max(50, int(0.002 * n_rest))  # a single huge component would dominate the test set
    rng = random.Random(a.seed)
    candidates = sorted(r for r in rest if len(comp[r]) <= cap)
    rng.shuffle(candidates)
    test_roots, test_rows = set(), 0
    for r in candidates:
        if test_rows >= a.test_share * n_rest:
            break
        test_roots.add(r)
        test_rows += len(comp[r])

    assign = {}
    for r, members in comp.items():
        part = "dev" if r in dev_roots else "test" if r in test_roots else "train"
        for i in members:
            t, src = rows[i]
            assign[t["ticket_id"]] = {"split": part, "source": src, "group": int(r)}
    blob = json.dumps(assign, sort_keys=True).encode()
    (out / "split.json").write_bytes(blob)
    counts = Counter((v["split"], v["source"]) for v in assign.values())
    # leakage self-check: no reporter / joined text / message key shared between test and train
    key_parts = defaultdict(set)
    for i, (t, _) in enumerate(rows):
        part = assign[t["ticket_id"]]["split"]
        keys = [("r", t.get("reporter_hash"))] + ([("j", joined[i])] if len(joined[i]) >= MIN_TEXT else []) + \
               [("m", m) for m in (norm(x) for x in player_msgs(t)) if len(m) >= MIN_TEXT]
        for k in keys:
            key_parts[k].add(part)
    leaks = Counter(k[0] for k, parts in key_parts.items() if "test" in parts and len(parts) > 1)
    manifest = {
        "schema": 1, "seed": a.seed, "test_share": a.test_share,
        "inputs_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (train_path, valid_path)},
        "split_sha256": hashlib.sha256(blob).hexdigest(),
        "rows": n, "components": len(comp), "largest_components": sizes[:5], "test_component_cap": cap,
        "links_merged": dict(links), "near_copy_pairs": len(pairs), "near_pairs_compared": compared,
        "counts": {f"{s}/{src}": c for (s, src), c in sorted(counts.items())},
        "test_leak_keys_shared_with_other_parts": dict(leaks),
        "policy": {"min_text_chars": MIN_TEXT, "near_copy": f"char5 Jaccard>={NEAR_J}, texts>={MIN_NEAR} chars, MinHash 64x16 LSH",
                   "normalization": "nick/number/<placeholder>/punctuation masked, casefold, ё->е",
                   "dev": "all components touching tickets_valid (valid was used for development)",
                   "test": "seeded random components not touching valid, components above the cap stay in train"},
        "limitations": ["semantic paraphrases are not linked", f"texts shorter than {MIN_TEXT} chars link only via reporter",
                        "LSH may miss some near copies (probabilistic, bounded comparisons per bucket)",
                        "test labels are weak (from admin outcomes / keywords), no human gold"],
        "historically_blind": False,
    }
    (out / "split_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf8")
    print(json.dumps({k: manifest[k] for k in ("rows", "components", "largest_components", "links_merged", "counts",
                                                 "test_leak_keys_shared_with_other_parts", "split_sha256")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
