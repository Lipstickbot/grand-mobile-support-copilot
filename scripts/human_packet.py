"""Blind packet for independent human labelling of the group-split TEST (private, never published).

Contains only what an admin sees from the player: kind, server_lang, player messages, images_count,
rule_refs_raw. No model predictions, no weak labels, no admin replies / verdict / punishment.
Sampling: seeded, stratified by kind x server_lang only (model-free). Gold fields are empty;
compare_v8 blocks human accuracy until every row has gold_category filled by a human reviewer.

  python scripts/human_packet.py --n 240
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIV = ROOT / ".private"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    split = json.loads((PRIV / "split/split.json").read_text(encoding="utf8"))
    rows = [json.loads(l) for l in gzip.open(PRIV / "data/tickets_train.jsonl.gz", "rt", encoding="utf8")]
    test = [r for r in rows if split[r["ticket_id"]]["split"] == "test"]
    strata = defaultdict(list)
    for r in test:
        strata[(r["kind"], r.get("server_lang"))].append(r)
    rng = random.Random(a.seed)
    picked = []
    for key in sorted(strata, key=str):
        k = max(1, round(a.n * len(strata[key]) / len(test)))
        picked += rng.sample(strata[key], min(k, len(strata[key])))
    rng.shuffle(picked)
    out = PRIV / "v8/human_packet"
    out.mkdir(parents=True, exist_ok=True)
    taxonomy = json.loads((PRIV / "data/taxonomy.json").read_text(encoding="utf8"))
    key = {}
    with open(out / "packet.jsonl", "w", encoding="utf8") as f:
        for i, r in enumerate(picked, 1):
            bid = f"H{i:03d}"
            key[bid] = r["ticket_id"]
            f.write(json.dumps({"blind_id": bid, "kind": r["kind"], "server_lang": r.get("server_lang"),
                                "rule_refs_raw": r.get("rule_refs_raw"), "images_count": r.get("images_count"),
                                "player_messages": [m["text"] for m in r["messages"] if m.get("role") == "player"],
                                "gold_category": "", "gold_priority": "", "gold_lang": "", "reviewer": "", "notes": ""},
                               ensure_ascii=False) + "\n")
    (out / "key_DO_NOT_SHOW_TO_REVIEWER.json").write_text(json.dumps(key, indent=1), encoding="utf8")
    (out / "INSTRUCTIONS.md").write_text(
        "# Разметка (только человек)\n\nДля каждой строки `packet.jsonl` заполните `gold_category` (код из списка ниже, "
        "только с префиксом своего `kind`), `gold_priority` (p0–p3 по критериям ниже), `gold_lang` (ru/kk/uz/en/es/pt) и "
        "`reviewer` (имя). Не смотрите прогнозы модели и ответы администраторов. Спорные случаи — отметьте в `notes`.\n\n"
        + "\n".join(f"- `{c['code']}` — {c.get('description', c.get('title', ''))}" for c in taxonomy["categories"])
        + "\n\nПриоритеты:\n" + json.dumps(taxonomy.get("priority", taxonomy.get("priorities", {})), ensure_ascii=False, indent=1) + "\n",
        encoding="utf8")
    meta = {"n": len(picked), "seed": a.seed, "from": "group-split TEST", "strata": "kind x server_lang",
            "packet_sha256": hashlib.sha256((out / "packet.jsonl").read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256((PRIV / "split/split.json").read_bytes()).hexdigest(),
            "gold_filled": 0, "human_accuracy": "BLOCKED until gold_category is filled by a human"}
    (out / "packet_manifest.json").write_text(json.dumps(meta, indent=1), encoding="utf8")
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
