"""Stream the licensed handout locally; only aggregate statistics leave memory.

Review rows are private, blinded to model/admin outcomes and have empty labels.
They are NOT a verified evaluation set until independent reviewers annotate them.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.text import analyze_language, normalize_retrieval_query
from support_copilot.policy import classify


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def player_view(row):
    return "\n".join(m["text"] for m in row.get("messages", [])
                     if m.get("role") == "player" and isinstance(m.get("text"), str))


def key(text):
    return digest(" ".join(text.casefold().replace("ё", "е").split()))


def template_key(text):
    text = re.sub(r"\b[A-Za-z]+_[A-Za-z]+\b|<[^>]+>|\d+", " TOKEN ", text.casefold())
    return key(text)


def new_stats():
    return {"n": 0, "kind": Counter(), "source_heuristic_lang": Counter(), "detected_lang": Counter(),
            "kind_lang": Counter(), "server_lang": Counter(), "lengths": [], "flags": Counter(),
            "normalization_flags": Counter(), "keyword_category_proxy_not_ground_truth": Counter(),
            "verdict_counts_not_category_labels": Counter(), "admin_template_hashes": Counter(),
            "texts": Counter(), "reporters": set(), "schema_keys": Counter()}


def add(stats, row):
    text = player_view(row)
    a = analyze_language(text, row.get("server_lang"), row.get("region"))
    normalized, flags = normalize_retrieval_query(a["normalized_text"])
    stats["n"] += 1
    for field, value in (("kind", row.get("kind", "missing")),
                         ("source_heuristic_lang", row.get("lang", "missing")),
                         ("detected_lang", a["language"]), ("server_lang", row.get("server_lang", "missing"))):
        stats[field][str(value)] += 1
    stats["kind_lang"][str(row.get("kind"))+"/"+str(row.get("lang", "missing"))] += 1
    stats["lengths"].append(len(text))
    stats["texts"][key(a["normalized_text"])] += 1
    if row.get("reporter_hash"):
        stats["reporters"].add(row["reporter_hash"])
    stats["schema_keys"].update(row.keys())
    stats["flags"].update({"empty": not text.strip(), "chars_le_10": len(text) <= 10,
        "chars_le_30": len(text) <= 30, "chars_gt_2000": len(text) > 2000,
        "mixed_language_heuristic": a["mixed"], "legacy_kk_restored": bool(a["normalization_flags"]),
        "latin_and_cyrillic_script": bool(re.search(r"[a-zA-Z]", text) and re.search(r"[а-яА-Я]", text)),
        "replacement_character": "\ufffd" in text, "has_images_count": row.get("images_count", 0) > 0,
        "category_label_present": row.get("category") is not None,
        "priority_label_present": row.get("priority") is not None})
    stats["normalization_flags"].update(flags)
    stats["keyword_category_proxy_not_ground_truth"][classify(normalized, row["kind"])[0]] += 1
    stats["verdict_counts_not_category_labels"][str(row.get("verdict", "missing"))] += 1
    for m in row.get("messages", []):
        if m.get("role") == "admin" and isinstance(m.get("text"), str):
            stats["admin_template_hashes"][template_key(m["text"])] += 1
    return text, a


def aggregate(stats):
    lengths = sorted(stats["lengths"])
    templates = stats["admin_template_hashes"]
    out = {k: dict(v) for k, v in stats.items() if isinstance(v, Counter)
           and k not in {"texts", "admin_template_hashes"}}
    out.update(n=stats["n"], distinct_reporters=len(stats["reporters"]),
        normalized_distinct_texts=len(stats["texts"]),
        duplicate_excess_rows=sum(v-1 for v in stats["texts"].values()),
        rows_in_repeated_text_groups=sum(v for v in stats["texts"].values() if v > 1),
        text_chars={"p50": lengths[len(lengths)//2], "p95": lengths[min(len(lengths)-1, int(len(lengths)*.95))],
                    "max": max(lengths), "mean": sum(lengths)/len(lengths)},
        admin_template_skeleton={"messages": sum(templates.values()), "distinct": len(templates),
            "largest_cluster_sizes": sorted(templates.values(), reverse=True)[:10],
            "method": "lowercase, replace synthetic nicknames/placeholders/digits; NOT semantic clustering"})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path.home()/"Downloads/handout.zip")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review-size", type=int, default=100)
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/".private").resolve()) or output.exists() or not 1 <= args.review_size <= 1000:
        raise ValueError("Choose a new private output directory and review size1..1000")
    stats = {"train": new_stats(), "valid": new_stats()}
    candidates = defaultdict(list)
    with ZipFile(args.archive) as z:
        taxonomy = json.loads(z.read("handout/taxonomy.json"))
        rules = json.loads(z.read("handout/rules.json"))
        with z.open("handout/tickets_train.jsonl.gz") as c, gzip.GzipFile(fileobj=c) as stream:
            for line in stream:
                add(stats["train"], json.loads(line))
        with z.open("handout/tickets_valid.jsonl") as stream:
            for line in stream:
                row = json.loads(line)
                text, a = add(stats["valid"], row)
                # Group and text disjointness are required, not silently relaxed.
                if key(a["normalized_text"]) in stats["train"]["texts"] or row.get("reporter_hash") in stats["train"]["reporters"]:
                    continue
                record = {"review_id": digest("blinded-review-v1|"+str(row["ticket_id"])),
                    "kind": row["kind"], "messages": [{"role": "player", "text": text}],
                    "server_lang": row.get("server_lang"), "region": row.get("region"),
                    "images_count": row.get("images_count", 0), "source_language_heuristic": row.get("lang"),
                    "category": None, "priority": None, "verified_language": None,
                    "reviewer_1": None, "reviewer_2": None, "adjudicator": None,
                    "approved_for_evaluation": False, "human_gold": False}
                candidates[(row["kind"], str(row.get("lang")))].append((key(a["normalized_text"]), row.get("reporter_hash"), record))
    chosen, seen_text, seen_reporter = [], set(), set()
    for values in candidates.values():
        values.sort(key=lambda t: t[2]["review_id"], reverse=True)
    while len(chosen) < args.review_size and any(candidates.values()):
        for group in sorted(candidates):
            while candidates[group]:
                text_key, reporter, record = candidates[group].pop()
                if text_key in seen_text or reporter and reporter in seen_reporter:
                    continue
                seen_text.add(text_key)
                if reporter: seen_reporter.add(reporter)
                chosen.append(record)
                break
            if len(chosen) == args.review_size:
                break
    output.mkdir(parents=True)
    result = {"schema_version": 1, "human_gold": False,
        "sources": {"archive_sha256": hashlib.sha256(args.archive.read_bytes()).hexdigest()},
        "splits": {name: aggregate(s) for name,s in stats.items()},
        "overlap": {"reporters": len(stats["train"]["reporters"] & stats["valid"]["reporters"]),
            "normalized_text_groups": len(stats["train"]["texts"].keys() & stats["valid"]["texts"].keys())},
        "authority": {"categories": [x["code"] for x in taxonomy["categories"]],
            "rule_sections": len(rules["rules"]), "numbered_points": sum(len(s["points"]) for s in rules["rules"])},
        "review_packet": {"requested": args.review_size, "prepared": len(chosen), "verified": 0,
            "model_predictions_hidden": True, "administrator_outcomes_hidden": True,
            "group_text_disjoint_from_train": True, "representative_random_sample": False,
            "selection": "deterministic round-robin kind/source-heuristic-language; one reporter and text per packet"},
        "limitations": ["no category/priority gold", "source languages and detector are heuristics",
            "keyword proxy distribution is not category frequency", "typo normalization flags are not all spelling errors",
            "admin skeletons measure form repetition, not response quality"]}
    (output/"analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    (output/"review-packet.jsonl").write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in chosen), encoding="utf8")
    from validate_review_packet import selection_hash
    (output/"review-selection.json").write_text(json.dumps({"schema_version":1,
        "source_archive_sha256":result["sources"]["archive_sha256"],
        "request_sha256_by_review_id":{r["review_id"]:selection_hash(r) for r in chosen}},indent=2)+"\n",encoding="utf8")
    print(json.dumps({"counts": {k:s["n"] for k,s in stats.items()}, "overlap": result["overlap"], "review": result["review_packet"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
