"""Build source-extractive RU SFT silver data from a private, versioned recipe.

No historical administrator replies, model drafts, network or model weights.
Exact source support is checked mechanically; human quality is not implied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.coretypes import RetrievedRule
from support_copilot.generation import build_messages
from support_copilot.intake import intake_reply


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def conduct(text):
    """Strip sanction suffix at an outside-brackets pipe, preserving [GZ | ...]."""
    depth = 0
    for i, char in enumerate(text):
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth-1)
        elif char == "|" and depth == 0:
            return text[:i].strip()
    return text.strip()


def make_example(item, rules):
    if set(item) - {"id", "question", "category", "canonical_id", "quote", "intake_topic"}:
        raise ValueError("Freeform assistant targets or unknown fields are not accepted")
    question = item["question"]
    if not isinstance(question, str) or not 10 <= len(question) <= 2000:
        raise ValueError("Invalid authored question")
    key = item.get("canonical_id")
    context = []
    if key:
        rule = rules[key]
        quote = item["quote"].strip()
        source = conduct(rule.text)
        if not quote or quote not in source or len(quote) < 35:
            raise ValueError("Quote must be an exact substantive substring before sanctions")
        if re.search(r"<\||<think>|\[rules:|\[internal:", quote):
            raise ValueError("Unexpected source controls")
        context = [RetrievedRule(rule.canonical_id, rule.section_id, rule.section_title, rule.ref, source)]
        target = "По этому правилу: «" + quote + "» [" + key + "]. Конкретный случай требует проверки сотрудником."
    else:
        if item.get("quote") or item.get("intake_topic") not in {"account", "payment", "other"}:
            raise ValueError("Source-free examples must use explicit safe intake templates")
        target = intake_reply("ru", {item["intake_topic"]})
    messages = build_messages(question, "ru", context, item["category"])
    return {"messages": messages + [{"role": "assistant", "content": target}]}, {
        "id": item["id"], "language": "ru", "question_sha256": hashlib.sha256(question.casefold().encode()).hexdigest(),
        "source_id": key, "source_quote_verified_exact": bool(key), "human_gold": False,
        "label_kind": "source_extractive_silver_or_explicit_intake_not_human_gold"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recipe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    private = (ROOT / ".private").resolve()
    output = args.output.resolve()
    if not output.is_relative_to(private) or output.exists():
        raise ValueError("Choose a new private output directory")
    recipe = json.loads(args.recipe.read_text(encoding="utf8"))
    if recipe.get("human_gold") is not False or set(recipe) != {"human_gold", "train", "heldout"}:
        raise ValueError("Explicit non-gold train/heldout recipe required")
    source = ROOT / ".private/data/rules.json"
    rules = {f"rules:{s['id']}:{point['ref']}": RetrievedRule(
        f"rules:{s['id']}:{point['ref']}", s['id'], s['category'], point['ref'], point['text'])
        for s in json.loads(source.read_text(encoding="utf8"))["rules"] for point in s["points"]}
    splits, seen, ids = {}, set(), set()
    for split in ("train", "heldout"):
        rows, metadata = [], []
        for item in recipe[split]:
            row, meta = make_example(item, rules)
            if meta["question_sha256"] in seen or meta["id"] in ids:
                raise ValueError("Duplicate question or ID across training/heldout")
            seen.add(meta["question_sha256"]); ids.add(meta["id"])
            rows.append(row); metadata.append(meta)
        if not rows:
            raise ValueError("Empty split")
        splits[split] = (rows, metadata)
    output.mkdir(parents=True)
    for split, (rows, metadata) in splits.items():
        for suffix, values in (("", rows), ("_metadata", metadata)):
            with (output / (split+suffix+".jsonl")).open("x", encoding="utf8") as f:
                for row in values:
                    f.write(json.dumps(row, ensure_ascii=False)+"\n")
    manifest = {"human_gold": False, "recipe_sha256": sha(args.recipe), "rules_sha256": sha(source),
        "train_rows": len(splits["train"][0]), "heldout_rows": len(splits["heldout"][0]),
        "language": "ru_only", "historical_ticket_targets": 0, "generated_model_targets": 0,
        "train_sha256": sha(output/"train.jsonl"), "heldout_sha256": sha(output/"heldout.jsonl"),
        "production_promotion": "requires_matched_generation_evaluation_not_loss"}
    (output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf8")
    print(json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
