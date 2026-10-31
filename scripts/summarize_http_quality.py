"""Summarize a completed private HTTP run without changing its original scores.

The expanded bracket audit was added AFTER seeing baseline outputs. It is a
diagnostic, not an independently held-out metric or a semantic quality score.
No inference, network calls or source customer records are needed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = re.compile(r"rules:\d+:\d+(?:\.\d+)+")
BRACKET = re.compile(r"\[([^\]\n]{1,100})\]")
AUTHORITY_LIKE = re.compile(r"^(?:rules?[:_ -]|complaint\.|report\.|internal:)", re.I)
NUMBER = re.compile(r"\d+(?:[.-]\d+)*")


def markers(text):
    return [m for m in BRACKET.findall(text)
            if not CANONICAL.fullmatch(m) and (AUTHORITY_LIKE.match(m) or NUMBER.fullmatch(m))]


def summarize(path):
    source = path.read_bytes()
    rows = [json.loads(line) for line in source.decode("utf8").splitlines() if line.strip()]
    manifest = rows[0]
    summary = rows[-1]
    if manifest.get("record_type") != "manifest" or summary.get("record_type") != "summary":
        raise ValueError("A completed run with manifest and summary is required")
    triage = [r for r in rows if r.get("record_type") == "triage_case"]
    classified = [r for r in rows if r.get("record_type") == "classifier_case"]
    if len(triage) != manifest["triage_cases"] or len(classified) != manifest["classifier_cases"]:
        raise ValueError("Case count mismatch")
    suspects = []
    for row in triage:
        response = row["attempt"].get("response") or {}
        found = markers((response.get("draft_reply") or {}).get("text", ""))
        if found:
            suspects.append({"case_id": row["case"]["id"], "markers": found})
    suspect_ids = {r["case_id"] for r in suspects}
    required = [r for r in triage if r["case"]["required_citation_ids"]]

    def retrieval_hit(row, prefix):
        reasons = (row["attempt"].get("response") or {}).get("reasons", [])
        selected = next((r[len(prefix):].split(",") for r in reasons if r.startswith(prefix)), [])
        return set(row["case"]["required_citation_ids"]) <= set(selected)

    return {"source": str(path.relative_to(ROOT)), "source_sha256": hashlib.sha256(source).hexdigest(),
        "human_gold": False, "original_summary": summary,
        "triage_violations": dict(Counter(v for r in triage for v in r["assessment"]["violations"])),
        "retrieval": {"expected_rule_cases": len(required),
            "required_in_raw_top10": sum(retrieval_hit(r, "retrieved=") for r in required),
            "required_after_topic_gate": sum(retrieval_hit(r, "topic_gate_selected=") for r in required),
            "meaning": "authored expected ID coverage; not representative relevance/semantic accuracy"},
        "posthoc_expanded_marker_audit": {"added_after_baseline": True,
            "method": "noncanonical bracket content starting rule/rules/complaint/report/internal or entirely numeric",
            "case_count": len(suspects), "cases": suspects,
            "original_checks_and_expanded_marker_check_pass": sum(
                r["assessment"]["all_observable_constraints_pass"] and r["case"]["id"] not in suspect_ids for r in triage),
            "limits": "Observable authority-like markers only; not all hallucinations or semantic entailment."}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    boundary = (ROOT / ".private/quality-live").resolve()
    if not source.is_relative_to(boundary) or not output.is_relative_to(boundary) or source == output:
        raise ValueError("Use separate private quality-live input and output")
    report = summarize(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), "source_sha256": report["source_sha256"],
        "posthoc_authority_like_marker_cases": report["posthoc_expanded_marker_audit"]["case_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
