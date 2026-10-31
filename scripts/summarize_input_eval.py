"""Export only frozen synthetic questions, effective answers and aggregate metrics."""
import argparse
import hashlib
import json
from pathlib import Path

from build_metrics_report import sanitize_text
from evaluate_robustness import summarize, LABEL_KIND

ROOT = Path(__file__).resolve().parents[1]


def build(evaluation, cases_path):
    records = [json.loads(line) for line in evaluation.read_text(encoding="utf8").splitlines()]
    cases = {row["id"]: row for row in map(json.loads, cases_path.read_text(encoding="utf8").splitlines())}
    manifest = records[0]
    if (manifest.get("human_gold") is not False or manifest.get("label_kind") != LABEL_KIND
            or manifest.get("cases_sha256") != hashlib.sha256(cases_path.read_bytes()).hexdigest()
            or records[-1].get("record_type") != "summary"):
        raise ValueError("Invalid frozen synthetic provenance")
    rows = [row for row in records if row["record_type"] == "case"]
    if len(rows) != manifest["cases"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Missing or duplicate attempts")
    for row in rows:
        if (row.get("human_gold") is not False or not row["id"].startswith("robust-v4:")
                or row["id"] not in cases or row["request"] != cases[row["id"]]["request"]):
            raise ValueError("A row is not the frozen synthetic request")
    examples = []
    for language in ("ru", "kk", "uz", "en", "und"):
        items = [row for row in rows if row["expected_language"] == language]
        failures = [row for row in items if row["assessment"]["violations"]]
        chosen = (failures[:1] + [row for row in items if row not in failures[:1]])[:2]
        for row in chosen:
            response = row["response"]
            examples.append({"language": language,
                "question": sanitize_text("\n".join(m["text"] for m in row["request"]["messages"] if m["role"] == "player"))[0],
                "answer": sanitize_text(response.get("draft_reply", {}).get("text", ""))[0],
                "citations": response.get("draft_reply", {}).get("cited", []),
                "needs_human": response.get("needs_human"), "seconds": row["wall_seconds"],
                "violations": row["assessment"]["violations"]})
    return {"schema_version": 1, "human_gold": False, "label_kind": LABEL_KIND,
            "provenance": {"cases_sha256": manifest["cases_sha256"],
                "evaluation_sha256": hashlib.sha256(evaluation.read_bytes()).hexdigest(),
                "adapter_requested": manifest["adapter_requested"]},
            "summary": summarize(rows), "examples": examples,
            "limitations": ["synthetic_development_not_human_quality", "source_ID_is_not_entailment",
                "failed_attempts_in_denominators", "separate_generation_and_clarification_latency"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / ".private/input-v4/robustness-cases.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    private = (ROOT / ".private/input-v4").resolve()
    if any(not path.resolve().is_relative_to(private) for path in (args.evaluation, args.cases, args.output)) or args.output.exists():
        parser.error("Use private input-v4 paths and a new output")
    report = build(args.evaluation, args.cases)
    with args.output.open("x", encoding="utf8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({"cases": report["summary"]["overall"]["cases"], "examples": len(report["examples"])}))


if __name__ == "__main__":
    main()
