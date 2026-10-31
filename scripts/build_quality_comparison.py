"""Publish only validated aggregates from completed, preserved local HTTP runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.quality_report import PublicQualityReport
from summarize_http_quality import markers


def read_run(path):
    path = path.resolve()
    if not path.is_relative_to((ROOT / ".private/quality-live").resolve()):
        raise ValueError("Expected a private quality-live run")
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf8").splitlines() if line.strip()]
    manifest, summary = rows[0], rows[-1]
    triage = [r for r in rows if r.get("record_type") == "triage_case"]
    if (manifest.get("record_type") != "manifest" or summary.get("record_type") != "summary"
            or len(triage) != manifest["triage_cases"] or manifest.get("human_gold") is not False
            or not all(summary[k] for k in ("health_ready_after", "health_identity_stable", "code_unchanged", "fixtures_unchanged"))):
        raise ValueError("Incomplete or drifting run")
    return {"manifest": manifest, "summary": summary, "triage": triage,
            "sha256": hashlib.sha256(raw).hexdigest(), "source": str(path.relative_to(ROOT))}


def response(row):
    return row["attempt"].get("response") or {}


def pseudo_count(rows):
    return sum(bool(markers((response(r).get("draft_reply") or {}).get("text", ""))) for r in rows)


def aggregate(run):
    summary, rows = run["summary"], run["triage"]
    triage, classifier = summary["triage"]["overall"], summary["classifier"]
    return {"cases": len(rows), "category_correct": classifier["category"]["correct"],
        "category_cases": classifier["category"]["n"],
        "category_macro_f1": classifier["category"]["macro_f1_supported_classes"],
        "priority_macro_f1": classifier["priority"]["macro_f1_supported_classes"],
        "p0_recall": classifier["priority"]["per_class"]["p0"]["recall"],
        "required_citation_hits": triage["required_citations_satisfied"],
        "required_citation_cases": triage["citation_required_cases"],
        "pseudo_citation_cases": pseudo_count(rows),
        "intake_template_cases": sum("draft=source_absent_intake_template" in response(r).get("reasons", []) for r in rows),
        "fallback_cases": triage["fallback_cases"],
        "latency_p50_seconds": triage["latency_all"]["p50_seconds"],
        "latency_p95_seconds": triage["latency_all"]["p95_seconds"]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--followup", type=Path)
    p.add_argument("--replace", action="store_true", help="Archive previous derived aggregates by hash before atomic replacement; raw runs stay untouched")
    args = p.parse_args()
    before, after = read_run(args.before), read_run(args.after)
    if (before["manifest"]["fixture_sha256"] != after["manifest"]["fixture_sha256"]
            or before["manifest"]["server_identity"] != after["manifest"]["server_identity"]
            or [r["case"] for r in before["triage"]] != [r["case"] for r in after["triage"]]):
        raise ValueError("Comparison needs identical cases/order and model settings; failures stay in denominators")
    report = {"schema_version": 1, "label_kind": "authored_synthetic_quality_comparison_not_human_gold",
        "human_gold": False, "status": "complete", "comparison": {"before": aggregate(before), "after": aggregate(after)},
        "limitations": ["synthetic_not_human_gold", "no_native_language_review", "templates_are_not_model_training",
            "same_cases_used_for_development", "semantic_accuracy_unmeasured", "single_sequential_run_not_SLO"]}
    provenance = {"before": {k: before[k] for k in ("source", "sha256")},
                  "after": {k: after[k] for k in ("source", "sha256")}}
    if args.followup:
        extra = read_run(args.followup)
        if extra["manifest"]["code_sha256"] != after["manifest"]["code_sha256"]:
            raise ValueError("Followup used different serving code")
        rows = [r for r in extra["triage"] if r["case"]["dataset"] == "fresh_challenge"]
        critical = [r for r in rows if r["case"].get("maximum_priority") == "p0"]
        required = [r for r in rows if r["case"]["required_citation_ids"]]
        report["extra_stress"] = {"cases": len(rows), "required_citation_hits": sum(r["assessment"]["checks"]["required_citations_present"] for r in required),
            "required_citation_cases": len(required), "pseudo_citation_cases": pseudo_count(rows),
            "critical_correct": sum(response(r).get("priority") == "p0" and response(r).get("category") == r["case"]["expected_category"] for r in critical),
            "critical_cases": len(critical), "language_matches": sum(r["assessment"]["checks"]["primary_language_match"] for r in rows)}
        provenance["followup"] = {k: extra[k] for k in ("source", "sha256")}
    safe = PublicQualityReport.model_validate(report).model_dump(mode="json")
    directory = ROOT / ".private/quality-v5"
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("dashboard-quality.json", safe), ("dashboard-provenance.json", provenance)):
        target = directory / name
        text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if target.exists():
            if not args.replace:
                raise FileExistsError("Derived report exists; use --replace to archive it first")
            old = target.read_bytes()
            history = directory / "report-history"
            history.mkdir(exist_ok=True)
            archive = history / (target.stem + "-" + hashlib.sha256(old).hexdigest() + ".json")
            if archive.exists():
                if archive.read_bytes() != old:
                    raise ValueError("History content mismatch")
            else:
                archive.write_bytes(old)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".json.tmp", delete=False) as tmp:
                tmp.write(text.encode("utf8"))
            Path(tmp.name).replace(target)
        else:
            with target.open("x", encoding="utf8") as handle:
                handle.write(text)
    print(json.dumps(safe, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
