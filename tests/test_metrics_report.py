"""CPU-only reporting checks using invented records, no model or real ticket data."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("metrics_builder", Path(__file__).parents[1] / "scripts/build_metrics_report.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)

SHA = "a" * 64
RULE = "rules:900:2.1"


def case(index=0, language="ru", required=True, complete=True, fallback=False):
    checks = {name: True for name in builder._CHECKS}
    checks["raw_generation_complete"] = complete
    checks["raw_required_citations_present"] = complete or not required
    checks["effective_required_citations_present"] = complete or not required
    violations = [name for name, value in checks.items() if not value]
    details = {}
    for prefix in ("raw", "effective"):
        details.update({prefix + "_unknown_ids": [], prefix + "_valid_but_disallowed_ids": [],
                        prefix + "_citation_ids": [RULE] if complete else [], prefix + "_guard_reasons": []})
    return {"record_type": "case", "id": f"synthetic-eval:fixture{index}:{language}", "mode": "pipeline",
            "human_gold": False, "expected_language": language, "wall_seconds": index + 1,
            "required_citation_ids": [RULE] if required else [], "allowed_citation_ids": [RULE] if required else [],
            "forbidden_citation_ids": [], "fatal_error": None,
            "request": {"messages": [{"role": "player", "text": "Synthetic player question"},
                                     {"role": "admin", "text": "PRIVATE_ADMIN_MUST_NOT_APPEAR"}]},
            "response": {"category": "report.question", "priority": "p3", "needs_human": True,
                         "draft_reply": {"text": "Synthetic answer", "cited": [RULE] if complete else []},
                         "reasons": ["retrieved=" + RULE, "topic_gate_selected=" + (RULE if complete else "")] +
                                    (["draft=localized_template_fallback"] if fallback else [])},
            "generation": {"status": "ok" if complete else "error", "finish_reason": "eos" if complete else "max_tokens",
                           "prompt_messages": [{"role": "system", "content": "PRIVATE_PROMPT"}],
                           "raw_text": "PRIVATE_RAW_GENERATION", "model_path": "C:/PRIVATE_MODEL"},
            "assessment": {"human_gold": False, "checks": checks, "violations": violations,
                           "all_observable_constraints_pass": all(checks.values()), "citations_required": required,
                           "details": details}}


def records(cases=None):
    cases = cases if cases is not None else [case()]
    manifest = {"record_type": "manifest", "human_gold": False, "mode": "pipeline", "cases": len(cases),
                "code_sha256": {"engine.py": SHA}, "private": "PRIVATE_MANIFEST"}
    manifest.update({name: SHA for name in builder._HASH_FIELDS})
    return [manifest, {"record_type": "runtime", "health": {"path": "PRIVATE_RUNTIME"}}, *cases,
            {"record_type": "summary", "human_gold": False,
             "overall": {"cases": len(cases), "full_eos": 999, "private": "PRIVATE_SUMMARY"}}]


def test_all_attempts_and_required_citation_denominators_are_preserved():
    report = builder.build_report(records([case(0), case(1, complete=False, fallback=True), case(2, required=False)]), SHA)
    metrics = report["metrics"]["overall"]
    assert metrics["cases"] == metrics["latency"]["n"] == 3
    assert metrics["full_generations"] == 2 and metrics["generation_errors_or_skips"] == 1
    assert metrics["fallback_cases"] == 1
    assert metrics["citation_required_cases"] == 2
    assert metrics["required_raw_citations_satisfied"] == metrics["required_effective_citations_satisfied"] == 1
    assert metrics["latency"]["p50_seconds"] == 2 and metrics["latency"]["p95_seconds"] == 3
    assert metrics["retrieval"]["required_source_in_top10"] == 2
    assert metrics["retrieval"]["required_source_selected"] == 1


def test_allowlist_does_not_copy_prompt_admin_raw_runtime_paths_or_forged_summary_counts():
    report = builder.build_report(records(), SHA)
    assert "PRIVATE_" not in json.dumps(report)
    assert report["metrics"]["overall"]["full_generations"] == 1
    assert report["schema_version"] == 1 and report["human_gold"] is False
    assert "classifier_accuracy" in report["not_measured"]


def test_examples_are_bounded_balanced_and_show_available_failures():
    cases = [case(index, language, complete=index != 19) for language in builder.LANGUAGES for index in range(20)]
    report = builder.build_report(records(cases), SHA)
    examples = report["examples"]
    assert len(examples) == 12
    for language in builder.LANGUAGES:
        subset = [example for example in examples if example["language"] == language]
        assert len(subset) == 4
        assert any(example["violations"] for example in subset)
        assert any(not example["violations"] for example in subset)
    assert report["metrics"]["overall"]["cases"] == 60


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.pop(-2),
    lambda rows: rows.pop(),
    lambda rows: rows.insert(-1, deepcopy(rows[2])),
    lambda rows: rows[2].update(id="historical-ticket:123"),
    lambda rows: rows[2].update(human_gold=True),
    lambda rows: rows[2].update(wall_seconds=float("nan")),
    lambda rows: rows[2]["generation"].update(status="error"),
    lambda rows: rows[2]["assessment"].update(violations=[] ,all_observable_constraints_pass=False),
])
def test_incomplete_duplicate_non_synthetic_and_inconsistent_reports_reject(mutation):
    rows = records()
    mutation(rows)
    with pytest.raises(ValueError):
        builder.build_report(rows, SHA)


def test_recognizable_credentials_and_local_paths_in_example_text_are_redacted():
    row = case()
    row["request"]["messages"][0]["text"] = "api_key=SUPERSENSITIVE\nC:\\Users\\Private Name\\secret.txt"
    row["response"]["draft_reply"]["text"] = "Bearer ABCDEFGHIJKLMNOP\n/home/person/private.txt"
    report = builder.build_report(records([row]), SHA)
    text = json.dumps(report)
    assert all(secret not in text for secret in ("SUPERSENSITIVE", "Private Name", "ABCDEFGHIJKLMNOP", "/home/person"))
    assert report["examples"][0]["text_redacted"]


def test_example_text_is_bounded_without_removing_attempt_from_metrics():
    row = case()
    row["request"]["messages"][0]["text"] = "x" * 8000
    result = builder.build_report(records([row]), SHA)
    assert len(result["examples"][0]["question"]) < 4050
    assert result["examples"][0]["text_truncated"]
    assert result["metrics"]["overall"]["cases"] == 1


def test_oracle_missing_retrieval_diagnostics_are_unknown_not_fabricated_zero():
    rows = records()
    rows[0]["mode"] = rows[2]["mode"] = "oracle"
    rows[2]["response"]["reasons"] = []
    metrics = builder.build_report(rows, SHA)["metrics"]["overall"]
    assert metrics["retrieval"]["diagnostic_cases"] == 0
    assert metrics["retrieval"]["required_source_in_top10"] is None
    assert metrics["retrieval"]["missing_diagnostic_cases"] == 1


def test_top10_positions_include_non_rule_retrieval_hits():
    row = case()
    row["response"]["reasons"] = ["retrieved=" + ",".join(["internal:900:term"] * 10 + [RULE]),
                                  "topic_gate_selected=" + RULE]
    metrics = builder.build_report(records([row]), SHA)["metrics"]["overall"]
    assert metrics["retrieval"]["required_source_in_top10"] == 0
    assert metrics["retrieval"]["required_source_selected"] == 1


def test_write_refuses_existing_output_and_paths_outside_private(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    private = tmp_path / ".private"
    private.mkdir()
    source = private / "evaluation.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in records()), encoding="utf8")
    output = private / "report.json"
    builder.write_report(source, output)
    original = output.read_bytes()
    with pytest.raises(ValueError):
        builder.write_report(source, output)
    assert output.read_bytes() == original
    with pytest.raises(ValueError):
        builder.write_report(source, tmp_path / "public.json")
    with pytest.raises(ValueError):
        builder.write_report(source, private / "models/new.json")
