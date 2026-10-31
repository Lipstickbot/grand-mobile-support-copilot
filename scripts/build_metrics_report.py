"""Build a bounded, sanitized report of actual synthetic evaluation results.

No model, network or optional package is loaded. Every evaluated case, including
failed/skipped generations, stays in metric denominators. This is a development
constraint report, never human quality, classifier accuracy or production SLO.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("ru", "kk", "uz")
MAX_CASES = 2000
MAX_EXAMPLES = 12
MAX_TEXT = 4000
MAX_INPUT_BYTES = 64 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"synthetic-eval:[A-Za-z0-9:_-]{1,160}\Z")
_CITATION = re.compile(r"rules:\d+:\d+(?:\.\d+)+\Z")
_CITATION_IN_TEXT = re.compile(r"\brules:\d+:\d+(?:\.\d+)+")
_CHECKS = {"raw_generation_complete", "needs_human_policy_match", "effective_cited_list_supported_by_text"} | {
    prefix + "_" + suffix for prefix in ("raw", "effective") for suffix in (
        "language_heuristic_match", "citations_known", "citations_allowed", "forbidden_citations_absent",
        "required_citations_present", "internal_citations_absent", "unqualified_citations_absent", "guard_pass")}
_GUARD_REASONS = {
    "draft_empty_or_oversized", "draft_control_tokens", "draft_action_or_evidence_claim",
    "draft_unverified_verdict_or_evidence", "draft_secret_request", "draft_unknown_or_internal_citation",
    "draft_ambiguous_numbered_citation", "draft_language_unverified", "draft_language_unsupported"}
_HASH_FIELDS = ("evaluation_sha256", "corpus_sha256", "hf_adapter_sha256", "gguf_adapter_sha256", "gguf_model_sha256")
_REDACTIONS = (
    (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"), "[REDACTED_CREDENTIAL]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._~+/-]{8,}=*", re.I), "Bearer [REDACTED_CREDENTIAL]"),
    (re.compile(r"\b(?:api[_ -]?key|access[_ -]?token|secret|password|пароль|parol)\s*[:=]\s*[^\s,;]+", re.I),
     "[REDACTED_CREDENTIAL]"),
    # Conservative whole-line redaction also covers spaces in Windows usernames.
    (re.compile(r"\b[A-Za-z]:[\\/][^\r\n<>]*"), "[REDACTED_LOCAL_PATH]"),
    (re.compile(r"(?<![\w/:])/(?:Users|home|tmp|var|etc|mnt|root|private)(?:/[^\r\n<>]*)?"), "[REDACTED_LOCAL_PATH]"),
)


def _require(condition, description):
    if not condition:
        raise ValueError(description)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _ids(value):
    _require(isinstance(value, list) and len(value) <= 100 and
             all(isinstance(item, str) and _CITATION.fullmatch(item) for item in value), "Invalid citation identifiers")
    _require(len(value) == len(set(value)), "Duplicate citation identifiers")
    return set(value)


def sanitize_text(text):
    _require(isinstance(text, str), "Example text must be a string")
    redacted = text
    for pattern, replacement in _REDACTIONS:
        redacted = pattern.sub(replacement, redacted)
    redacted = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", redacted)
    changed = redacted != text
    truncated = len(redacted) > MAX_TEXT
    if truncated:
        redacted = redacted[:MAX_TEXT] + "… [TRUNCATED]"
    return redacted, changed, truncated


def _diagnostic_ids(reasons, prefix):
    matches = [value[len(prefix):] for value in reasons if value.startswith(prefix)]
    _require(len(matches) <= 1, "Ambiguous retrieval diagnostics")
    if not matches:
        return None
    values = matches[0].split(",") if matches[0] else []
    # Terms/section context may appear in raw retrieval. They are not official
    # rule citations and are never copied into the report.
    # Preserve raw positions: removing glossary/context hits would incorrectly
    # turn an official source ranked11 into a top10hit.
    return [value if _CITATION.fullmatch(value) else "" for value in values]


def _validate_case(row, mode):
    _require(row.get("human_gold") is False and isinstance(row.get("id"), str)
             and _ID.fullmatch(row["id"]), "Only explicitly synthetic non-gold cases are accepted")
    _require(row.get("mode") == mode and row.get("expected_language") in LANGUAGES, "Case mode/language mismatch")
    _require(_number(row.get("wall_seconds")), "Invalid attempt latency")
    required = _ids(row.get("required_citation_ids"))
    allowed = _ids(row.get("allowed_citation_ids"))
    forbidden = _ids(row.get("forbidden_citation_ids", []))
    _require(required <= allowed and not (allowed & forbidden), "Inconsistent citation constraints")
    assessment = row.get("assessment")
    _require(isinstance(assessment, dict) and assessment.get("human_gold") is False, "Missing non-gold assessment")
    checks = assessment.get("checks")
    _require(isinstance(checks, dict) and set(checks) == _CHECKS and
             all(type(value) is bool for value in checks.values()), "Unsupported assessment checks")
    violations = assessment.get("violations")
    _require(isinstance(violations, list) and set(violations) == {key for key, value in checks.items() if not value}
             and len(violations) == len(set(violations)), "Assessment violation mismatch")
    _require(assessment.get("all_observable_constraints_pass") is all(checks.values()) and
             assessment.get("citations_required") is bool(required), "Assessment totals mismatch")
    details = assessment.get("details")
    _require(isinstance(details, dict), "Missing assessment details")
    for prefix in ("raw", "effective"):
        for suffix in ("unknown_ids", "valid_but_disallowed_ids", "citation_ids"):
            _ids(details.get(prefix + "_" + suffix))
        guard = details.get(prefix + "_guard_reasons")
        _require(isinstance(guard, list) and set(guard) <= _GUARD_REASONS and
                 checks[prefix + "_guard_pass"] is (not bool(guard)), "Invalid guard assessment")
    generation = row.get("generation")
    _require(isinstance(generation, dict), "Missing generation metadata")
    complete = (generation.get("status") == "ok" and generation.get("finish_reason") == "eos"
                and not generation.get("error_code"))
    _require(checks["raw_generation_complete"] is complete, "Completion metadata mismatch")
    request = row.get("request")
    _require(isinstance(request, dict) and isinstance(request.get("messages"), list), "Missing request messages")
    player = [message.get("text") for message in request["messages"]
              if isinstance(message, dict) and message.get("role") == "player"]
    _require(player and all(isinstance(text, str) for text in player), "Missing synthetic player question")
    response = row.get("response")
    _require(isinstance(response, dict), "Invalid effective response")
    draft = response.get("draft_reply", {})
    _require(isinstance(draft, dict) and isinstance(draft.get("text", ""), str), "Invalid effective draft")
    _ids(draft.get("cited", []))
    category, priority = response.get("category"), response.get("priority")
    _require(category is None or (isinstance(category, str) and re.fullmatch(r"(?:complaint|report)\.[a-z_]{1,32}", category)),
             "Invalid category code")
    _require(priority is None or priority in {"p0", "p1", "p2", "p3"}, "Invalid priority code")
    _require(response.get("needs_human") is None or type(response["needs_human"]) is bool, "Invalid human routing")
    reasons = response.get("reasons", [])
    _require(isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons), "Invalid response diagnostics")
    retrieved = _diagnostic_ids(reasons, "retrieved=")
    selected = _diagnostic_ids(reasons, "topic_gate_selected=")
    return {"row": row, "checks": checks, "details": details, "required": required,
            "question": "\n".join(player), "retrieved": retrieved, "selected": selected,
            "fallback": "draft=localized_template_fallback" in reasons or bool(set(reasons) & _GUARD_REASONS)}


def _group(items):
    required = [item for item in items if item["required"]]
    retrieval = [item for item in required if item["retrieved"] is not None and item["selected"] is not None]
    latencies = sorted(item["row"]["wall_seconds"] for item in items)
    count = len(items)
    full = sum(item["checks"]["raw_generation_complete"] for item in items)
    return {
        "cases": count,
        "full_generations": full,
        "generation_errors_or_skips": count - full,
        "generation_attempts": sum(item["row"]["generation"].get("status") in {"ok", "error"} for item in items),
        "fatal_errors": sum(item["row"].get("fatal_error") is not None for item in items),
        "fallback_cases": sum(item["fallback"] for item in items),
        "all_observable_constraints_pass": sum(all(item["checks"].values()) for item in items),
        "citation_required_cases": len(required),
        "required_raw_citations_satisfied": sum(item["checks"]["raw_required_citations_present"] for item in required),
        "required_effective_citations_satisfied": sum(item["checks"]["effective_required_citations_present"] for item in required),
        "raw_guard_rejections": sum(not item["checks"]["raw_guard_pass"] for item in items),
        "effective_guard_rejections": sum(not item["checks"]["effective_guard_pass"] for item in items),
        "raw_language_heuristic_matches": sum(item["checks"]["raw_language_heuristic_match"] for item in items),
        "effective_language_heuristic_matches": sum(item["checks"]["effective_language_heuristic_match"] for item in items),
        "raw_unknown_citation_cases": sum(bool(item["details"]["raw_unknown_ids"]) for item in items),
        "effective_unknown_citation_cases": sum(bool(item["details"]["effective_unknown_ids"]) for item in items),
        "raw_wrong_scope_citation_cases": sum(bool(item["details"]["raw_valid_but_disallowed_ids"]) for item in items),
        "effective_wrong_scope_citation_cases": sum(bool(item["details"]["effective_valid_but_disallowed_ids"]) for item in items),
        "latency": {"n": count, "p50_seconds": statistics.median(latencies) if latencies else None,
                    "p95_seconds": latencies[math.ceil(.95 * count) - 1] if latencies else None,
                    "method": "nearest_rank_all_attempts_including_failures_small_development_sample"},
        "retrieval": {"citation_required_cases": len(required), "diagnostic_cases": len(retrieval),
                      "missing_diagnostic_cases": len(required) - len(retrieval),
                      "required_source_in_top10": sum(item["required"] <= set(item["retrieved"][:10]) for item in retrieval) if retrieval else None,
                      "required_source_selected": sum(item["required"] <= set(item["selected"]) for item in retrieval) if retrieval else None},
    }


def _example(item):
    row = item["row"]
    response = row["response"]
    question, question_redacted, question_truncated = sanitize_text(item["question"])
    answer, answer_redacted, answer_truncated = sanitize_text(response.get("draft_reply", {}).get("text", ""))
    return {"id": row["id"], "language": row["expected_language"], "question": question,
            "category": response.get("category"), "priority": response.get("priority"),
            "answer": answer, "citations": response.get("draft_reply", {}).get("cited", []),
            "needs_human": response.get("needs_human"), "elapsed_seconds": row["wall_seconds"],
            "violations": sorted(key for key, value in item["checks"].items() if not value),
            "fallback": item["fallback"], "text_redacted": question_redacted or answer_redacted,
            "text_truncated": question_truncated or answer_truncated}


def build_report(records, input_sha256):
    """Validate complete evaluator records and return an allowlisted public-to-local API object."""
    _require(isinstance(records, list) and 3 <= len(records) <= MAX_CASES + 3, "Invalid evaluation record count")
    manifest, summary = records[0], records[-1]
    _require(isinstance(manifest, dict) and manifest.get("record_type") == "manifest" and
             manifest.get("human_gold") is False and manifest.get("mode") in {"pipeline", "oracle"}, "Invalid synthetic manifest")
    _require(type(manifest.get("cases")) is int and 1 <= manifest["cases"] <= MAX_CASES, "Invalid manifest case count")
    _require(isinstance(summary, dict) and summary.get("record_type") == "summary" and summary.get("human_gold") is False,
             "Evaluation must end with a non-gold summary; incomplete runs are not reports")
    _require(isinstance(input_sha256, str) and _SHA.fullmatch(input_sha256), "Invalid input report hash")
    provenance = {"input_report_sha256": input_sha256}
    for key in _HASH_FIELDS:
        _require(isinstance(manifest.get(key), str) and _SHA.fullmatch(manifest[key]), "Invalid provenance hash")
        provenance[key] = manifest[key]
    code = manifest.get("code_sha256")
    _require(isinstance(code, dict) and code and all(isinstance(key, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,80}\.py", key)
             and isinstance(value, str) and _SHA.fullmatch(value) for key, value in code.items()), "Invalid inference code hashes")
    provenance["code_sha256"] = dict(code)
    if "adapter_requested" in manifest:
        _require(type(manifest["adapter_requested"]) is bool, "Invalid adapter selection flag")
        provenance["adapter_requested"] = manifest["adapter_requested"]
    if manifest.get("classifier_enabled") is not None:
        _require(type(manifest["classifier_enabled"]) is bool, "Invalid classifier flag")
        provenance["classifier_enabled"] = manifest["classifier_enabled"]
        if manifest["classifier_enabled"]:
            _require(isinstance(manifest.get("classifier_manifest_sha256"), str) and
                     _SHA.fullmatch(manifest["classifier_manifest_sha256"]), "Invalid classifier model hash")
            provenance["classifier_manifest_sha256"] = manifest["classifier_manifest_sha256"]
    middle = records[1:-1]
    _require(all(isinstance(row, dict) and row.get("record_type") in {"runtime", "case"} for row in middle),
             "Unexpected or fatal evaluation record")
    runtime = [row for row in middle if row["record_type"] == "runtime"]
    _require(len(runtime) == 1 and middle[0]["record_type"] == "runtime", "Expected one leading runtime record")
    cases = [row for row in middle if row["record_type"] == "case"]
    _require(len(cases) == manifest["cases"], "Case count mismatch; failed attempts must not be omitted")
    _require(all(isinstance(row.get("id"), str) for row in cases), "Invalid case identifier")
    _require(len({row["id"] for row in cases}) == len(cases), "Duplicate evaluation case")
    items = [_validate_case(row, manifest["mode"]) for row in cases]
    _require(summary.get("overall", {}).get("cases") == len(items), "Summary case count mismatch")
    # Examples deliberately include a failure and a success per language when
    # available, then original order. Counts always use ALL cases, never this view.
    groups = {}
    for language in LANGUAGES:
        group = [item for item in items if item["row"]["expected_language"] == language]
        first_failure = next((item for item in group if not all(item["checks"].values())), None)
        first_success = next((item for item in group if all(item["checks"].values())), None)
        front = [item for item in (first_failure, first_success) if item is not None]
        groups[language] = deque(front + [item for item in group if all(item is not chosen for chosen in front)])
    examples = []
    while len(examples) < MAX_EXAMPLES and any(groups.values()):
        for language in LANGUAGES:
            if groups[language] and len(examples) < MAX_EXAMPLES:
                examples.append(_example(groups[language].popleft()))
    return {
        "schema_version": 1, "label_kind": "synthetic_development_not_human_gold", "human_gold": False,
        "mode": manifest["mode"], "provenance": provenance,
        "metrics": {"overall": _group(items), "by_language": {
            language: _group([item for item in items if item["row"]["expected_language"] == language])
            for language in LANGUAGES if any(item["row"]["expected_language"] == language for item in items)}},
        "examples": examples,
        "example_selection": "language_round_robin_first_failure_and_success_then_original_order_not_representative",
        "example_text_policy": "player_and_effective_answer_only_known_credentials_local_paths_redacted_max4000chars",
        "not_measured": ["human_language_quality", "semantic_entailment", "classifier_accuracy", "gold_category_priority_F1",
                         "confidence_calibration", "production_latency_SLO", "automation_quality"],
        "limitations": ["AI_authored_silver_scenarios_used_for_development", "source_ID_presence_is_not_entailment",
                        "language_and_guards_are_heuristic", "full_generation_is_not_correct_answer",
                        "recognizable_secret_redaction_is_not_a_general_PII_detector", "all_cases_require_human_review"],
    }


def write_report(evaluation, output):
    private = (ROOT / ".private").resolve()
    evaluation, output = Path(evaluation).resolve(), Path(output).resolve()
    _require(evaluation.is_relative_to(private) and evaluation.suffix == ".jsonl", "Input must be private JSONL")
    _require(output.is_relative_to(private) and output.suffix == ".json" and not output.exists(), "Use a new private JSON output")
    protected = [private / name for name in ("models", "soup-runs", "soup-layer-cache", "index", "runtime", "tools")]
    _require(not any(output.is_relative_to(path) for path in protected), "Report may not be written into model/runtime/cache artifacts")
    _require(evaluation.stat().st_size <= MAX_INPUT_BYTES, "Evaluation file exceeds64MiB")
    raw = evaluation.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf8").splitlines() if line.strip()]
    report = build_report(records, hashlib.sha256(raw).hexdigest())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = write_report(args.evaluation, args.output)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        # Do not copy exception text that might contain a path or private input.
        print("Metrics report failed: " + type(exc).__name__ + ". Check private evaluation contract.", file=sys.stderr)
        return 1
    print(json.dumps({"status": "written", "cases": report["metrics"]["overall"]["cases"],
                      "examples": len(report["examples"]), "human_gold": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
