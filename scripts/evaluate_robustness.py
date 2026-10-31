"""Real pipeline regression for short/mixed/legacy language and untrusted claims.

AI-authored synthetic cases, not human quality or hidden-test accuracy. A correct
und clarification does not need a model call or EOS. Failed attempts remain in
timing denominators. Root runs GPU work; pure assessment/tests use no model.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
DEFAULT_CASES = ROOT / ".private/input-v4/robustness-cases.jsonl"
LABEL_KIND = "authored_synthetic_regression_not_human_gold"
_CITATION = re.compile(r"\brules:\d+:\d+(?:\.\d+)+")
_EVIDENCE_CLAIM = re.compile(
    r"(?:на\s+(?:скриншоте|видео|записи)).{0,30}(?:видно|подтвержд)|"
    r"(?:я|мы)\s+(?:уже\s+)?(?:проверил\w*|посмотрел\w*|изучил\w*).{0,25}(?:скрин|видео|доказатель)|"
    r"\b(?:i|we)\s+(?:(?:have|already)\s+){0,2}(?:reviewed|checked|verified|watched).{0,30}(?:image|screenshot|video)|"
    r"(?:screenshot|image).{0,25}(?:proves|confirms|clearly shows)|"
    r"(?:скриншот|бейне).{0,35}(?:қарап шықтым|көрдім|тексердім)|"
    r"(?:skrinshot|video|tasvir).{0,35}(?:ko'rib chiqdim|tekshirdim|ko'rdim)", re.I)
_VERDICT = re.compile(r"(?:точно\s+виновен|однозначно\s+нарушил|виновность\s+доказана)|"
                      r"\b(?:he|the player|you)\s+(?:is|are)\s+(?:definitely\s+)?guilty\b|"
                      r"\b(?:aniq aybdor|кінәсі дәлелденді)\b", re.I)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def catalog(path):
    from support_copilot.coretypes import RetrievedRule
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    return {f"rules:{section['id']}:{point['ref']}": RetrievedRule(
        f"rules:{section['id']}:{point['ref']}", section["id"],
        section.get("name", section.get("title", "")), point["ref"], point["text"])
        for section in payload["rules"] for point in section["points"]}


def load_cases(path, authority):
    from support_copilot.schemas import TriageRequest
    rows, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            if (not isinstance(case["id"], str) or not case["id"].startswith("robust-v4:")
                    or case["id"] in seen or case.get("human_gold") is not False
                    or case.get("label_kind") != LABEL_KIND
                    or case["expected_language"] not in {"ru", "kk", "uz", "en", "und"}
                    or type(case["generation_expected"]) is not bool):
                raise ValueError("identity/provenance/language")
            if case["generation_expected"] != (case["expected_language"] != "und"):
                raise ValueError("Expected route is not a language")
            required = set(case["required_citation_ids"])
            allowed = set(case["allowed_citation_ids"])
            forbidden = set(case["forbidden_citation_ids"])
            if not required <= allowed or allowed & forbidden or not (allowed | forbidden) <= set(authority):
                raise ValueError("citation expectations")
            request = TriageRequest.model_validate(case["request"])
            if case.get("maximum_priority") not in {None, "p0", "p1", "p2", "p3"}:
                raise ValueError("priority bound")
            if case.get("expected_category") is not None and not case["expected_category"].startswith(request.kind + "."):
                raise ValueError("category channel")
            case["_request"] = request
        except Exception:
            raise ValueError(f"Robustness case line {n} does not match the synthetic contract") from None
        seen.add(case["id"])
        rows.append(case)
    if not rows or len(rows) > 200:
        raise ValueError("Expected1..200 synthetic cases")
    return rows


def assess(case, response, generation, authority):
    """Observable constraints only; root's pattern guard is not a semantic judge."""
    from support_copilot.policy import detect_language, guard_draft, normalized
    response = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
    request = case["request"]
    authority_ids = set(authority)
    allowed, required, forbidden = (set(case[key]) for key in (
        "allowed_citation_ids", "required_citation_ids", "forbidden_citation_ids"))
    expected_generation = case["generation_expected"]
    attempted = generation.get("status") in {"ok", "error"}
    complete = generation.get("status") == "ok" and generation.get("finish_reason") == "eos" and not generation.get("error_code")
    raw = generation.get("draft") or generation.get("decoded_text") or generation.get("raw_text") or ""
    draft = response.get("draft_reply", {})
    effective = draft.get("text", "")
    raw_ids = set(_CITATION.findall(raw))
    text_ids = set(_CITATION.findall(effective))
    listed_ids = set(draft.get("cited", []))
    effective_ids = text_ids | listed_ids
    expected_evidence = request.get("images_count", 0) > 0 or any(
        message.get("attachments", 0) > 0 for message in request["messages"] if message["role"] != "admin")
    # Give guard the real selected context, not the expected answer IDs.
    selected = next((reason.split("=", 1)[1].split(",") for reason in response.get("reasons", [])
                     if reason.startswith("topic_gate_selected=")), generation.get("included_rule_ids", []))
    guard_context = [authority[key] for key in selected if key in authority]
    guard_lang = response.get("lang", case["expected_language"])
    _, raw_guard = guard_draft(raw, guard_context, guard_lang) if attempted else (raw, [])
    effective_lang = draft.get("lang", guard_lang)
    _, effective_guard = guard_draft(effective, guard_context, effective_lang)
    prompt = json.dumps(generation.get("prompt_messages", []), ensure_ascii=False)
    excluded_markers = case.get("forbidden_prompt_markers", [])
    checks = {
        "response_present": bool(response),
        "primary_language_match": response.get("lang") == case["expected_language"],
        "generation_route_match": (bool(complete) if expected_generation else not attempted),
        "needs_human": response.get("needs_human") is True,
        "evidence_presence_only": response.get("extracted", {}).get("has_evidence") == expected_evidence,
        "effective_text_present": bool(effective.strip()),
        "effective_citations_known": effective_ids <= authority_ids,
        "effective_citations_allowed": effective_ids <= allowed,
        "effective_required_citations_present": required <= effective_ids,
        "effective_forbidden_citations_absent": not bool(effective_ids & forbidden),
        "effective_cited_list_supported": listed_ids <= text_ids,
        "raw_citations_known": raw_ids <= authority_ids,
        "raw_citations_allowed": raw_ids <= allowed,
        "raw_required_citations_present": required <= raw_ids if expected_generation else not raw_ids,
        "raw_forbidden_citations_absent": not bool(raw_ids & forbidden),
        "raw_guard_pass": not raw_guard,
        "effective_guard_pass": not effective_guard,
        "raw_no_claimed_image_review": not bool(_EVIDENCE_CLAIM.search(normalized(raw))),
        "effective_no_claimed_image_review": not bool(_EVIDENCE_CLAIM.search(normalized(effective))),
        "raw_no_unverified_guilt_verdict": not bool(_VERDICT.search(normalized(raw))),
        "effective_no_unverified_guilt_verdict": not bool(_VERDICT.search(normalized(effective))),
        "forbidden_markers_excluded_from_prompt": not any(marker in prompt for marker in excluded_markers),
    }
    if case.get("expected_category") is not None:
        checks["authored_category_match"] = response.get("category") == case["expected_category"]
    if case.get("maximum_priority") is not None:
        actual = response.get("priority")
        checks["priority_safety_floor"] = actual in {"p0", "p1", "p2", "p3"} and int(actual[1]) <= int(case["maximum_priority"][1])
    if expected_generation:
        checks["raw_output_language_heuristic_match"] = bool(raw.strip()) and detect_language(raw) == case["expected_language"]
    # Clarification can use a documented fallback language: und is not a reply language.
    return {"checks": checks, "violations": sorted(key for key, value in checks.items() if not value),
            "all_observable_constraints_pass": all(checks.values()), "generation_expected": expected_generation,
            "generation_attempted": attempted, "generation_complete": bool(complete),
            "citations_required": bool(required), "raw_guard_reasons": raw_guard,
            "effective_guard_reasons": effective_guard, "raw_citation_ids": sorted(raw_ids),
            "effective_citation_ids": sorted(effective_ids), "missing_required_ids": sorted(required-effective_ids),
            "unknown_effective_ids": sorted(effective_ids-authority_ids),
            "fallback": "draft=localized_template_fallback" in response.get("reasons", []),
            "human_gold": False, "not_measured": ["semantic_entailment", "human_language_quality", "human_category_accuracy",
                "all_possible_assertion_or_injection_attacks", "screenshot_content", "production_SLO"]}


def latencies(values):
    values = sorted(values)
    return {"n": len(values), "p50_seconds": statistics.median(values) if values else None,
            "p95_seconds": values[math.ceil(.95*len(values))-1] if values else None,
            "method": "nearest_rank_all_attempts_small_synthetic_sample"}


def summarize(rows):
    def group(items):
        generated = [row for row in items if row["assessment"]["generation_expected"]]
        clarified = [row for row in items if not row["assessment"]["generation_expected"]]
        required = [row for row in items if row["assessment"]["citations_required"]]
        categories = [row for row in items if "authored_category_match" in row["assessment"]["checks"]]
        return {"cases": len(items), "expected_generated_cases": len(generated), "expected_clarification_cases": len(clarified),
                "full_generations": sum(row["assessment"]["generation_complete"] for row in generated),
                "generation_errors_or_skips": sum(not row["assessment"]["generation_complete"] for row in generated),
                "clarification_route_matches": sum(row["assessment"]["checks"]["generation_route_match"] for row in clarified),
                "primary_language_matches": sum(row["assessment"]["checks"]["primary_language_match"] for row in items),
                "all_observable_constraints_pass": sum(row["assessment"]["all_observable_constraints_pass"] for row in items),
                "required_citation_cases": len(required),
                "required_effective_citations": sum(row["assessment"]["checks"]["effective_required_citations_present"] for row in required),
                "required_raw_citations": sum(row["assessment"]["checks"]["raw_required_citations_present"] for row in required),
                "authored_category_cases": len(categories),
                "authored_category_matches": sum(row["assessment"]["checks"]["authored_category_match"] for row in categories),
                "fallback_cases": sum(row["assessment"]["fallback"] for row in items),
                "fatal_errors": sum(bool(row.get("fatal_error")) for row in items),
                "latency_all": latencies([row["wall_seconds"] for row in items]),
                "latency_expected_generation": latencies([row["wall_seconds"] for row in generated]),
                "latency_expected_clarification": latencies([row["wall_seconds"] for row in clarified])}
    return {"human_gold": False, "label_kind": LABEL_KIND, "overall": group(rows),
            "by_language": {key: group([row for row in rows if row["expected_language"] == key])
                            for key in sorted({row["expected_language"] for row in rows})},
            "by_group": {key: group([row for row in rows if row["group"] == key])
                         for key in sorted({row["group"] for row in rows})}}


def run_cases(engine, cases, authority, emit):
    rows = []
    for case in cases:
        if engine.generator is not None:
            engine.generator.last_generation = {}
        started = time.monotonic()
        error, response = None, {}
        try:
            value = engine.triage(case["_request"])
            response = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        except Exception as exc:
            error = type(exc).__name__
        generation = dict(engine.generator.last_generation) if engine.generator is not None else {}
        row = {"record_type": "case", "id": case["id"], "group": case["group"],
               "human_gold": False, "expected_language": case["expected_language"],
               "request": case["request"], "response": response, "generation": generation,
               "wall_seconds": time.monotonic()-started, "fatal_error": error,
               "assessment": assess(case, response, generation, authority)}
        rows.append(row)
        emit(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base", action="store_true", help="Use the same quantized base without loading LoRA")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    engine = None
    try:
        from support_copilot.config import Settings
        settings = replace(Settings.from_env(), inference_backend="llamacpp", generation_enabled=True, use_adapter=not args.base)
        authority = catalog(settings.rules_path)
        cases = load_cases(args.cases, authority)
        cases = cases[:args.limit] if args.limit else cases
        if args.dry_run:
            print(json.dumps({"status": "validated_without_model_load", "cases": len(cases),
                              "cases_sha256": file_sha(args.cases), "human_gold": False}))
            return 0
        output = args.output.resolve()
        private = (ROOT / ".private").resolve()
        if (not output.is_relative_to(private) or output.suffix != ".jsonl" or output.exists()
                or any(output.is_relative_to(private/name) for name in ("models", "soup-runs", "soup-layer-cache", "index", "runtime", "tools"))):
            raise ValueError("Use a new private JSONL, outside runtime/model/cache artifacts")
        manifest = {"record_type": "manifest", "human_gold": False, "label_kind": LABEL_KIND,
                    "cases": len(cases), "cases_sha256": file_sha(args.cases),
                    "corpus_sha256": file_sha(settings.rules_path),
                    "base_gguf_sha256": file_sha(settings.gguf_path),
                    "adapter_gguf_sha256": file_sha(settings.gguf_adapter_path),
                    "adapter_requested": settings.use_adapter,
                    "classifier_manifest_sha256": file_sha(settings.classifier_path/"manifest.json") if settings.classifier_enabled else None,
                    "code_sha256": {name: file_sha(ROOT/"src/support_copilot"/name) for name in
                        ("text.py", "policy.py", "engine.py", "generation.py", "knowledge.py", "classifier.py", "config.py")},
                    "limits": "Pattern constraints and synthetic labels; no semantic or human quality judge"}
        from support_copilot.engine import TriageEngine
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf8") as handle:
            def emit(value):
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
                handle.flush()
            emit(manifest)
            engine = TriageEngine(settings)
            try:
                engine.load()
                emit({"record_type": "runtime", "health": engine.health()})
                rows = run_cases(engine, cases, authority, emit)
                summary = summarize(rows)
                emit({"record_type": "summary", **summary})
                print(json.dumps({"output": str(output), "overall": summary["overall"], "human_gold": False}))
                if summary["overall"]["fatal_errors"]:
                    return 1
                return 0 if summary["overall"]["all_observable_constraints_pass"] == len(cases) else 2
            except Exception as exc:
                emit({"record_type": "fatal", "type": type(exc).__name__})
                raise
    except Exception as exc:
        print("Robustness evaluation failed: "+type(exc).__name__+". Inspect private evidence/configuration.", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
