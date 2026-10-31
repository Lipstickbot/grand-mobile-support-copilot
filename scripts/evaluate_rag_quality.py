"""Evaluate observable constraints on AI-authored silver scenarios, never human quality.

Pipeline mode exercises real retrieval/classification/generation/guards. Oracle mode
isolates generation using scenario-selected IDs resolved from the authoritative
catalog; supplied passage text and reference drafts are never given to the model.
All prompts/raw outputs remain in a new private JSONL. Exit 0: all constraints pass;
2: completed with constraint failures; 1: setup/provider/integration failure.
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
DEFAULT_EVALUATION = ROOT / ".private/rag-training-v2/prepared-reviewed/evaluation.jsonl"
_CITATION = re.compile(r"\brules:\d+:\d+(?:\.\d+)+")
_INTERNAL = re.compile(r"\binternal:[\w:.-]+")
_UNQUALIFIED = re.compile(r"(?:пункт\w*|правил\w*|п\.|rule\w*|тармақ\w*|band\w*)\s*(?:№\s*)?\d+(?:\.\d+)+|\[\d+(?:\.\d+)+\]", re.I)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_dict(value):
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value or {}


def evaluate_case(case, response, rawmeta, authoritative_ids):
    """Pure observable checks; ID existence is not semantic entailment.

    Raw and effective outputs are scored separately so a safe fallback cannot hide
    a missing citation or a failed generation. Needs-human is only policy routing.
    """
    from support_copilot.coretypes import RetrievedRule
    from support_copilot.policy import detect_language, guard_draft

    response = _as_dict(response)
    authority = set(authoritative_ids)
    allowed = set(case.get("allowed_citation_ids", []))
    required = set(case.get("required_citation_ids", allowed))
    forbidden = set(case.get("forbidden_citation_ids", []))
    expected_lang = case["expected_language"]
    # Identifier-only records suffice for the existing nonsemantic guard; these
    # empty bodies are never retrieval context or model input.
    guard_rules = [RetrievedRule(key, int(key.split(":")[1]), "", key.split(":")[2], "")
                   for key in sorted(authority)]
    raw_text = rawmeta.get("draft") or rawmeta.get("decoded_text") or rawmeta.get("raw_text") or ""
    effective = response.get("draft_reply", {})
    effective_text = effective.get("text", "")
    raw_ids = set(_CITATION.findall(raw_text))
    effective_text_ids = set(_CITATION.findall(effective_text))
    listed_ids = set(effective.get("cited", []))
    effective_ids = effective_text_ids | listed_ids
    _, raw_guard = guard_draft(raw_text, guard_rules, expected_lang)
    _, effective_guard = guard_draft(effective_text, guard_rules, expected_lang)
    complete = (rawmeta.get("status") == "ok" and rawmeta.get("finish_reason") == "eos"
                and not rawmeta.get("error_code"))
    details = {}
    checks = {"raw_generation_complete": complete,
              "needs_human_policy_match": response.get("needs_human") == case.get("expected_needs_human", True)}
    for prefix, text, ids, guard in (("raw", raw_text, raw_ids, raw_guard),
                                     ("effective", effective_text, effective_ids, effective_guard)):
        details[prefix + "_citation_ids"] = sorted(ids)
        details[prefix + "_unknown_ids"] = sorted(ids - authority)
        details[prefix + "_valid_but_disallowed_ids"] = sorted((ids & authority) - allowed)
        details[prefix + "_forbidden_ids"] = sorted(ids & forbidden)
        details[prefix + "_missing_required_ids"] = sorted(required - ids)
        details[prefix + "_guard_reasons"] = guard
        details[prefix + "_detected_language"] = detect_language(text)
        checks.update({prefix + "_language_heuristic_match": detect_language(text) == expected_lang,
                       prefix + "_citations_known": not (ids - authority),
                       prefix + "_citations_allowed": not (ids - allowed),
                       prefix + "_forbidden_citations_absent": not (ids & forbidden),
                       prefix + "_required_citations_present": required <= ids,
                       prefix + "_internal_citations_absent": not bool(_INTERNAL.search(text)),
                       prefix + "_unqualified_citations_absent": not bool(_UNQUALIFIED.search(text)),
                       prefix + "_guard_pass": not guard})
    checks["effective_cited_list_supported_by_text"] = listed_ids <= effective_text_ids
    return {"checks": checks, "violations": [key for key, passed in checks.items() if not passed],
            "all_observable_constraints_pass": all(checks.values()),
            "citations_required": bool(required), "details": details,
            "human_gold": False,
            "limitations": ["language_is_heuristic", "guards_are_pattern_checks_not_safety_proof",
                            "source_id_existence_is_not_entailment", "needs_human_is_global_policy_not_quality"]}


def _latencies(values):
    values = sorted(values)
    return {"n": len(values), "median_seconds": statistics.median(values) if values else None,
            "p95_seconds": values[math.ceil(.95 * len(values)) - 1] if values else None,
            "method": "nearest_rank_small_synthetic_sample_all_attempts_including_failures"}


def summarize(rows):
    def group(items):
        required = [item for item in items if item["assessment"]["citations_required"]]
        return {"cases": len(items), "latency": _latencies([item["wall_seconds"] for item in items]),
                "generation_errors": sum(not item["assessment"]["checks"]["raw_generation_complete"] for item in items),
                "full_eos": sum(item["assessment"]["checks"]["raw_generation_complete"] for item in items),
                "all_observable_constraints_pass": sum(item["assessment"]["all_observable_constraints_pass"] for item in items),
                "citation_required_cases": len(required),
                "required_raw_citations_satisfied": sum(item["assessment"]["checks"]["raw_required_citations_present"] for item in required),
                "required_effective_citations_satisfied": sum(item["assessment"]["checks"]["effective_required_citations_present"] for item in required),
                "raw_guard_rejections": sum(not item["assessment"]["checks"]["raw_guard_pass"] for item in items),
                "raw_language_heuristic_matches": sum(item["assessment"]["checks"]["raw_language_heuristic_match"] for item in items),
                "raw_unknown_citation_cases": sum(bool(item["assessment"]["details"]["raw_unknown_ids"]) for item in items),
                "raw_wrong_scope_citation_cases": sum(bool(item["assessment"]["details"]["raw_valid_but_disallowed_ids"]) for item in items),
                "fatal_errors": sum(item.get("fatal_error") is not None for item in items)}
    return {"human_gold": False, "scope": "AI_authored_silver_observable_constraints_only",
            "overall": group(rows),
            "by_language": {lang: group([row for row in rows if row["expected_language"] == lang])
                            for lang in sorted({row["expected_language"] for row in rows})},
            "not_measured": ["human_language_quality", "semantic_entailment", "gold_category_priority_F1",
                             "confidence_calibration", "production_latency_SLO", "automation_quality"]}


def catalog_ids(path):
    payload = json.loads(Path(path).read_text(encoding="utf8"))
    return {f"rules:{section['id']}:{point['ref']}" for section in payload["rules"] for point in section["points"]}


def load_cases(path, authority):
    from support_copilot.schemas import TriageRequest
    cases = []
    seen = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            if not isinstance(case["id"], str) or case["id"] in seen or case.get("human_gold") is not False:
                raise ValueError("case identity/gold")
            if case["expected_language"] not in {"ru", "kk", "uz"}:
                raise ValueError("unsupported evaluation language")
            request = TriageRequest.model_validate(case["request"])
            allowed = set(case["allowed_citation_ids"])
            required = set(case.get("required_citation_ids", allowed))
            forbidden = set(case.get("forbidden_citation_ids", []))
            if not required <= allowed or allowed & forbidden or not (allowed | forbidden) <= authority:
                raise ValueError("citation constraints")
            prepared = case["prepared_messages"]
            users = [message for message in prepared if message["role"] == "user"]
            if len(users) != 1:
                raise ValueError("oracle payload count")
            payload = json.loads(users[0]["content"])
            ids = [item["canonical_id"] for item in payload.get("retrieved_passages", [])]
            if len(ids) != len(set(ids)) or not set(ids) <= authority or not set(ids) <= allowed:
                raise ValueError("oracle references")
            player = "\n".join(message.text for message in request.messages if message.role == "player")
            if payload.get("player_text") != player:
                raise ValueError("oracle differs from request player input")
            case["_oracle"] = {"ids": ids, "text": player, "category_hint": payload.get("category_hint")}
            case["_request"] = request
        except Exception:
            raise ValueError(f"Evaluation line {line_number} does not match the private scenario contract") from None
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("Evaluation has no cases")
    return cases


def _oracle_response(engine, case):
    from support_copilot.policy import guard_draft
    from support_copilot.generation import GenerationError
    oracle = case["_oracle"]
    rules = [engine.retriever.get(key) for key in oracle["ids"]]
    if any(rule is None for rule in rules):
        raise ValueError("Authoritative oracle rule disappeared")
    lang = case["expected_language"]
    raw = ""
    try:
        raw = engine.generator.generate(oracle["text"], lang, rules, oracle["category_hint"])
    except GenerationError:
        pass  # last_generation retains the real error; score it, never count fallback as EOS.
    effective, guard_reasons = guard_draft(raw, rules, lang)
    return {"lang": lang, "needs_human": True,
            "draft_reply": {"lang": lang, "text": effective,
                            "cited": sorted(set(_CITATION.findall(effective)))},
            "reasons": guard_reasons,
            "oracle_policy_defaults_applied": True,
            "oracle_expected_language_and_context_forced": True}


def run_cases(engine, cases, mode, authority, emit):
    rows = []
    for case in cases:
        engine.generator.last_generation = {}  # Do not accidentally score a previous row when policy skips generation.
        started = time.monotonic()
        fatal = None
        response = {}
        try:
            response = _as_dict(engine.triage(case["_request"])) if mode == "pipeline" else _oracle_response(engine, case)
        except Exception as exc:
            fatal = type(exc).__name__  # Never print or copy exception text containing a private input.
        metadata = dict(engine.generator.last_generation)
        elapsed = time.monotonic() - started
        row = {"record_type": "case", "id": case["id"], "topic": case.get("topic"),
               "mode": mode, "expected_language": case["expected_language"],
               "human_gold": False, "request": case["_request"].model_dump(mode="json"),
               "allowed_citation_ids": case["allowed_citation_ids"],
               "required_citation_ids": case.get("required_citation_ids", case["allowed_citation_ids"]),
               "forbidden_citation_ids": case.get("forbidden_citation_ids", []),
               "response": response, "generation": metadata, "wall_seconds": elapsed, "fatal_error": fatal,
               "assessment": evaluate_case(case, response, metadata, authority)}
        rows.append(row)
        emit(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--gguf-adapter-path", type=Path)
    parser.add_argument("--mode", choices=("pipeline", "oracle"), default="pipeline")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--base", action="store_true", help="Evaluate the local base without loading LoRA")
    args = parser.parse_args(argv)
    if not args.dry_run and args.output is None:
        parser.error("--output is required unless --dry-run")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    engine = None
    try:
        from support_copilot.config import Settings
        settings = replace(Settings.from_env(), generation_enabled=True, use_adapter=not args.base, inference_backend="llamacpp")
        if args.adapter_path:
            settings = replace(settings, adapter_path=args.adapter_path.resolve())
        if args.gguf_adapter_path:
            settings = replace(settings, gguf_adapter_path=args.gguf_adapter_path.resolve())
        authority = catalog_ids(settings.rules_path)
        cases = load_cases(args.evaluation, authority)
        cases = cases[:args.limit] if args.limit else cases
        evaluation_sha = file_sha(args.evaluation)
        if args.dry_run:
            print(json.dumps({"status": "validated_without_model_load", "cases": len(cases),
                              "evaluation_sha256": evaluation_sha, "human_gold": False}))
            return 0
        output = args.output.resolve()
        if not output.is_relative_to((ROOT / ".private").resolve()) or output.suffix != ".jsonl" or output.exists():
            raise ValueError("Use a new .jsonl output under .private")
        protected = [settings.model_path, settings.adapter_path, settings.gguf_path.parent,
                     settings.gguf_adapter_path.parent, ROOT / ".private/soup-layer-cache"]
        if any(output.is_relative_to(Path(path).resolve()) for path in protected):
            raise ValueError("Output may not be inside model, adapter or cache artifacts")
        manifest = {"record_type": "manifest", "human_gold": False, "mode": args.mode,
                    "adapter_requested": settings.use_adapter,
                    "cases": len(cases), "evaluation_sha256": evaluation_sha,
                    "corpus_sha256": file_sha(settings.rules_path),
                    "hf_adapter_sha256": file_sha(settings.adapter_path / "adapter_model.safetensors"),
                    "gguf_adapter_sha256": file_sha(settings.gguf_adapter_path),
                    "gguf_model_sha256": file_sha(settings.gguf_path),
                    "code_sha256": {name: file_sha(ROOT / "src/support_copilot" / name) for name in
                                    ("engine.py", "policy.py", "knowledge.py", "generation.py", "llamacpp.py", "classifier.py", "text.py", "config.py")},
                    "classifier_enabled": settings.classifier_enabled,
                    "classifier_manifest_sha256": file_sha(settings.classifier_path / "manifest.json") if settings.classifier_enabled else None,
                    "limitations": "AI-authored silver constraints, no human gold or semantic judge; small-sample timings only"}
        from support_copilot.engine import TriageEngine
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf8") as handle:
            def emit(value):
                handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
            emit(manifest)
            engine = TriageEngine(settings)
            try:
                engine.load()
                emit({"record_type": "runtime", "health": engine.health()})
                rows = run_cases(engine, cases, args.mode, authority, emit)
                summary = summarize(rows)
                emit({"record_type": "summary", **summary})
                overall = summary["overall"]
                print(json.dumps({"output": str(output), "overall": overall, "human_gold": False}))
                if overall["generation_errors"] or overall["fatal_errors"]:
                    return 1
                return 0 if overall["all_observable_constraints_pass"] == len(rows) else 2
            except Exception as exc:
                emit({"record_type": "fatal", "type": type(exc).__name__, "human_gold": False})
                raise
    except Exception as exc:
        print("Evaluation failed: " + type(exc).__name__ + ". Inspect private evidence/configuration.", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
