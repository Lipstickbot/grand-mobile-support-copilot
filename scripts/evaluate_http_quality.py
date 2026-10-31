"""Measure the running loopback service: effective outputs, not hidden raw/EOS.

Sequential synthetic requests only; no server/model startup, retries or external
API. Exit0 all observable checks,2 completed with failures,1 fatal setup error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
CLASSIC = ROOT / ".private/rag-training-v2/prepared-reviewed/evaluation.jsonl"
ROBUST = ROOT / ".private/input-v4/robustness-cases.jsonl"
CLASSIFIER = ROOT / ".private/classifier-v1/synthetic-rows.jsonl"
RULES = ROOT / ".private/data/rules.json"
TAXONOMY = ROOT / ".private/data/taxonomy.json"
LABEL_KIND = "authored_synthetic_development_not_human_gold"
CITATION = re.compile(r"\brules:\d+:\d+(?:\.\d+)+")
INTERNAL = re.compile(r"\binternal:[\w:.-]+")
PSEUDO_RULE = re.compile(r"\[\s*(?:rule|правило|пункт)[_\- #]*\d+(?:[._-]\d+)*\s*\]", re.I)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def loopback_url(value):
    try:
        u = urlsplit(value)
        valid = (u.scheme == "http" and u.hostname == "127.0.0.1" and u.username is None
                 and u.password is None and u.path in {"", "/"} and not u.query and not u.fragment
                 and (u.port is None or 1 <= u.port <= 65535))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Only an explicit HTTP127.0.0.1 origin is allowed")
    return value.rstrip("/")


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def load_inputs(classic=CLASSIC, robust=ROBUST, classifier=CLASSIFIER, rules=RULES, challenge=None):
    from support_copilot.coretypes import RetrievedRule
    from support_copilot.schemas import TriageRequest
    authority = {f"rules:{s['id']}:{p['ref']}": RetrievedRule(
        f"rules:{s['id']}:{p['ref']}", s["id"], s.get("name", s.get("title", "")), p["ref"], p["text"])
        for s in json.loads(Path(rules).read_text(encoding="utf8"))["rules"] for p in s["points"]}
    triage, seen = [], set()
    fixtures = [(classic, "classic36", "synthetic-eval:"), (robust, "robust40", "robust-v4:")]
    if challenge:
        fixtures.append((challenge, "fresh_challenge", ""))
    for path, dataset, prefix in fixtures:
        for item in read_rows(path):
            required = set(item.get("required_citation_ids", item["allowed_citation_ids"]))
            allowed, forbidden = set(item["allowed_citation_ids"]), set(item["forbidden_citation_ids"])
            if (item.get("human_gold") is not False or not item["id"].startswith(prefix)
                    or item["id"] in seen or not required <= allowed or allowed & forbidden
                    or not (allowed | forbidden) <= set(authority)
                    or item["expected_language"] not in {"ru", "kk", "uz", "en", "und"}):
                raise ValueError("Invalid synthetic triage case")
            request = TriageRequest.model_validate(item["request"]).model_dump(mode="json", exclude_none=True)
            seen.add(item["id"])
            triage.append({"id": item["id"], "dataset": dataset,
                "group": "fresh_challenge" if dataset == "fresh_challenge" else item.get("group", item.get("topic", "other")),
                "scenario_group": item.get("group", item.get("topic", "other")), "fixture_source": str(Path(path).relative_to(ROOT)) if Path(path).is_relative_to(ROOT) else Path(path).name,
                "label_kind": item.get("label_kind", LABEL_KIND),
                "expected_language": item["expected_language"], "request": request,
                "required_citation_ids": sorted(required), "allowed_citation_ids": sorted(allowed),
                "forbidden_citation_ids": sorted(forbidden), "expected_category": item.get("expected_category"),
                "maximum_priority": item.get("maximum_priority"), "human_gold": False})
    labeled = []
    for n, item in enumerate(read_rows(classifier)):
        if (item.get("human_gold") is not False or item.get("source") != "authored_multilingual_synthetic"
                or item.get("lang") not in {"ru", "kk", "uz"} or item.get("kind") not in {"complaint", "report"}
                or not item["category"].startswith(item["kind"]+".") or item["priority"] not in {"p0", "p1", "p2", "p3"}):
            raise ValueError("Invalid authored classifier case")
        request = TriageRequest.model_validate({"kind": item["kind"], "messages": [{"role": "player", "text": item["text"]}]}).model_dump(mode="json", exclude_none=True)
        labeled.append({"id": f"classifier-synthetic:{n:03d}", "expected_language": item["lang"],
            "expected_category": item["category"], "expected_priority": item["priority"], "request": request,
            "human_gold": False})
    if not triage or not labeled:
        raise ValueError("Empty synthetic fixture input")
    return triage, labeled, authority


def post(client, endpoint, payload):
    """One actual attempt; timeout/error never disappears from the denominator."""
    import httpx
    start = time.perf_counter()
    response, status, error, error_body = None, None, None, None
    try:
        r = client.post(endpoint, json=payload)
        status = r.status_code
        if status == 200:
            response = r.json()
            if not isinstance(response, dict):
                response, error = None, "invalid_json_shape"
        else:
            error = "http_status_"+str(status)
            error_body = r.text
    except httpx.TimeoutException:
        error = "timeout"
    except httpx.HTTPError as exc:
        error = type(exc).__name__
    except (ValueError, TypeError):
        error = "invalid_json"
    return {"status_code": status, "response": response, "error": error,
            "error_response_body": error_body, "wall_seconds": time.perf_counter()-start}


def assess_triage(case, attempt, authority):
    from support_copilot.policy import detect_language, guard_draft
    from support_copilot.schemas import TriageResponse
    data = attempt.get("response") or {}
    valid = False
    if attempt.get("status_code") == 200 and not attempt.get("error"):
        try:
            TriageResponse.model_validate(data)
            valid = True
        except Exception:
            pass
    draft = data.get("draft_reply") if isinstance(data.get("draft_reply"), dict) else {}
    text = draft.get("text") if isinstance(draft.get("text"), str) else ""
    listed = draft.get("cited") if isinstance(draft.get("cited"), list) else []
    listed = {item for item in listed if isinstance(item, str)}
    text_ids = set(CITATION.findall(text))
    ids = text_ids | listed
    reasons = data.get("reasons") if isinstance(data.get("reasons"), list) else []
    reasons = [r for r in reasons if isinstance(r, str)]
    selected = next((r.split("=", 1)[1].split(",") for r in reasons if r.startswith("topic_gate_selected=")), [])
    context = [authority[key] for key in selected if key in authority]
    effective_lang = draft.get("lang") if isinstance(draft.get("lang"), str) else case["expected_language"]
    guard_reasons = guard_draft(text, context, effective_lang)[1]
    expected_evidence = case["request"].get("images_count", 0) > 0 or any(
        m.get("attachments", 0) > 0 for m in case["request"]["messages"] if m["role"] != "admin")
    required, allowed, forbidden = (set(case[k]) for k in ("required_citation_ids", "allowed_citation_ids", "forbidden_citation_ids"))
    checks = {"http_success": attempt.get("status_code") == 200 and not attempt.get("error"),
        "response_schema": valid, "primary_language_match": data.get("lang") == case["expected_language"],
        "effective_text_present": bool(text.strip()), "needs_human": data.get("needs_human") is True,
        "evidence_presence_only": isinstance(data.get("extracted"), dict) and data["extracted"].get("has_evidence") == expected_evidence,
        "citations_known": ids <= set(authority), "citations_allowed": ids <= allowed,
        "required_citations_present": required <= ids, "forbidden_citations_absent": not bool(ids & forbidden),
        "cited_list_supported_by_text": listed <= text_ids, "internal_citations_absent": not bool(INTERNAL.search(text)),
        "effective_noncanonical_rule_markers_absent": not bool(PSEUDO_RULE.search(text)),
        "effective_guard_pass": not guard_reasons,
        "effective_output_language_heuristic_match": bool(text) and detect_language(text) == (effective_lang if case["expected_language"] == "und" else case["expected_language"])}
    if case.get("expected_category"):
        checks["authored_category_match"] = data.get("category") == case["expected_category"]
    if case.get("maximum_priority"):
        actual = data.get("priority")
        checks["priority_safety_floor"] = actual in {"p0", "p1", "p2", "p3"} and int(actual[1]) <= int(case["maximum_priority"][1])
    reported_errors = [r for r in reasons if r.startswith("generation_error=")]
    fallback = "draft=localized_template_fallback" in reasons
    clarification = "draft=language_clarification" in reasons
    checks["no_reported_generation_error"] = not reported_errors
    checks["routing_reason_match"] = clarification if case["expected_language"] == "und" else not clarification
    return {"checks": checks, "violations": sorted(k for k, value in checks.items() if not value),
        "all_observable_constraints_pass": all(checks.values()), "citations_required": bool(required),
        "missing_required_ids": sorted(required-ids), "unknown_ids": sorted(ids-set(authority)),
        "valid_but_disallowed_ids": sorted((ids & set(authority))-allowed), "forbidden_ids": sorted(ids & forbidden),
        "guard_reasons": guard_reasons, "fallback": fallback, "clarification": clarification,
        "reported_generation_errors": reported_errors, "raw_output_and_eos_available": False,
        "not_measured": ["raw_generation_safety", "EOS_or_full_generation", "semantic_entailment", "human_language_quality", "production_SLO"]}


def assess_classifier(case, attempt):
    from support_copilot.schemas import ClassificationResponse
    data = attempt.get("response") or {}
    valid = False
    if attempt.get("status_code") == 200 and not attempt.get("error"):
        try:
            ClassificationResponse.model_validate(data)
            valid = True
        except Exception:
            pass
    checks = {"http_success": attempt.get("status_code") == 200 and not attempt.get("error"),
        "response_schema": valid, "primary_language_match": data.get("lang") == case["expected_language"],
        "category_match": data.get("category") == case["expected_category"],
        "priority_match": data.get("priority") == case["expected_priority"], "needs_human": data.get("needs_human") is True}
    return {"checks": checks, "violations": sorted(k for k, value in checks.items() if not value),
            "all_observable_constraints_pass": all(checks.values()), "human_gold": False}


def latency(values):
    values = sorted(values)
    return {"n": len(values), "p50_seconds": statistics.median(values) if values else None,
            "p95_seconds": values[math.ceil(.95*len(values))-1] if values else None,
            "method": "nearest_rank_all_HTTP_attempts_including_failures"}


def health_identity(health):
    """Only deployment identity/settings; changing last-request counters are not drift."""
    if not isinstance(health, dict):
        return None
    generation = health.get("generation") or {}
    classifier = health.get("classifier") or {}
    return {"generation": {k: generation.get(k) for k in
        ("backend", "adapter_loaded", "max_prompt_tokens", "max_new_tokens", "timeout_seconds", "context_size")},
        "classifier_artifact_sha256": classifier.get("artifact_sha256"),
        "corpus_version": health.get("corpus_version"), "taxonomy_version": health.get("taxonomy_version")}


def classification_metrics(rows, field, labels):
    """Failed/invalid predictions count as FN and never shrink n."""
    invalid = "__invalid_or_failed__"
    matrix_labels = list(labels)+[invalid]
    matrix = [[0 for _ in matrix_labels] for _ in matrix_labels]
    for row in rows:
        actual = row["case"]["expected_"+field]
        prediction = (row["attempt"].get("response") or {}).get(field)
        if (row["attempt"].get("error") or row["attempt"].get("status_code") != 200
                or not row["assessment"]["checks"]["response_schema"] or prediction not in labels):
            prediction = invalid
        matrix[matrix_labels.index(actual)][matrix_labels.index(prediction)] += 1
    per_class, f1s = {}, []
    correct = sum(matrix[i][i] for i in range(len(labels)))
    for i, label in enumerate(labels):
        tp, support, guesses = matrix[i][i], sum(matrix[i]), sum(r[i] for r in matrix)
        precision, recall = tp/guesses if guesses else 0., tp/support if support else 0.
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.
        per_class[label] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
        if support:
            f1s.append(f1)
    return {"n": len(rows), "correct": correct, "accuracy": correct/len(rows) if rows else None,
        "macro_f1_supported_classes": statistics.mean(f1s) if f1s else None,
        "classes_with_support": len(f1s), "total_classes": len(labels),
        "confusion_labels": matrix_labels, "confusion_matrix": matrix, "per_class": per_class,
        "label_kind": LABEL_KIND, "human_gold": False}


def summarize(triage, classified, categories):
    def group(rows):
        required = [r for r in rows if r["assessment"]["citations_required"]]
        return {"cases": len(rows), "http_errors": sum(bool(r["attempt"]["error"]) for r in rows),
            "valid_responses": sum(r["assessment"]["checks"]["response_schema"] for r in rows),
            "language_matches": sum(r["assessment"]["checks"]["primary_language_match"] for r in rows),
            "all_observable_constraints_pass": sum(r["assessment"]["all_observable_constraints_pass"] for r in rows),
            "citation_required_cases": len(required),
            "required_citations_satisfied": sum(r["assessment"]["checks"]["required_citations_present"] for r in required),
            "unknown_citation_cases": sum(bool(r["assessment"]["unknown_ids"]) for r in rows),
            "wrong_scope_citation_cases": sum(bool(r["assessment"]["valid_but_disallowed_ids"]) for r in rows),
            "effective_guard_rejections": sum(not r["assessment"]["checks"]["effective_guard_pass"] for r in rows),
            "fallback_cases": sum(r["assessment"]["fallback"] for r in rows),
            "reported_generation_error_cases": sum(bool(r["assessment"]["reported_generation_errors"]) for r in rows),
            "latency_all": latency([r["attempt"]["wall_seconds"] for r in rows]),
            "latency_expected_generation": latency([r["attempt"]["wall_seconds"] for r in rows if r["case"]["expected_language"] != "und"]),
            "latency_expected_clarification": latency([r["attempt"]["wall_seconds"] for r in rows if r["case"]["expected_language"] == "und"])}
    return {"human_gold": False, "label_kind": LABEL_KIND, "triage": {"overall": group(triage),
        "by_language": {lang: group([r for r in triage if r["case"]["expected_language"] == lang]) for lang in sorted({r["case"]["expected_language"] for r in triage})},
        "by_dataset": {d: group([r for r in triage if r["case"]["dataset"] == d]) for d in sorted({r["case"]["dataset"] for r in triage})},
        "by_group": {g: group([r for r in triage if r["case"]["group"] == g]) for g in sorted({r["case"]["group"] for r in triage})}},
        "classifier": {"cases": len(classified), "http_errors": sum(bool(r["attempt"]["error"]) for r in classified),
            "category": classification_metrics(classified, "category", categories),
            "priority": classification_metrics(classified, "priority", ["p0", "p1", "p2", "p3"]),
            "latency": latency([r["attempt"]["wall_seconds"] for r in classified]),
            "by_language": {lang: {field: classification_metrics([r for r in classified if r["case"]["expected_language"] == lang], field, labels)
                for field, labels in (("category", categories), ("priority", ["p0", "p1", "p2", "p3"]))}
                for lang in sorted({r["case"]["expected_language"] for r in classified})}},
        "not_measured": ["raw_model_text", "EOS/full_generation", "human_blind_quality", "semantic_entailment", "gold_hidden_accuracy", "production_SLO"]}


def run(client, triage, classified, authority, emit):
    triage_rows, classifier_rows = [], []
    for cases, endpoint, target, name in ((triage, "/triage", triage_rows, "triage_case"),
                                         (classified, "/v1/classify", classifier_rows, "classifier_case")):
        for case in cases:
            attempt = post(client, endpoint, case["request"])
            assessment = assess_triage(case, attempt, authority) if name == "triage_case" else assess_classifier(case, attempt)
            row = {"record_type": name, "case": case, "attempt": attempt, "assessment": assessment}
            target.append(row)
            emit(row)
    return triage_rows, classifier_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--challenge", type=Path, help="Optional authored fresh challenge JSONL; reported separately, never human gold")
    parser.add_argument("--challenge-only", action="store_true", help="Evaluate only the supplied challenge; retain the separate42 classifier scenarios")
    args = parser.parse_args(argv)
    try:
        import httpx
        origin = loopback_url(args.base_url)
        output = args.output.resolve()
        if not output.is_relative_to((ROOT/".private/quality-live").resolve()) or output.exists() or output.suffix != ".jsonl":
            raise ValueError("Use a new JSONL under.private/quality-live")
        if args.limit is not None and args.limit < 1:
            raise ValueError("Positive limit required")
        triage, classified, authority = load_inputs(challenge=args.challenge)
        if args.challenge_only:
            if not args.challenge:
                raise ValueError("--challenge-only requires --challenge")
            triage = [case for case in triage if case["dataset"] == "fresh_challenge"]
        if args.limit:
            triage = triage[:args.limit]
        categories = [r["code"] for r in json.loads(TAXONOMY.read_text(encoding="utf8"))["categories"]]
        with httpx.Client(base_url=origin, timeout=httpx.Timeout(50, connect=3), trust_env=False, follow_redirects=False) as client:
            health_response = client.get("/ready")
            health_response.raise_for_status()
            health = health_response.json()
            if not health.get("ready"):
                raise RuntimeError("Server not ready")
            manifest = {"record_type": "manifest", "human_gold": False, "label_kind": LABEL_KIND,
                "transport": "actual_sequential_HTTP_loopback", "base_url": origin,
                "selection": "challenge_only" if args.challenge_only else "all",
                "triage_cases": len(triage), "classifier_cases": len(classified),
                "fixture_sha256": {"classic36": sha(CLASSIC), "robust40": sha(ROBUST), "classifier42": sha(CLASSIFIER), "rules": sha(RULES), "taxonomy": sha(TAXONOMY)},
                "code_sha256": {p.name: sha(p) for p in sorted((ROOT/"src/support_copilot").glob("*.py"))},
                "evaluator_sha256": sha(Path(__file__)), "health_before": health,
                "raw_output_and_eos_available": False}
            manifest["server_identity"] = health_identity(health)
            fingerprint_paths = {"classic36": CLASSIC, "robust40": ROBUST, "classifier42": CLASSIFIER, "rules": RULES, "taxonomy": TAXONOMY}
            if args.challenge:
                manifest["fixture_sha256"]["fresh_challenge"] = sha(args.challenge)
                manifest["challenge_source"] = str(args.challenge)
                fingerprint_paths["fresh_challenge"] = args.challenge
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf8") as handle:
                def emit(value):
                    handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
                    handle.flush()
                emit(manifest)
                t, c = run(client, triage, classified, authority, emit)
                summary = summarize(t, c, categories)
                try:
                    after = client.get("/ready")
                    health_after = after.json() if after.status_code == 200 else None
                    emit({"record_type": "health_after", "status_code": after.status_code, "health": health_after})
                except (httpx.HTTPError, ValueError):
                    health_after = None
                    emit({"record_type": "health_after", "health": None, "error": "health_unavailable"})
                summary["health_ready_after"] = isinstance(health_after, dict) and health_after.get("ready") is True
                summary["health_identity_stable"] = health_identity(health_after) == health_identity(health)
                summary["code_sha256_after"] = {p.name: sha(p) for p in sorted((ROOT/"src/support_copilot").glob("*.py"))}
                summary["code_unchanged"] = summary["code_sha256_after"] == manifest["code_sha256"]
                summary["fixture_sha256_after"] = {name: sha(path) for name, path in fingerprint_paths.items()}
                summary["fixtures_unchanged"] = summary["fixture_sha256_after"] == manifest["fixture_sha256"]
                emit({"record_type": "summary", **summary})
                print(json.dumps({"triage": summary["triage"]["overall"],
                    "classifier_category": summary["classifier"]["category"]["accuracy"],
                    "classifier_priority_macro_f1": summary["classifier"]["priority"]["macro_f1_supported_classes"],
                    "output": str(output), "human_gold": False}))
                return 0 if (summary["health_ready_after"] and summary["health_identity_stable"] and summary["code_unchanged"] and summary["fixtures_unchanged"]
                    and all(r["assessment"]["all_observable_constraints_pass"] for r in t+c)) else 2
    except Exception as exc:
        print("HTTP quality evaluation failed: "+type(exc).__name__+". Inspect local configuration/private evidence.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
