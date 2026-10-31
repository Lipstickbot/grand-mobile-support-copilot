"""Invented fixtures + httpx MockTransport only. No live HTTP or model calls."""
import copy
import importlib.util
import json
from pathlib import Path

import httpx
import pytest

SPEC = importlib.util.spec_from_file_location("evaluate_http_quality", Path(__file__).resolve().parents[1]/"scripts/evaluate_http_quality.py")
quality = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quality)
from support_copilot.coretypes import RetrievedRule

AUTHORITY = {key: RetrievedRule(key, n, "Invented", f"{n}.1", "Invented test rule")
             for n, key in enumerate(("rules:1:1.1", "rules:2:2.1"), 1)}


def case(lang="ru", required=True):
    return {"id": "invented", "dataset": "test", "group": "test", "expected_language": lang,
            "request": {"kind": "complaint", "messages": [{"role": "player", "text": "Помогите с жалобой"}]},
            "required_citation_ids": ["rules:1:1.1"] if required else [],
            "allowed_citation_ids": ["rules:1:1.1"], "forbidden_citation_ids": ["rules:2:2.1"]}


def response(text="Ваша жалоба передана администрации для проверки. [rules:1:1.1]", cited=None):
    return {"lang": "ru", "category": "complaint.insult", "priority": "p2", "is_spam_or_duplicate": False,
            "extracted": {"nickname": None, "rule_refs": [], "has_evidence": False, "server": None},
            "draft_reply": {"lang": "ru", "text": text, "cited": ["rules:1:1.1"] if cited is None else cited},
            "needs_human": True, "confidence": .5, "reasons": ["topic_gate_selected=rules:1:1.1"]}


def attempt(data=None, error=None, status=200):
    return {"status_code": status, "response": response() if data is None else data,
            "error": error, "wall_seconds": 2.0}


def test_valid_effective_response_does_not_claim_raw_generation():
    actual = quality.assess_triage(case(), attempt(), AUTHORITY)
    assert actual["all_observable_constraints_pass"], actual
    assert actual["raw_output_and_eos_available"] is False
    assert "semantic_entailment" in actual["not_measured"]


@pytest.mark.parametrize("source,expected", [("rules:999:999.1", "citations_known"),
    ("rules:2:2.1", "citations_allowed"), ("", "required_citations_present")])
def test_invented_out_of_scope_and_missing_sources_fail(source, expected):
    actual = quality.assess_triage(case(), attempt(response("Ваша жалоба передана администрации. "+source,
        [source] if source else [])), AUTHORITY)
    assert actual["checks"][expected] is False
    assert actual["all_observable_constraints_pass"] is False


@pytest.mark.parametrize("marker", ["[rule_1234]", "[rule-12345]", "[Rule 12345]", "[правило_123]"])
def test_pseudocitations_fail_independently_of_canonical_list(marker):
    actual = quality.assess_triage(case(required=False), attempt(response("Ваша жалоба передана администрации. "+marker, [])), AUTHORITY)
    assert actual["checks"]["effective_noncanonical_rule_markers_absent"] is False


@pytest.mark.parametrize("text,issue", [("Пришлите пароль от аккаунта для проверки.", "draft_secret_request"),
    ("Я проверил скриншот, игрок виновен.", "draft_unverified_verdict_or_evidence")])
def test_effective_safety_violations_are_observable(text, issue):
    actual = quality.assess_triage(case(required=False), attempt(response(text, [])), AUTHORITY)
    assert issue in actual["guard_reasons"]
    assert actual["checks"]["effective_guard_pass"] is False


def test_attachment_presence_is_not_image_verification():
    sample = case()
    sample["request"]["messages"][0]["attachments"] = 2
    actual = quality.assess_triage(sample, attempt(), AUTHORITY)
    assert actual["checks"]["evidence_presence_only"] is False
    sample["request"]["messages"][0]["role"] = "admin"
    assert quality.assess_triage(sample, attempt(), AUTHORITY)["checks"]["evidence_presence_only"] is True


def test_und_requires_explicit_clarification_and_allows_documented_ru_draft():
    data = response("Уточните, пожалуйста, на каком языке вы хотите получить ответ.", [])
    data["lang"] = "und"
    data["reasons"] = ["draft=language_clarification"]
    result = quality.assess_triage(case("und", required=False), attempt(data), AUTHORITY)
    assert result["checks"]["routing_reason_match"]
    assert result["checks"]["primary_language_match"]
    assert result["checks"]["effective_output_language_heuristic_match"]


def test_fallback_and_reported_generation_error_are_separate():
    data = response()
    data["reasons"] += ["generation_error=deadline_exceeded", "draft=localized_template_fallback"]
    result = quality.assess_triage(case(), attempt(data), AUTHORITY)
    assert result["fallback"]
    assert result["reported_generation_errors"] == ["generation_error=deadline_exceeded"]
    assert result["checks"]["no_reported_generation_error"] is False
    assert result["checks"]["effective_guard_pass"]


@pytest.mark.parametrize("url", ["https://127.0.0.1:8000", "http://localhost:8000", "http://127.0.0.1.evil.test", "http://127.0.0.1:8000/x", "http://user@127.0.0.1", "http://[::1]", "http://127.0.0.1?x=1"])
def test_only_literal_loopback_origin_accepted(url):
    with pytest.raises(ValueError):
        quality.loopback_url(url)


def test_literal_loopback_accepted():
    assert quality.loopback_url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"


def test_timeout_and_bad_status_remain_in_denominators_and_continue():
    calls = []
    def transport(req):
        calls.append(req.url.path)
        if len(calls) == 1:
            raise httpx.ReadTimeout("invented timeout", request=req)
        return httpx.Response(503, text="invented unavailable")
    records = []
    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(transport)) as client:
        rows, labeled = quality.run(client, [case(), case("und", required=False)], [], AUTHORITY, records.append)
    summary = quality.summarize(rows, labeled, ["complaint.insult"])["triage"]["overall"]
    assert calls == ["/triage", "/triage"]
    assert summary["cases"] == summary["http_errors"] == summary["latency_all"]["n"] == 2
    assert summary["citation_required_cases"] == 1
    assert summary["required_citations_satisfied"] == 0
    assert summary["latency_expected_generation"]["n"] == 1
    assert summary["latency_expected_clarification"]["n"] == 1
    assert rows[0]["attempt"]["error"] == "timeout"
    assert rows[1]["attempt"]["error_response_body"] == "invented unavailable"


@pytest.mark.parametrize("payload,error", [(b"[]", "invalid_json_shape"), (b"invalid", "invalid_json")])
def test_invalid_json_preserved_as_failed_attempt(payload, error):
    with httpx.Client(base_url="http://127.0.0.1", transport=httpx.MockTransport(lambda req: httpx.Response(200, content=payload))) as client:
        actual = quality.post(client, "/triage", {})
    assert actual["error"] == error
    assert actual["wall_seconds"] >= 0


def test_invalid_schema_prediction_cannot_inflate_classifier_accuracy():
    rows = []
    for valid in (True, False):
        rows.append({"case": {"expected_category": "complaint.insult"}, "attempt": attempt({"category": "complaint.insult"}),
                     "assessment": {"checks": {"response_schema": valid}}})
    result = quality.classification_metrics(rows, "category", ["complaint.insult"])
    assert result["n"] == 2 and result["accuracy"] == .5
    assert result["per_class"]["complaint.insult"]["recall"] == .5
    assert result["confusion_matrix"][0] == [1, 1]
    assert result["human_gold"] is False


def test_malformed_draft_language_does_not_abort_remaining_cases():
    data = response()
    data["draft_reply"]["lang"] = {"malformed": True}
    result = quality.assess_triage(case(), attempt(data), AUTHORITY)
    assert result["checks"]["response_schema"] is False


def test_health_identity_ignores_request_counters_but_detects_adapter_change():
    before = {"generation": {"backend": "invented", "adapter_loaded": False, "last_status": None},
              "classifier": {"artifact_sha256": "invented"}, "corpus_version": "invented"}
    after = copy.deepcopy(before)
    after["generation"]["last_status"] = "ok"
    assert quality.health_identity(before) == quality.health_identity(after)
    after["generation"]["adapter_loaded"] = True
    assert quality.health_identity(before) != quality.health_identity(after)


def test_fresh_challenge_remains_separate_and_strips_answer_keys(tmp_path):
    rule_path = tmp_path/"rules.json"
    rule_path.write_text(json.dumps({"rules": [{"id": 1, "points": [{"ref": "1.1", "text": "Invented"}]},
        {"id": 2, "points": [{"ref": "2.1", "text": "Invented unrelated"}]}]}), encoding="utf8")
    paths = [tmp_path/name for name in ("classic.jsonl", "robust.jsonl", "classifier.jsonl", "challenge.jsonl")]
    for path, prefix in ((paths[0], "synthetic-eval:a"), (paths[1], "robust-v4:a"), (paths[3], "challenge:a")):
        row = {**case(), "id": prefix, "human_gold": False, "reference_draft": "MUST NOT SEND", "prepared_messages": ["MUST NOT SEND"]}
        path.write_text(json.dumps(row)+"\n", encoding="utf8")
    paths[2].write_text(json.dumps({"human_gold": False, "source": "authored_multilingual_synthetic", "lang": "ru", "kind": "complaint", "category": "complaint.insult", "priority": "p2", "text": "Invented example"})+"\n", encoding="utf8")
    triage, classifier, authority = quality.load_inputs(*paths[:3], rule_path, paths[3])
    assert [r["dataset"] for r in triage] == ["classic36", "robust40", "fresh_challenge"]
    assert triage[-1]["group"] == "fresh_challenge"
    assert "MUST NOT SEND" not in json.dumps(triage)
    assert len(classifier) == 1 and len(authority) == 2
