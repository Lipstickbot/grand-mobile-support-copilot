"""Invented pure robustness-assessment fixtures, no model or source documents."""
import importlib.util
from pathlib import Path

from support_copilot.coretypes import RetrievedRule

spec = importlib.util.spec_from_file_location("robust_eval", Path(__file__).parents[1]/"scripts/evaluate_robustness.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
RULE = RetrievedRule("rules:900:2.1", 900, "Invented", "2.1", "Invented source")


def case(lang="en", generation=True):
    return {"expected_language": lang, "generation_expected": generation,
            "request": {"messages": [{"role": "player", "text": "Please help"}]},
            "allowed_citation_ids": [], "required_citation_ids": [], "forbidden_citation_ids": []}


def response(lang="en", text="Please provide details."):
    return {"lang": lang, "needs_human": True, "extracted": {"has_evidence": False},
            "draft_reply": {"lang": "en" if lang == "und" else lang, "text": text, "cited": []},
            "reasons": ["topic_gate_selected="]}


def test_intentional_unknown_language_clarification_does_not_require_eos():
    result = evaluation.assess(case("und", False), response("und"), {}, {RULE.canonical_id: RULE})
    assert result["all_observable_constraints_pass"]
    assert not result["generation_complete"] and not result["generation_expected"]


def test_generation_skip_for_known_language_cannot_count_as_success():
    result = evaluation.assess(case(), response(), {}, {RULE.canonical_id: RULE})
    assert not result["checks"]["generation_route_match"]


def test_screenshot_count_does_not_make_an_image_review_claim_valid():
    c = case()
    c["request"]["messages"][0]["attachments"] = 3
    r = response(text="The screenshot clearly shows a violation.")
    r["extracted"]["has_evidence"] = True
    result = evaluation.assess(c, r, {"status": "ok", "finish_reason": "eos", "draft": r["draft_reply"]["text"]}, {})
    assert result["checks"]["evidence_presence_only"]
    assert not result["checks"]["effective_no_claimed_image_review"]


def test_known_but_unrelated_source_does_not_pass_allowed_citations():
    r = response(text="Please clarify [rules:900:2.1].")
    r["draft_reply"]["cited"] = [RULE.canonical_id]
    result = evaluation.assess(case(), r, {}, {RULE.canonical_id: RULE})
    assert result["checks"]["effective_citations_known"]
    assert not result["checks"]["effective_citations_allowed"]


def test_required_citation_is_not_hidden_by_a_safe_fallback():
    c = case()
    c["required_citation_ids"] = c["allowed_citation_ids"] = [RULE.canonical_id]
    result = evaluation.assess(c, response(), {}, {RULE.canonical_id: RULE})
    assert not result["checks"]["effective_required_citations_present"]
