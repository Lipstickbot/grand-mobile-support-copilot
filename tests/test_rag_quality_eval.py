"""Observable evaluation regression tests using invented rules and no model."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("rag_quality_eval", Path(__file__).parents[1] / "scripts/evaluate_rag_quality.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)

RULE = "rules:900:2.1"
OTHER = "rules:901:2.1"
TEXT = f"Уточните время обращения для проверки [{RULE}]."


def scenario():
    return {"expected_language": "ru", "expected_needs_human": True,
            "allowed_citation_ids": [RULE], "required_citation_ids": [RULE],
            "forbidden_citation_ids": [OTHER]}


def assess(raw=TEXT, effective=TEXT, cited=None, **metadata):
    return evaluation.evaluate_case(scenario(), {"needs_human": True,
        "draft_reply": {"text": effective, "cited": [RULE] if cited is None else cited}},
        {"draft": raw, "status": "ok", "finish_reason": "eos", **metadata}, {RULE, OTHER})


def test_complete_known_citation_can_pass_observable_checks_only():
    result = assess()
    assert result["all_observable_constraints_pass"]
    assert result["human_gold"] is False
    assert "source_id_existence_is_not_entailment" in result["limitations"]


def test_fallback_does_not_hide_raw_missing_citation():
    result = assess(raw="Уточните время и сервер.")
    assert "raw_required_citations_present" in result["violations"]
    assert result["checks"]["effective_required_citations_present"]


@pytest.mark.parametrize("metadata", [dict(finish_reason="max_tokens"), dict(status="error"),
                                      dict(error_code="timeout")])
def test_partial_or_failed_generations_cannot_pass_even_with_valid_effective_text(metadata):
    assert not assess(**metadata)["checks"]["raw_generation_complete"]


def test_existing_but_wrong_scope_source_is_distinct_from_unknown_source():
    result = assess(raw=f"Уточните обстоятельства [{OTHER}].")
    assert result["details"]["raw_unknown_ids"] == []
    assert result["details"]["raw_valid_but_disallowed_ids"] == [OTHER]
    assert result["details"]["raw_forbidden_ids"] == [OTHER]
    result = assess(raw="Уточните обстоятельства [rules:999:9.9].")
    assert result["details"]["raw_unknown_ids"] == ["rules:999:9.9"]


def test_cited_list_cannot_invent_a_text_citation():
    result = assess(effective="Уточните обстоятельства.")
    assert not result["checks"]["effective_cited_list_supported_by_text"]


def test_guard_and_language_failures_remain_visible():
    result = assess(raw="Игрок будет наказан.")
    assert not result["checks"]["raw_guard_pass"]
    assert not assess(raw="Please provide details.")["checks"]["raw_language_heuristic_match"]


def test_latency_nearest_rank_and_failed_attempts_are_counted():
    rows = [{"assessment": assess(status="error" if i == 0 else "ok"),
             "wall_seconds": i + 1, "expected_language": "ru", "fatal_error": None} for i in range(4)]
    summary = evaluation.summarize(rows)
    assert summary["overall"]["latency"]["n"] == 4
    assert summary["overall"]["latency"]["p95_seconds"] == 4
    assert summary["overall"]["generation_errors"] == 1
    assert summary["overall"]["full_eos"] == 3
