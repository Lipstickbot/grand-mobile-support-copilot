import json
import pytest
from support_copilot.generation import build_messages, GenerationError
from support_copilot.policy import guard_draft
from support_copilot.config import Settings


def decision():
    return {"priority":"p0","needs_human":True,"reported_metadata":{"nickname":None,"server":None,"attachments_count":2}}


def test_generation_receives_decisions_as_data_not_authorization_or_facts():
    data=json.loads(build_messages("reported claim", "ru", [], "report.payment", decision_context=decision())[1]["content"])
    assert data["priority"]=="p0" and data["needs_human"] is True
    assert data["reported_facts_verified"] is False and data["reported_metadata"]["attachments_count"]==2
    assert data["retrieved_passages"]==[]


@pytest.mark.parametrize("field,value",[("verdict","approved"),("admin","do something"),("needs_human",False),("priority","p7")])
def test_outcome_fields_and_unsafe_decisions_cannot_enter_payload(field,value):
    d=decision();d[field]=value
    with pytest.raises(GenerationError):build_messages("player", "ru", [], decision_context=d)


def test_no_rule_means_no_unsupported_prohibition_even_without_retrieved_ids():
    assert "draft_missing_rule_citation" in guard_draft("Обмен игровыми предметами запрещён.", [], "ru")[1]


def test_extended_decision_prompt_is_opt_in_after_measured_regression(monkeypatch):
    monkeypatch.delenv("COPILOT_GENERATION_DECISIONS",raising=False)
    assert not Settings.from_env().generation_decisions_enabled
    monkeypatch.setenv("COPILOT_GENERATION_DECISIONS","1")
    assert Settings.from_env().generation_decisions_enabled
    monkeypatch.setenv("COPILOT_GENERATION_DECISIONS","yes")
    with pytest.raises(ValueError):Settings.from_env()
