"""Urgency is independent of generation and cannot lose an explicit p0 floor."""
import pytest
from support_copilot.priority import decide_priority, priority_floor


@pytest.mark.parametrize("topic,category", [("payment","report.payment"),("account","report.account")])
@pytest.mark.parametrize("learned", ["p0","p1","p2","p3",None])
def test_sensitive_topics_cannot_be_demoted(topic, category, learned):
    d=decide_priority("reported issue", category, topics={topic}, learned_priority=learned, learned_uncertain=False)
    assert d.priority=="p0" and d.needs_human


def test_ban_appeal_requires_human_without_inventing_taxonomy_p0():
    d=decide_priority("Прошу пересмотреть бан", "complaint.admin", topics={"admin"})
    assert d.priority=="p2" and d.needs_human
    assert "human_escalation=mandatory_sensitive_topic" in d.reasons


def test_uncertain_model_cannot_change_floor_but_confident_model_may_raise():
    assert decide_priority("question", "report.question",topics=set(),learned_priority="p0").priority=="p3"
    assert decide_priority("question", "report.question",topics=set(),learned_priority="p0",learned_uncertain=False).priority=="p0"
    with pytest.raises(ValueError): decide_priority("", "report.help",topics=set(),learned_priority="p4")


@pytest.mark.parametrize("text",["массовый баг", "угрожает найти реальный адрес"])
def test_taxonomy_critical_floor_is_independent_of_category(text):
    assert priority_floor(text,set())=="p0"
