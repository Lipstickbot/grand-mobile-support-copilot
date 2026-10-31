"""Regression contracts for observed citation, sanction and no-source failures."""
from types import SimpleNamespace

import pytest

from support_copilot.config import Settings
from support_copilot.coretypes import RetrievedRule
from support_copilot.engine import TriageEngine
from support_copilot.intake import intake_reply
from support_copilot.policy import guard_draft, select_relevant_rules
from support_copilot.schemas import TriageRequest


RULE = RetrievedRule("rules:30:2.1", 30, "Invented", "2.1", "Invented rule: insults prohibited.")


@pytest.mark.parametrize("marker", ["[rule_1234]", "[rule-12345]", "[rule:12345]",
    "[rule:payment_confirmation]", "[complaint.cheat.001]", "[12345]",
    "[rules:abc:xyz]", "[правило_123]", "[пункт 1.2]"])
def test_unknown_authority_like_markers_are_rejected_even_without_cited_ids(marker):
    text = "Уточните время и сервер для проверки. " + marker
    assert "draft_noncanonical_citation" in guard_draft(text, [], "ru")[1]


def test_bracketed_nick_and_prose_are_not_official_citations():
    text = "Уточните время события с игроком [Test_Name]. [предоставьте описание]"
    assert "draft_noncanonical_citation" not in guard_draft(text, [], "ru")[1]


def test_retrieved_rules_do_not_license_an_uncited_permission():
    issues = guard_draft("Бронежилет разрешён при любых обстоятельствах.", [RULE], "ru")[1]
    assert "draft_missing_rule_citation" in issues


@pytest.mark.parametrize("text,lang", [
    ("The offending player should be muted for90-180 minutes.", "en"),
    ("The player must be banned.", "en"),
    ("Игроку следует выдать мут.", "ru"),
    ("По этому правилу бан3–10дней.", "ru"),
    ("Ойыншыға бан3күн берілуі керек.", "kk"),
    ("O‘yinchiga ban3kun berish kerak.", "uz"),
])
def test_sanction_assignment_and_durations_are_left_to_staff(text, lang):
    # Missing spaces are deliberate: model outputs also produce compact forms.
    text = text.replace("for90", "for 90").replace("бан3", "бан 3").replace("ban3", "ban 3")
    assert "draft_sanction_recommendation" in guard_draft(text, [], lang)[1]


def test_general_staff_review_is_not_a_sanction_recommendation():
    text = "Возможное нарушение требует проверки; решение о наказании принимает сотрудник."
    assert "draft_sanction_recommendation" not in guard_draft(text, [], "ru")[1]


@pytest.mark.parametrize("text,lang", [
    ("Сообщите данные для входа, пожалуйста.", "ru"),
    ("Please provide login credentials for review.", "en"),
    ("Кіру деректерін жіберіңіз.", "kk"),
    ("Kirish ma’lumotlarini yuboring.", "uz"),
])
def test_vague_credential_bundle_requests_are_unsafe(text, lang):
    assert "draft_secret_request" in guard_draft(text, [], lang)[1]


@pytest.mark.parametrize("lang", ["ru", "kk", "uz", "en"])
@pytest.mark.parametrize("topic", ["payment", "account", "other"])
@pytest.mark.parametrize("attached", [False, True])
def test_intake_templates_pass_guard_without_fabricated_authority(lang, topic, attached):
    text = intake_reply(lang, {topic}, has_attachments=attached)
    assert guard_draft(text, [], lang) == (text, [])
    assert "[" not in text and "rules:" not in text


def test_question_repetition_is_rejected_without_rejecting_an_answer():
    question = "Мы обсуждаем не фургон, а семейное мероприятие, можно ли там надеть бронежилет?"
    assert "draft_question_echo" in guard_draft(question + " [rules:30:2.1]", [RULE], "ru", player_text=question)[1]
    answer = "По этому правилу оскорбления запрещены; ситуацию должен проверить сотрудник [rules:30:2.1]."
    assert "draft_question_echo" not in guard_draft(answer, [RULE], "ru", player_text=question)[1]


@pytest.mark.parametrize("text,topic", [
    ("Чужой человек получил доступ к моему персонажу", "account"),
    ("Бөгде адам менің кейіпкеріме қол жеткізіп алды", "account"),
    ("Begona odam mening qahramonimga kirish huquqini oldi", "account"),
    ("Valyuta uchun haq yubordim, lekin hisobga qo‘shilmadi", "payment"),
])
def test_critical_signal_overrides_confident_wrong_head(text, topic):
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.classifier = SimpleNamespace(predict=lambda *a: {"category": "report.offtopic", "priority": "p3",
        "uncertain": False, "priority_uncertain": False, "confidence": .9, "reasons": []})
    category, priority, *_ = engine._classify(text, "report")
    assert category == "report." + topic and priority == "p0"


def test_missing_faq_does_not_call_llm_or_ask_for_already_reported_image():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = SimpleNamespace(search=lambda *a, **kw: [RULE])
    def forbidden(*a, **kw):
        raise AssertionError("Absent FAQ must not be invented by the LLM")
    engine.generator = SimpleNamespace(generate=forbidden)
    result = engine.triage(TriageRequest.model_validate({"kind": "report", "images_count": 1,
        "messages": [{"role": "player", "text": "Оплатил валюту, бонус не пришёл"}]}))
    assert result.priority == "p0" and result.needs_human
    assert result.draft_reply.cited == []
    assert "generation=skipped_no_verified_rule" in result.reasons
    assert "draft=source_absent_intake_template" in result.reasons
    assert "содержимое здесь недоступно" in result.draft_reply.text
    assert "приложите" not in result.draft_reply.text


def test_english_event_permission_does_not_select_family_prohibition():
    van = RetrievedRule("rules:42:1.7", 42, "Van", "1.7", "Invented van rule")
    family = RetrievedRule("rules:36:1.23", 36, "Family", "1.23", "Invented family rule")
    assert select_relevant_rules("Can I wear body armor in the battle for the van?", [family, van]) == [van]


def test_confident_account_category_cannot_receive_background_priority():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.classifier = SimpleNamespace(predict=lambda *a: {"category": "report.account", "priority": "p3",
        "uncertain": False, "priority_uncertain": False, "confidence": .9, "reasons": []})
    category, priority, *_ = engine._classify("Персонаж исчез после смены входных данных", "report")
    assert category == "report.account" and priority == "p0"


@pytest.mark.parametrize("text", [
    "На сотрудника полиции напали в GZ. Можно ли открыть ответный огонь?",
    "Полиция қызметкеріне жасыл аймақта шабуыл жасалды, оқ ату шарты қандай?",
    "Politsiyachiga yashil hududda hujum qilindi, javoban o'q uzish mumkinmi?",
    "A police officer was attacked in the green zone. What about return fire?",
])
def test_gz_return_fire_does_not_include_unasked_entry_or_area_lists(text):
    combat = RetrievedRule("rules:32:1.1", 32, "GZ", "1.1", "Invented combat rule")
    entry = RetrievedRule("rules:32:1.5", 32, "GZ", "1.5", "Invented entry rule")
    areas = RetrievedRule("rules:32:2.1", 32, "GZ", "2.1", "Invented area list")
    assert select_relevant_rules(text, [entry, areas, combat]) == [combat]


@pytest.mark.parametrize("text", ["Соперник использует автокликер для игровых преимуществ.",
    "Қарсыласым макрос қолданып, автоматты түрде басып жүр.",
    "Raqib makros bilan avtomatik bosishni ishlatyapti.",
    "The opponent uses an autoclicker in the game."])
def test_macros_are_known_software_rule_queries_not_an_absent_faq(text):
    software = RetrievedRule("rules:30:5.2", 30, "Software", "5.2", "Invented macro rule")
    assert select_relevant_rules(text, [RULE, software]) == [software]


def test_gz_entry_during_shooting_is_not_misclassified_as_return_fire_only():
    combat = RetrievedRule("rules:32:1.1", 32, "GZ", "1.1", "Invented combat rule")
    entry = RetrievedRule("rules:32:1.5", 32, "GZ", "1.5", "Invented entry rule")
    selected = select_relevant_rules("Can I enter the green zone during a shootout?", [entry, combat])
    assert entry in selected
