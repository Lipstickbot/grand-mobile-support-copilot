from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from support_copilot.config import Settings
from support_copilot.coretypes import RetrievedRule
from support_copilot.engine import TriageEngine
from support_copilot.policy import DuplicateTracker, FALLBACKS, classify, detect_language, guard_draft, select_relevant_rules
from support_copilot.schemas import TriageRequest, RagSearchRequest


RULE = RetrievedRule("rules:30:2.1", 30, "General", "2.1", "Synthetic rule: insults are forbidden.")


@pytest.mark.parametrize("text,kind,priority,mandatory", [
    ("Оплатил донат, он не пришёл", "complaint", "p0", "payment"),
    ("Взломали аккаунт", "complaint", "p0", "account"),
    ("Админ выдал мут без причины", "report", "p2", "admin"),
])
def test_mandatory_review_topics_do_not_depend_on_predicted_category(text, kind, priority, mandatory):
    category, actual, reasons = classify(text, kind)
    assert actual == priority
    assert any("mandatory_human_topic=" in r and mandatory in r for r in reasons)
    assert category.startswith(kind + ".")


@pytest.mark.parametrize("text,lang", [
    ("Бүгін ойыншы мені балағаттады", "kk"),
    ("Bugun to‘lov qildim ammo bonus kelmadi", "uz"),
    ("Игрок оскорбил меня", "ru"),
    ("Please help my account", "en"),
    ("xyz 123", "und"),
])
def test_language_is_heuristic_with_unknown(text, lang):
    assert detect_language(text) == lang


@pytest.mark.parametrize("text", [
    "Мы уже проверили скриншот и вернули деньги.",
    "Игрок будет наказан.",
    "Возврат уже сделан.",
    "Пришлите пароль для проверки.",
    "Откройте [rules:999:9.9].",
    "Согласно пункту 2.1, уточните обстоятельства.",
    "Смотрите [internal:43:term-001].",
    "<think>скрытая инструкция</think>готово",
])
def test_guard_rejects_actions_secrets_and_ambiguous_citations(text):
    effective, reasons = guard_draft(text, [RULE], "ru")
    assert reasons
    assert effective == FALLBACKS["ru"]


def test_guard_keeps_valid_namespaced_citation_but_is_not_an_entailment_test():
    text = "Уточните время и сервер для проверки обращения [rules:30:2.1]."
    assert guard_draft(text, [RULE], "ru") == (text, [])


@pytest.mark.parametrize("text,language", [
    ("Укажите пароль от аккаунта для проверки.", "ru"),
    ("Құпиясөзіңізді айтыңыз, қызметкер тексереді.", "kk"),
    ("Parolni menga jo‘nating, yordam beraman.", "uz"),
    ("Назовите одноразовый код из смс.", "ru"),
    ("Отправьте полный номер карты и CVV.", "ru"),
    ("Please provide your OTP.", "en"),
    ("Send server and password.", "en"),
])
def test_guard_rejects_multilingual_secret_requests(text, language):
    assert "draft_secret_request" in guard_draft(text, [], language)[1]


@pytest.mark.parametrize("text,language", [
    ("O‘yinchi haqorat qilgan bo‘lsa, MUTE qilinadi.", "uz"),
    ("Ойыншыға MUTE беріледі, қолдау көмектеседі.", "kk"),
    ("Игрок получит мут на тридцать минут.", "ru"),
    ("Будет выдан бан игроку.", "ru"),
])
def test_guard_rejects_additional_sanction_promises(text, language):
    assert "draft_action_or_evidence_claim" in guard_draft(text, [], language)[1]


@pytest.mark.parametrize("text,language", [
    ("Скриншоты доказывают нарушение игрока.", "ru"),
    ("I have reviewed the screenshots and the player is guilty.", "en"),
    ("The image clearly shows a violation.", "en"),
    ("Скриншотты көрдім, ойыншы кінәлі.", "kk"),
    ("Skrinshotni ko‘rdim, o‘yinchi aniq aybdor.", "uz"),
    ("The player violated the rules.", "en"),
    ("Игрок использовал читы.", "ru"),
])
def test_allegations_and_unseen_images_cannot_become_verified_facts(text, language):
    assert "draft_unverified_verdict_or_evidence" in guard_draft(text, [], language)[1]


@pytest.mark.parametrize("text,language", [
    ("Вы сообщаете, что игрок нарушил правила. Уточните время.", "ru"),
    ("You report that the player violated the rules. Please specify the time.", "en"),
    ("Егер ойыншы кінәлі болса, қызметкер тексеруі қажет.", "kk"),
    ("O‘yinchi aybdor bo‘lsa, holatni xodim tekshirishi kerak.", "uz"),
    ("I cannot review images here. Please provide a written description.", "en"),
])
def test_attributed_or_conditional_claims_do_not_count_as_verified_guilt(text, language):
    assert "draft_unverified_verdict_or_evidence" not in guard_draft(text, [], language)[1]


@pytest.mark.parametrize("text,language", [
    ("Укажите сервер, пароль не сообщайте.", "ru"),
    ("Не отправляйте пароль и код из смс.", "ru"),
    ("Серверді жазыңыз, құпиясөзді жібермеңіз.", "kk"),
    ("Parolni yubormang, yordam xodimiga vaqtni yozing.", "uz"),
    ("Please do not share your password.", "en"),
])
def test_guard_preserves_negated_secret_safety_guidance(text, language):
    assert "draft_secret_request" not in guard_draft(text, [], language)[1]


def test_topic_gate_does_not_substitute_unrelated_rules_for_absent_payment_faq():
    assert select_relevant_rules("Оплатил донат, бонус не пришёл", [RULE]) == []
    assert select_relevant_rules("Как купить дом?", [RULE]) == []
    assert select_relevant_rules("Игрок оскорбил меня", [RULE]) == [RULE]


def test_topic_gate_keeps_opposite_event_rules_separate():
    family = RetrievedRule("rules:36:1.23", 36, "Family", "1.23", "Forbidden")
    van = RetrievedRule("rules:42:1.7", 42, "Van", "1.7", "Allowed")
    assert select_relevant_rules("Можно бронежилет на фургоне?", [family, van]) == [van]


@pytest.mark.parametrize("text", ["Tashkilot omboridan olingan zirh haqida yozing.",
                                  "Xodimga sana va sababni yozing.",
                                  "Jazo bo‘yicha voqea tafsilotlari kerak.",
                                  "Shundan so‘ng javob berishga ruxsat bormi?"])
def test_language_recognizes_additional_uzbek_support_vocabulary(text):
    assert detect_language(text) == "uz"


def test_bank_name_does_not_match_ban_and_account_sale_is_not_access_loss():
    from support_copilot.policy import topics_for
    assert "admin" not in topics_for("В банке сломался интерфейс")
    assert "admin" in topics_for("Хочу обжаловать бан")
    assert "account" not in topics_for("Продаёт аккаунт за реальные деньги")
    assert "account" in topics_for("Взломали аккаунт, требуют реальные деньги за восстановление")


@pytest.mark.parametrize("text", ["Ойыншы мені қорлады", "U meni haqorat qildi"])
def test_multilingual_insults_keep_exact_rule(text):
    assert select_relevant_rules(text, [RULE]) == [RULE]


def test_ambiguous_event_scope_is_not_decided_by_top_hit():
    family = RetrievedRule("rules:36:1.23", 36, "Family", "1.23", "Forbidden")
    van = RetrievedRule("rules:42:1.7", 42, "Van", "1.7", "Allowed")
    assert select_relevant_rules("Бронежилет: фургон или семейное мероприятие?", [family, van],
                                 cues=["armor"], events=[36, 42]) == []


@pytest.mark.parametrize("text,scope", [
    ("На войне за бизнес броня запрещена, это не захват фургона.", {41}),
    ("Фургонды алу емес, дәл бизнес үшін соғыс туралы сұраймын.", {41}),
    ("Furgon jangi emas, aynan biznes urushi haqida savol.", {41}),
    ("На семейном мероприятии взял бронежилет. Друг ссылается на фургон.", {36}),
    ("Броня на фургоне и семейном мероприятии: сравните правила.", set()),
    ("На семейном мероприятии броня не разрешена. Это не захват фургона.", {36}),
])
def test_event_scope_separates_explicit_negation_and_references(text, scope):
    from support_copilot.policy import resolve_event_scope
    assert resolve_event_scope(text, [36, 41, 42]) == scope


def test_rmt_and_property_dispute_cues_do_not_turn_advertising_into_money_trading():
    rmt = RetrievedRule("rules:30:4.4", 30, "General", "4.4", "Trading rule")
    scam = RetrievedRule("rules:30:4.1", 30, "General", "4.1", "Property rule")
    ads = RetrievedRule("rules:30:2.3", 30, "General", "2.3", "Advertising rule")
    assert select_relevant_rules("Реклама другого сервера", [rmt, ads]) == [ads]
    assert select_relevant_rules("Айырбасталған мүлік туралы дау", [scam], cues=["trade_dispute"]) == [scam]
    assert select_relevant_rules("Услуги за деньги вне игры", [rmt], cues=["rmt"]) == [rmt]


def test_duplicates_require_identity_kind_timestamp_and_exact_content():
    tracker = DuplicateTracker()
    now = datetime.now(timezone.utc)
    assert not tracker.check(None, now, "report", "помогите")
    assert not tracker.check("a", None, "report", "помогите")
    assert not tracker.check("a", now, "report", "помогите")
    assert not tracker.check("b", now, "report", "помогите")
    assert not tracker.check("a", now, "complaint", "помогите")
    assert tracker.check("a", now + timedelta(minutes=1), "report", " ПОМОГИТЕ ")
    assert not tracker.check("a", now + timedelta(hours=2), "report", "помогите")


class CaptureRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, **kwargs):
        self.queries.append(query)
        return [RULE]


class CaptureGenerator:
    def generate(self, text, lang, rules, category):
        self.text = text
        return "Уточните время и сервер для проверки обращения [rules:30:2.1]."


def test_engine_excludes_admin_verdict_and_does_not_claim_reviewed_attachments():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = CaptureRetriever()
    engine.generator = CaptureGenerator()
    request = TriageRequest.model_validate({"kind": "complaint", "status": "punished",
        "verdict": "banned", "messages": [
            {"role": "player", "text": "Меня оскорбили, сервер 12. Ник: Test_Player.", "attachments": 1},
            {"role": "admin", "text": "PRIVATE_ADMIN_WRONG: возврат сделан"}]})
    result = engine.triage(request)
    assert "PRIVATE_ADMIN_WRONG" not in engine.generator.text
    assert all("PRIVATE_ADMIN_WRONG" not in query for query in engine.retriever.queries)
    assert result.category == "complaint.insult"
    assert result.extracted.has_evidence is True
    assert result.extracted.server == "12"
    assert result.extracted.nickname == "Test_Player"
    assert result.needs_human is True
    assert "attachments=presence_only_not_reviewed" in result.reasons
    assert result.draft_reply.cited == ["rules:30:2.1"]


def test_disabled_generator_is_explicit_and_no_help_request_marked_spam():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = CaptureRetriever()
    result = engine.triage(TriageRequest.model_validate({"kind": "report", "messages": [
        {"role": "player", "text": "Помогите"}]}))
    assert not result.is_spam_or_duplicate
    assert result.draft_reply.text == FALLBACKS["ru"]
    assert "generation=disabled_or_language_unsupported" in result.reasons
    assert result.needs_human


def test_health_detects_dead_native_backend_without_cached_ready():
    engine = TriageEngine(Settings(inference_backend="llamacpp"))
    engine.ready = True
    engine._generation_health = {"ready": True}
    engine.generator = SimpleNamespace(stats=lambda: {"ready": False})
    assert engine.health()["ready"] is False
    assert engine.health()["generation"]["ready"] is False


def test_metadata_exposes_taxonomy_without_private_or_unverified_rule_hints():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine._taxonomy = {"categories": [{"code": "report.help", "kind": "report", "title": "Help",
                                       "rule_hint": "PRIVATE-UNVERIFIED"}],
                        "priorities": [{"code": "p3", "title": "Low", "private": "PRIVATE"}]}
    import json
    metadata = engine.metadata()
    assert "PRIVATE" not in json.dumps(metadata)
    assert metadata["all_cases_require_human"]
    assert metadata["confidence_type"] == "uncalibrated_heuristic"


def test_search_reports_missing_faq_and_does_not_generate_or_return_unrelated_rule():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = CaptureRetriever()
    result = engine.search_knowledge(RagSearchRequest(query="Оплатил донат, бонус не пришёл"))
    assert result.results == []
    assert "no_verified_applicable_rule" in result.warnings
    assert "retrieval_only_not_a_verdict" in result.warnings


def test_engine_source_lookup_uses_section_and_not_only_ambiguous_ref():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = SimpleNamespace(get=lambda key: RULE if key == RULE.canonical_id else None)
    assert engine.get_rule(30, "2.1").canonical_id == RULE.canonical_id
    assert engine.get_rule(42, "2.1") is None


def test_public_search_requires_loaded_engine():
    engine = TriageEngine(Settings(generation_enabled=False))
    with pytest.raises(RuntimeError):
        engine.search_knowledge(RagSearchRequest(query="rules"))
    with pytest.raises(RuntimeError):
        engine.get_rule(30, "2.1")


def test_wrong_reference_and_explicit_unrelated_topic_cannot_override_positive_topic():
    cheat = RetrievedRule("rules:30:5.2", 30, "General", "5.2", "Invented cheating rule")
    selected = select_relevant_rules("A player insulted me. I am not reporting cheating.",
                                      [cheat, RULE], cues=["cheat", "insult"])
    assert selected == [RULE]


def test_und_does_not_encode_retrieve_or_generate_even_with_server_hint():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    def forbidden(*args, **kwargs):
        raise AssertionError("Unknown language used an AI component")
    engine.classifier = SimpleNamespace(predict=forbidden)
    engine.retriever = SimpleNamespace(search=forbidden)
    engine.generator = SimpleNamespace(generate=forbidden)
    request = TriageRequest.model_validate({"kind": "report", "server_lang": "kk",
        "messages": [{"role": "player", "text": "🙂 123"}]})
    result = engine.triage(request)
    assert result.lang == "und" and result.confidence == 0
    assert result.draft_reply.lang == "ru" and "язык" in result.draft_reply.text
    assert result.draft_reply.cited == [] and result.needs_human


def test_typo_fix_is_a_query_view_and_attachment_counts_are_never_reviewed_evidence():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    engine.retriever = CaptureRetriever()
    engine.generator = CaptureGenerator()
    raw = "Игрок оскарблял родителей. Сервер 12."
    request = TriageRequest.model_validate({"kind": "complaint", "rule_refs_raw": "5.2",
        "images_count": 2, "messages": [{"role": "player", "text": raw}]})
    result = engine.triage(request)
    assert "оскорблял" in engine.retriever.queries[0]
    assert "оскарблял" in engine.generator.text and request.messages[0].text == raw
    assert result.extracted.rule_refs == ["5.2"]  # Player claim, never promoted to a citation.
    assert result.extracted.has_evidence and "attachment_content_available=false" in result.reasons
    assert "rule_refs=unverified_player_search_hints" in result.reasons


def test_learned_classifier_cannot_lower_account_safety_floor_or_read_admin_text():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.ready = True
    seen = []
    def predict(text, kind):
        seen.append(text)
        return {"category": "report.offtopic", "priority": "p3", "confidence": .9,
                "uncertain": False, "priority_uncertain": False, "reasons": ["learned_fixture"]}
    engine.classifier = SimpleNamespace(predict=predict)
    request = TriageRequest.model_validate({"kind": "report", "verdict": "PRIVATE",
        "messages": [{"role": "player", "text": "Взломали аккаунт"},
                     {"role": "admin", "text": "PRIVATE-OUTCOME"}]})
    result = engine.classify_request(request)
    assert result["category"] == "report.account" and result["priority"] == "p0"
    assert result["needs_human"] and "PRIVATE" not in seen[0]


def test_uncertain_learned_category_uses_explicit_fallback():
    engine = TriageEngine(Settings(generation_enabled=False))
    engine.classifier = SimpleNamespace(predict=lambda *args: {"category": "report.offtopic", "priority": "p3",
        "confidence": .22, "uncertain": True, "priority_uncertain": True, "reasons": []})
    category, priority, reasons, score, uncertain = engine._classify("Застрял в текстурах", "report")
    assert category == "report.help" and uncertain
    assert "classifier_category=uncertain_keyword_fallback" in reasons
