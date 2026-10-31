"""Invented language/noise regressions; no official ticket text, model or GPU."""
import pytest

from support_copilot.text import analyze_language, normalize_retrieval_query


@pytest.mark.parametrize("text,language", [
    ("Помогите", "ru"), ("помагите", "ru"), ("Игрок оскорбил меня", "ru"),
    ("Уточните время и обстоятельства обращения.", "ru"),
    ("Көмек", "kk"), ("комек", "kk"), ("Ойыншы мені балағаттады", "kk"),
    ("Жағдайды қолдау қызметкері тексеруі қажет.", "kk"),
    ("Yordam", "uz"), ("yordm", "uz"), ("Bugun to‘lov qildim ammo bonus kelmadi", "uz"),
    ("Parolni yubormang, yordam xodimiga vaqtni yozing.", "uz"),
    ("Help", "en"), ("hlep", "en"), ("Please help, my account was stolen.", "en"),
    ("Please specify the player name and circumstances.", "en"),
    ("Hola, necesito ayuda con mi cuenta", "es"),
    ("Olá, preciso de ajuda com minha conta", "pt"),
])
def test_language_from_meaningful_words(text, language):
    result = analyze_language(text)
    assert result["language"] == language
    assert language in result["candidates"]
    assert "language_heuristic_not_calibrated" in result["signals"]


@pytest.mark.parametrize("text", ["", "...", "123", "🙂🙂", "xyz", "фыва", "ru | kk | uz | en | und",
                                  "rules:30:2.1", "Arman_Player", "https://help.example/account"])
def test_unknown_short_and_metadata_do_not_become_language_from_server_hint(text):
    result = analyze_language(text, "kk")
    assert result["language"] == "und"
    assert result["ambiguous"] and not result["candidates"]


def test_one_kazakh_character_does_not_override_english_sentence():
    result = analyze_language("Please help, my account was stolen and I cannot access it. қ")
    assert result["language"] == "en"
    assert not result["mixed"]


def test_mixed_message_keeps_primary_language_and_secondary_evidence():
    result = analyze_language("Please help my account was stolen. Мені қорлады.")
    assert result["language"] == "en"
    assert result["mixed"] and set(result["candidates"]) == {"en", "kk"}


def test_equal_short_languages_need_clarification_or_weak_hint_tie_break():
    result = analyze_language("Помогите көмектесіңіз")
    assert result["language"] == "und" and result["mixed"] and result["ambiguous"]
    hinted = analyze_language("Помогите көмектесіңіз", "kk")
    assert hinted["language"] == "kk" and hinted["ambiguous"]
    assert "server_hint_breaks_evidence_tie" in hinted["signals"]


def test_legacy_restoration_and_detection_preserve_full_input_view():
    text = 'Маґан кјмек керек, ник: Player_Test, https://example.org. Ереже: «Запрещено использовать программы». [rules:30:5.2]'
    result = analyze_language(text, "kk", "kz")
    assert result["language"] == "kk"
    assert result["normalized_text"] == text.replace("ґ", "ғ").replace("ј", "ө")
    assert "Player_Test" in result["normalized_text"] and "https://example.org" in result["normalized_text"]
    assert "Запрещено" in result["normalized_text"] and "rules:30:5.2" in result["normalized_text"]
    assert result["normalization_flags"] == ["legacy_kazakh_restored"]


def test_quoted_russian_source_does_not_dominate_uzbek_question():
    result = analyze_language('Iltimos yordam bering. Правило: «Запрещено использовать программы и отправлять сообщения игрокам».')
    assert result["language"] == "uz"
    assert not result["mixed"]


def test_canonical_ids_and_named_nicknames_are_detection_metadata():
    result = analyze_language("yordam rules:30:2.1 ник: Help_Player")
    assert result["language"] == "uz" and result["candidates"] == ["uz"]


def test_exact_typo_query_view_preserves_case_and_negation():
    query, flags = normalize_retrieval_query("НЕ оскарбил. ЧИТЕРР не нужен, Акаунт не взломан.")
    assert query == "НЕ оскорбил. ЧИТЕР не нужен, Аккаунт не взломан."
    assert flags == ["domain_typo_query_view"]


def test_domain_typo_stems_keep_inflections_and_do_not_change_raw_input():
    text = "Не оскарбляли радителей. Ойыншы балагаттамайды, корламады."
    query, flags = normalize_retrieval_query(text)
    assert query == "Не оскорбляли родителей. Ойыншы балағаттамайды, қорламады."
    assert text == "Не оскарбляли радителей. Ойыншы балагаттамайды, корламады."
    assert flags == ["domain_typo_query_view"]


def test_query_correction_protects_names_handles_urls_refs_and_numbers():
    text = "ник: Акаунт, Hlep_Player @paymnt https://example.org/оскарбил rules:30:2.1 2.1: Оскарбил меня"
    query, flags = normalize_retrieval_query(text)
    assert query == text.replace("Оскарбил меня", "Оскорбил меня")
    assert flags


@pytest.mark.parametrize("text", ["Оскар пришёл", "аккаунт не продаю", "haqorat emas", "чистое небо", "my account", "ЧиТоР"])
def test_query_correction_does_not_fuzzily_change_other_words(text):
    assert normalize_retrieval_query(text) == (text, [])
