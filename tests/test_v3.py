"""Tests for the v3 pipeline. Integration tests run only when local data + model exist."""
from pathlib import Path

import pytest

from support_copilot.v3 import lang as langmod
from support_copilot.v3.labels import label_complaint, label_report
from support_copilot.v3.rulekb import parse_refs
from support_copilot.v3.templates import T, render

ROOT = Path(__file__).resolve().parents[1]
DATA, MODEL = ROOT / ".private/data", ROOT / ".private/v3"
KNOWN = {"rules:30:2.1", "rules:32:1.1", "rules:31:1.2.7", "rules:30:5.2"}


def test_language_never_und():
    assert langmod.detect("+", None, "uz")[0] == "uz"
    assert langmod.detect("оск родни", "und", "ru")[0] == "ru"
    assert langmod.detect("Meni haqorat qildi", "lat", "uz")[0] == "uz"
    assert langmod.detect("<url> 5.2 OP", "lat", "ru")[0] == "ru"
    assert langmod.detect("маған бан берді", None, "kk")[0] == "kk"
    assert langmod.detect("поможіть мені будь ласка", "kk", "ru")[0] == "ru"   # Ukrainian is answered in Russian
    assert langmod.detect("me banearon sin razón, quiero desban", None, "ru")[0] == "es"
    assert langmod.detect("roubaram minha conta, não consigo entrar", "pt", "ru")[0] == "pt"
    assert langmod.detect("menga 7 kun ban berdi, nohaq", "pt", "uz")[0] == "uz"


def test_section_aware_refs():
    assert parse_refs("1.1 GZ и 2.1 ОП", KNOWN) == ["rules:32:1.1", "rules:30:2.1"]
    assert parse_refs("отклонена по 1.2.7 правил подачи жалоб", KNOWN) == ["rules:31:1.2.7"]
    assert parse_refs("9.9", KNOWN) == []


def test_weak_labels_use_admin_outcome():
    t = {"kind": "complaint", "verdict": "approved", "violation": "5.2 ОП", "rule_refs_raw": "",
         "images_count": 1, "messages": [{"role": "player", "text": "он летает"}, {"role": "admin", "text": "бан 5.2 ОП"}]}
    assert label_complaint(t)[0] == "complaint.cheat"
    appeal = {**t, "messages": [{"role": "player", "text": "не согласен с наказанием, снимите бан"}]}
    assert label_complaint(appeal)[0] == "complaint.admin"
    assert label_report({"kind": "report", "messages": [{"role": "player", "text": "+"}]})[0] == "report.offtopic"


@pytest.mark.parametrize("lang", ["ru", "kk", "uz", "en", "es", "pt"])
def test_templates_complete_and_no_promises(lang):
    from support_copilot.v3.pipeline import PROMISE
    assert set(T[lang]) == set(T["ru"])
    for key in T[lang]:
        if key == "on_nick":
            continue
        text = render(key, lang, rule="X", rule2="Y", nick="Arman_Petrov", sanction="MUTE")
        assert "{" not in text and not PROMISE.search(text), (lang, key, text)


needs_model = pytest.mark.skipif(not (MODEL / "manifest.json").exists() or not (DATA / "rules.json").exists(),
                                 reason="local v3 model/data not present")


@pytest.fixture(scope="module")
def engine():
    from support_copilot.v3.pipeline import TriageV3
    return TriageV3(DATA, MODEL).load()


@needs_model
def test_contract_example(engine):
    out = engine.triage({"kind": "complaint", "images_count": 2, "messages": [
        {"role": "player", "text": "Arman_Petrov оскорблял меня и мою семью в чате, скрины есть, 2.1 ОП", "attachments": 2}]})
    assert out["lang"] == "ru" and out["category"] == "complaint.insult" and out["priority"] == "p2"
    assert out["extracted"]["nickname"] == "Arman_Petrov" and out["extracted"]["rule_refs"] == ["2.1"]
    assert out["draft_reply"]["cited"] == ["2.1"] and out["needs_human"] is True
    assert set(out) == {"lang", "category", "priority", "is_spam_or_duplicate", "extracted", "draft_reply",
                        "needs_human", "confidence", "reasons"}


@needs_model
@pytest.mark.parametrize("text,cat", [("взломали аккаунт, пропала машина", "report.account"),
                                      ("оплатил гк, не пришло", "report.payment")])
def test_sensitive_reports_go_to_human(engine, text, cat):
    out = engine.triage({"kind": "report", "messages": [{"role": "player", "text": text}]})
    assert out["category"] == cat and out["priority"] == "p0" and out["needs_human"] is True


@needs_model
def test_appeal_never_promises_unban(engine):
    out = engine.triage({"kind": "complaint", "messages": [{"role": "player", "text": "не согласен с баном, разбаньте пожалуйста"}]})
    assert out["category"] == "complaint.admin" and out["needs_human"] is True
    assert "разбан" not in out["draft_reply"]["text"].lower()


@needs_model
def test_duplicates_within_hour(engine):
    req = {"kind": "complaint", "reporter_hash": "r1", "reported_hash": "x1", "created_at": "2026-10-01T10:00:00",
           "messages": [{"role": "player", "text": "дм у банка"}]}
    assert engine.triage(req)["is_spam_or_duplicate"] is False
    assert engine.triage({**req, "created_at": "2026-10-01T10:20:00"})["is_spam_or_duplicate"] is True


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def polish(self, **kw):
        from support_copilot.v3.llm import REF_IN_TEXT
        allowed = {d.ref: d.canonical_id for d in kw["rules"]}
        refs = REF_IN_TEXT.findall(self.reply)
        if any(r not in allowed for r in refs):
            return None
        return self.reply, {allowed[r] for r in refs}


@needs_model
@pytest.mark.parametrize("reply,accepted", [
    ("Здравствуйте! Жалоба принята, по п. 2.1 общих правил администратор проверит скриншоты.", True),
    ("Здравствуйте! Нарушен п. 9.9, игрок нарушил правила.", False),      # invented rule
    ("Здравствуйте! Игрок получит бан на 7 дней по п. 2.1.", False),      # promises punishment
])
def test_llm_guard(engine, reply, accepted):
    engine.llm = FakeLLM(reply)
    try:
        out = engine.triage({"kind": "complaint", "images_count": 1, "messages": [{"role": "player", "text": "оск родни 2.1"}]})
    finally:
        engine.llm = None
    assert (out["draft_reply"]["text"] == reply) is accepted
    assert all(c in {"2.1", "1.3.1"} for c in out["draft_reply"]["cited"])


def test_sanctions_parsed_literally():
    from support_copilot.coretypes import RetrievedRule
    from support_copilot.v3.sanctions import sanction_text
    doc = RetrievedRule("rules:30:2.1", 30, "t", "2.1", "Запрещено... | Оскорбление игрока: MUTE - от 15 до 30 минут. | Упоминание родных: MUTE - от 90 до 180 минут.")
    assert sanction_text(doc, "ru") == "оскорбление игрока — MUTE от 15 до 30 мин; упоминание родных — MUTE от 90 до 180 мин"
    assert sanction_text(doc, "uz") == "o‘yinchini haqorat qilish — MUTE 15–30 daqiqa; qarindoshlarni tilga olish — MUTE 90–180 daqiqa"
    weird = RetrievedRule("rules:30:9.9", 30, "t", "9.9", "x | Нарушение: Блокировка персонажа - до 31 дня с обнулением имущества.")
    assert sanction_text(weird, "ru") == ""   # not fully understood -> nothing, never a guess


def test_server_validation():
    from support_copilot.v3.server import validate
    assert validate({"kind": "report", "messages": [{"role": "player", "text": "help"}]}) is None
    assert validate({"kind": "x", "messages": []})
    assert validate({"kind": "report", "messages": [{"role": "admin", "text": "hi"}]})


@needs_model
@pytest.mark.parametrize("kind,text,lang,cat", [
    ("complaint", "hola, me insultó a mi madre en el chat", "es", "complaint.insult"),
    ("report", "olá, estou preso na textura, me ajudem", "pt", "report.help"),
    ("report", "маған көмектесіңізші, тұрып қалдым", "kk", "report.help"),
    ("report", "купив донат але він не з'явився в інвентарі", "ru", "report.payment"),
])
def test_six_languages(engine, kind, text, lang, cat):
    out = engine.triage({"kind": kind, "images_count": 1, "messages": [{"role": "player", "text": text}]})
    assert out["lang"] == lang and out["category"] == cat and out["draft_reply"]["lang"] == lang


@needs_model
def test_real_life_threat_is_p0(engine):
    out = engine.triage({"kind": "complaint", "images_count": 1, "messages": [{"role": "player", "text": "оскорбление родных, угроза жизни"}]})
    assert out["priority"] == "p0" and out["needs_human"] is True
