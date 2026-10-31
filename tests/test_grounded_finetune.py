"""Invented rules only; source support and target leakage boundaries."""
import importlib.util
from pathlib import Path

import pytest
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from support_copilot.coretypes import RetrievedRule

SPEC = importlib.util.spec_from_file_location("grounded_data", Path(__file__).resolve().parents[1]/"scripts/prepare_grounded_finetune.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def test_internal_pipe_is_not_a_sanction_boundary():
    assert data.conduct("Ограничение [Area | A]. Полное условие. | MUTE -10") == "Ограничение [Area | A]. Полное условие."


def item():
    return {"id": "invented", "question": "Объясните выдуманное правило с условием", "category": "report.question",
        "canonical_id": "rules:1:1.1", "quote": "Выдуманное действие разрешено только при наличии условия."}


RULES = {"rules:1:1.1": RetrievedRule("rules:1:1.1", 1, "Invented", "1.1",
    "Выдуманное действие разрешено только при наличии условия. | WARN")}


def test_target_is_exact_rule_quote_and_not_a_player_guilt_verdict():
    row, meta = data.make_example(item(), RULES)
    assert item()["quote"] in row["messages"][-1]["content"]
    assert "[rules:1:1.1]" in row["messages"][-1]["content"]
    assert "WARN" not in str(row) and meta["source_quote_verified_exact"]
    assert meta["human_gold"] is False


@pytest.mark.parametrize("field,value", [("quote", "Действие разрешено всегда, условия не нужны."),
    ("assistant", "Непроверенный ответ"), ("verdict", "guilty"), ("admin_reply", "secret")])
def test_unsupported_or_historical_targets_are_rejected(field, value):
    case = item(); case[field] = value
    with pytest.raises(ValueError):
        data.make_example(case, RULES)


def test_missing_source_cannot_be_filled_with_invented_faq():
    case = {"id": "intake", "question": "Оплатил валюту и не получил бонус", "category": "report.payment", "intake_topic": "payment"}
    row, meta = data.make_example(case, RULES)
    assert meta["source_id"] is None and "rules:" not in row["messages"][-1]["content"]
    assert "Не присылайте пароль" in row["messages"][-1]["content"]


@pytest.mark.parametrize("value", ["", "ru,ru", "ru,und", "xx"])
def test_invalid_adapter_language_policy_fails_closed(monkeypatch, value):
    from support_copilot.config import Settings
    monkeypatch.setenv("COPILOT_ADAPTER_LANGUAGES", value)
    with pytest.raises(ValueError):
        Settings.from_env()


def test_ru_only_adapter_is_explicitly_disabled_for_other_languages():
    from support_copilot.config import Settings
    from support_copilot.engine import TriageEngine
    from support_copilot.schemas import TriageRequest
    rule = RetrievedRule("rules:30:2.1", 30, "Invented", "2.1", "Invented source")
    class Generator:
        disabled = False
        observations = []
        @contextmanager
        def disable_adapter(self):
            self.disabled = True
            try:
                yield
            finally:
                self.disabled = False
        def generate(self, text, lang, rules, category):
            self.observations.append((lang, self.disabled))
            return {"ru": "Уточните время и сервер для проверки [rules:30:2.1].",
                    "en": "Please specify the time and server for review [rules:30:2.1]."}[lang]
    engine = TriageEngine(replace(Settings(), use_adapter=True, adapter_languages=("ru",)))
    engine.ready = True
    engine.retriever = SimpleNamespace(search=lambda *a, **kw: [rule])
    engine.generator = Generator()
    for text in ("Игрок оскорбил меня", "The player insulted my family"):
        engine.triage(TriageRequest.model_validate({"kind": "complaint", "messages": [{"role": "player", "text": text}]}))
    assert engine.generator.observations == [("ru", False), ("en", True)]
    assert engine.generator.disabled is False
