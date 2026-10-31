"""Authored-intent candidate input boundary, no weights/network/GPU."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("head_candidate", Path(__file__).resolve().parents[1]/"scripts/train_classifier_candidate.py")
candidate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate)


def row():
    return {"text": "Я не могу войти в свой профиль", "kind": "report", "category": "report.account",
        "priority": "p0", "lang": "ru", "source": "authored_intent_or_general_rule_question"}


def test_authored_intent_labels_are_accepted():
    assert len(candidate.validate_rows([row()], ["report.account"])) == 1


def test_expected_language_does_not_leak_into_normalization_features():
    assert candidate.view({"text":"є", "lang":"kk"})==candidate.view({"text":"є", "lang":"ru"})=="є"
    assert candidate.view({"text":"є", "lang":"ru", "server_lang":"kk"})=="ә"


@pytest.mark.parametrize("key,value", [("source", "historical_player_weak_label"),
    ("admin_reply", "leaked outcome"), ("verdict", "guilty"),
    ("category", "complaint.account"), ("priority", "urgent")])
def test_outcome_or_keyword_teacher_data_cannot_enter_candidate(key, value):
    example = row(); example[key] = value
    with pytest.raises(ValueError):
        candidate.validate_rows([example], ["report.account"])


def test_normalized_duplicate_questions_are_rejected():
    a = row(); b = row(); b["text"] += "   "
    with pytest.raises(ValueError):
        candidate.validate_rows([a, b], ["report.account"])


def test_empty_training_corpus_is_not_a_model():
    with pytest.raises(ValueError):
        candidate.validate_rows([], ["report.account"])
