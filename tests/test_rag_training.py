"""Synthetic fixtures only; no official rule/ticket text is embedded in tests."""
import gzip
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

SPEC = importlib.util.spec_from_file_location("rag_training_tested", Path(__file__).resolve().parents[1] / "scripts/prepare_rag_training.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


def source():
    return {"rules": [{"id": 99, "category": "Synthetic fixture", "points": [
        {"ref": "1.1", "text": "Synthetic fixture rule about a fictional exercise. No real rule is represented."}]}]}


def recipe(topics=2):
    result = []
    for number in range(topics):
        refs = ["rules:99:1.1"] if number % 2 == 0 else []
        result.append({"id": f"topic{number}", "kind": "report", "category": f"fixture.{number}",
            "rule_ids": refs,
            "train": {lang: {"question": f"Unique synthetic {lang} training exercise {number}?",
                              "answer": "Synthetic support draft." + (" [rules:99:1.1]" if refs else "")}
                      for lang in DATA.LANGUAGES},
            "eval": {lang: {"question": f"A separate authored {lang} scenario asks about exercise {number}.",
                             "answer": "Independent synthetic reference." + (" [rules:99:1.1]" if refs else "")}
                     for lang in DATA.LANGUAGES}})
    return {"topics": result}


def fixed_counter(messages):
    return 100


def test_balanced_counts_independent_eval_and_plain_assistant_targets():
    result = DATA.build_dataset(recipe(12), DATA.catalog(source()), fixed_counter)
    assert len(result["train"]) == 288
    assert len(result["train108"]) == 108
    assert len(result["train64"]) == 64
    assert len(result["holdout"]) == len(result["evaluation"]) == 36
    metadata = {m["text_sha256"]: m for m in result["train_metadata"]}
    strata = {}
    for row in result["train108"]:
        text = json.loads(row["messages"][1]["content"])["player_text"]
        m = metadata[DATA.text_hash(text)]
        key = m["lang"], m["topic"]
        strata[key] = strata.get(key, 0) + 1
    assert len(strata) == 36 and set(strata.values()) == {3}
    train_keys = {m["text_sha256"] for m in result["train_metadata"]}
    holdout_keys = {m["text_sha256"] for m in result["holdout_metadata"]}
    assert train_keys.isdisjoint(holdout_keys)
    assert all(row["messages"][-1]["role"] == "assistant" and
               not row["messages"][-1]["content"].startswith("{") for row in result["train"])
    assert all(case["human_gold"] is False and case["expected_needs_human"] is True for case in result["evaluation"])


def test_deterministic_shuffle_and_source_overlap_is_rejected():
    first = DATA.build_dataset(recipe(), DATA.catalog(source()), fixed_counter, seed=2)
    second = DATA.build_dataset(recipe(), DATA.catalog(source()), fixed_counter, seed=2)
    assert DATA.jsonl(first["train"]) == DATA.jsonl(second["train"])
    text = json.loads(first["train"][0]["messages"][1]["content"])["player_text"]
    with pytest.raises(ValueError, match="overlaps"):
        DATA.build_dataset(recipe(), DATA.catalog(source()), fixed_counter, blocked={DATA.text_hash(text)})


def test_untrusted_control_tokens_are_data_and_missing_faq_has_no_context():
    result = DATA.build_dataset(recipe(), DATA.catalog(source()), fixed_counter)
    injections = []
    for row in result["train"]:
        payload = json.loads(row["messages"][1]["content"])
        assert "<|" not in row["messages"][1]["content"]
        if "<|" in payload["player_text"]:
            injections.append(row)
        if payload["category_hint"] == "fixture.1":
            assert payload["retrieved_passages"] == []
    assert len(injections) == 6


def test_unavailable_citation_and_target_truncation_are_failures():
    with pytest.raises(ValueError, match="unavailable"):
        DATA.build_row("question", "[rules:999:9.9]", "ru", "fixture", [], fixed_counter)
    with pytest.raises(ValueError, match="cannot fit"):
        DATA.build_row("question must remain complete", "answer remains complete", "ru", "fixture", [], lambda m: 513)


def test_only_passage_is_shortened_and_excerpt_marker_is_truthful():
    rule = DATA.RetrievedRule("rules:99:1.1", 99, "Fixture", "1.1", ("Synthetic complete sentence. " * 30))
    def counter(messages):
        return 300 + len(json.loads(messages[1]["content"])["retrieved_passages"][0]["text"])
    row, tokens = DATA.build_row("complete player question", "complete answer", "ru", "fixture", [rule], counter)
    payload = json.loads(row["messages"][1]["content"])
    assert tokens <= 512 and payload["player_text"] == "complete player question"
    assert row["messages"][-1]["content"] == "complete answer"
    assert payload["retrieved_passages"][0]["is_excerpt"] is True
    assert payload["retrieved_passages"][0]["text"].endswith(".")


def test_archive_exclusion_uses_player_only_not_admin_or_outcome(tmp_path):
    archive_path = tmp_path / "fixture.zip"
    train = {"messages": [{"role": "player", "text": "synthetic player"},
                           {"role": "admin", "text": "PRIVATE_OUTCOME"}], "verdict": "PRIVATE_OUTCOME"}
    valid = {"messages": [{"role": "player", "text": "synthetic excluded valid"}]}
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("handout/tickets_train.jsonl.gz", gzip.compress((json.dumps(train) + "\n").encode()))
        archive.writestr("handout/tickets_valid.jsonl", json.dumps(valid) + "\n")
    blocked, counts = DATA.archive_exclusions(archive_path)
    assert counts == {"train": 1, "valid_exclusion_only": 1}
    assert DATA.text_hash("synthetic player") in blocked
    assert DATA.text_hash("synthetic excluded valid") in blocked
    assert DATA.text_hash("PRIVATE_OUTCOME") not in blocked


def test_prepare_manifest_hashes_private_boundary_and_refuses_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(DATA, "PRIVATE", tmp_path)
    monkeypatch.setattr(DATA, "LocalTokenCounter", lambda path: SimpleNamespaceCounter())
    recipe_path, rules_path = tmp_path / "recipe.json", tmp_path / "rules.json"
    recipe_path.write_text(json.dumps(recipe()), encoding="utf8")
    rules_path.write_text(json.dumps(source()), encoding="utf8")
    output = tmp_path / "prepared"
    manifest = DATA.prepare(recipe_path, rules_path, tmp_path, output, None)
    assert manifest["historical_tickets_imported"] == manifest["historical_admin_targets_used"] == 0
    assert manifest["train_holdout_text_overlap"] == 0
    for name, expected in manifest["files"].items():
        assert DATA.sha((output / name).read_bytes()) == expected["sha256"]
    with pytest.raises(FileExistsError):
        DATA.prepare(recipe_path, rules_path, tmp_path, output, None)
    with pytest.raises(ValueError, match="private"):
        DATA.prepare(recipe_path, rules_path, tmp_path, tmp_path.parent / "outside", None)


class SimpleNamespaceCounter:
    template_sha256 = "synthetic-template"
    def __call__(self, messages):
        return 100
