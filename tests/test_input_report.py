import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def exporter(monkeypatch):
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("input_export_tested", scripts / "summarize_input_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(tmp_path):
    request = {"kind": "report", "messages": [{"role": "player", "text": "password: invented-secret"}]}
    case = {"id": "robust-v4:invented", "human_gold": False, "request": request,
            "expected_language": "und", "generation_expected": False,
            "allowed_citation_ids": [], "required_citation_ids": [], "forbidden_citation_ids": []}
    cases = tmp_path / "cases.jsonl"
    cases.write_text(json.dumps(case)+"\n", encoding="utf8")
    response = {"lang": "und", "needs_human": True, "extracted": {"has_evidence": False},
        "draft_reply": {"lang": "en", "text": "Please choose a language.", "cited": []}, "reasons": []}
    import evaluate_robustness as evaluation
    row = {"record_type": "case", "id": case["id"], "group": "und", "human_gold": False,
        "request": request, "response": response, "generation": {"prompt_messages": ["PRIVATE-PROMPT"]},
        "expected_language": "und", "wall_seconds": .001, "fatal_error": None,
        "assessment": evaluation.assess(case, response, {}, {})}
    records = [{"record_type": "manifest", "human_gold": False, "label_kind": evaluation.LABEL_KIND,
                "cases": 1, "cases_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(), "adapter_requested": False},
               row, {"record_type": "summary", "fake_counts": 999}]
    path = tmp_path / "evaluation.jsonl"
    path.write_text("\n".join(map(json.dumps, records)), encoding="utf8")
    return path, cases, records


def test_public_export_redacts_known_secrets_and_recomputes_counts(tmp_path, monkeypatch):
    module = exporter(monkeypatch)
    path, cases, records = fixture(tmp_path)
    report = module.build(path, cases)
    serialized = json.dumps(report)
    assert "invented-secret" not in serialized and "PRIVATE-PROMPT" not in serialized
    assert report["summary"]["overall"]["cases"] == 1
    assert len(report["examples"]) == 1


def test_export_rejects_replacement_of_frozen_player_request(tmp_path, monkeypatch):
    module = exporter(monkeypatch)
    path, cases, records = fixture(tmp_path)
    records[1]["request"] = {"messages": [{"role": "admin", "text": "PRIVATE-HISTORICAL"}]}
    path.write_text("\n".join(map(json.dumps, records)), encoding="utf8")
    with pytest.raises(ValueError):
        module.build(path, cases)
