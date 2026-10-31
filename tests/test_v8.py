"""Regression tests for the v8 evaluation tooling (synthetic data only)."""
import gzip
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _ticket(i, reporter, text, kind="complaint"):
    return {"ticket_id": f"t{i}", "kind": kind, "reporter_hash": reporter, "created_at": "2026-08-01T00:00:00",
            "messages": [{"role": "player", "text": text}]}


def test_group_split_keeps_linked_tickets_together(tmp_path):
    from group_split import main, norm
    assert norm("Ivan_Petrov77 убил меня 5 раз!!!") == norm("Anna_Smith2 убил меня 12 раз")  # nick/number/punct masked
    data = tmp_path / "data"; data.mkdir()
    train = [_ticket(i, f"r{i}", f"уникальный текст жалобы номер {chr(1072 + i % 30)}{chr(1072 + i // 30)} про игрока")
             for i in range(400)]
    train.append(_ticket(900, "r5", "совсем другой текст от того же заявителя"))           # same reporter as t5
    train.append(_ticket(901, "zz", train[7]["messages"][0]["text"].upper() + "!!!"))   # same text as t7, other reporter
    with gzip.open(data / "tickets_train.jsonl.gz", "wt", encoding="utf8") as f:
        f.writelines(json.dumps(t, ensure_ascii=False) + "\n" for t in train)
    (data / "tickets_valid.jsonl").write_text(json.dumps(_ticket(999, "r9", "жалоба из valid")) + "\n", encoding="utf8")
    out = tmp_path / "split"
    main(["--data", str(data), "--out", str(out), "--test-share", "0.2"])
    split = json.loads((out / "split.json").read_text(encoding="utf8"))
    manifest = json.loads((out / "split_manifest.json").read_text(encoding="utf8"))
    assert split["t5"]["split"] == split["t900"]["split"]
    assert split["t7"]["split"] == split["t901"]["split"]
    assert split["t9"]["split"] == "dev" and split["t999"]["split"] == "dev"  # valid-linked groups are dev
    assert manifest["test_leak_keys_shared_with_other_parts"] == {}
    assert manifest["historically_blind"] is False


@pytest.fixture(scope="module")
def engine():
    from support_copilot.v3.pipeline import TriageV3
    model = ROOT / ".private/v3"
    if not (model / "model_v3.pkl.gz").exists():
        pytest.skip("private model not available")
    return TriageV3(ROOT / ".private/data", model).load()


def test_ablation_switch_only_disables_that_override(engine):
    req = {"kind": "complaint", "images_count": 1, "messages": [{"role": "player", "text": "мне выдали бан незаслуженно, прошу снять"}]}
    assert engine.triage(req)["category"] == "complaint.admin"
    engine.disabled = {"appeal"}
    try:
        out = engine.triage(req)
    finally:
        engine.disabled = set()
    assert not any(r.startswith("override=explicit_appeal") for r in out["reasons"])
    assert out["needs_human"] is True  # disabling a category override never relaxes escalation of appeals/bans


def test_llm_rejection_reasons_are_counted():
    from support_copilot.coretypes import RetrievedRule
    from support_copilot.v3.llm import LLMPolisher
    llm = LLMPolisher("http://127.0.0.1:9")
    replies = iter(["По пункту 9.9 правил ...", "", "Qoida 2.1 бўйича"])
    llm._post = lambda payload: {"choices": [{"message": {"content": next(replies)}}]}
    rule = RetrievedRule("rules:30:2.1", 30, "t", "2.1", "Оскорбление игрока")
    for lang in ("ru", "ru", "uz"):
        assert llm.polish(text="x", lang=lang, category="complaint.insult", template="t", rules=[rule], exemplars=[]) is None
    assert llm.stats["responses"] == 3 and llm.stats["ok"] == 0
    assert llm.stats["rejected"] == {"unknown_rule_number": 1, "empty_or_too_long": 1, "cyrillic_in_uz": 1}
