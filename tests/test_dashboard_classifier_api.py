"""Invented model-free local browser/API boundary fixtures."""
from dataclasses import replace
import json
from types import SimpleNamespace
from uuid import UUID

from fastapi.testclient import TestClient
import pytest

import support_copilot.api as api
from support_copilot.config import Settings


def response():
    return {"lang": "ru", "category": "report.help", "priority": "p3", "confidence": .5,
            "confidence_type": "uncalibrated_model_score", "uncertain": True, "needs_human": True,
            "reasons": ["synthetic_fixture"]}


def app(settings=None):
    engine = SimpleNamespace(load=lambda: None, close=lambda: None,
                             classify_request=lambda request: response())
    return api.create_app(settings or Settings(), lambda settings: engine)


def test_dashboard_and_script_use_same_origin_and_text_content():
    with TestClient(app(), base_url="http://127.0.0.1") as client:
        page = client.get("/")
        assert page.status_code == 200 and 'lang="ru"' in page.text
        assert "script-src 'self'" in page.headers["content-security-policy"]
        code = client.get("/assets/dashboard.js")
        assert code.status_code == 200 and "textContent" in code.text and "innerHTML" not in code.text


@pytest.mark.parametrize("host", ["attacker.example", "localhost.attacker.example", "127.0.0.1.attacker.example"])
def test_foreign_hosts_are_rejected_before_model_loading(host):
    with TestClient(app(), base_url="http://"+host) as client:
        result = client.get("/")
        assert result.status_code == 400 and result.json()["error"]["code"] == "untrusted_host"
        UUID(result.headers["x-request-id"])


@pytest.mark.parametrize("origin", ["https://attacker.example", "null", "http://localhost:9999"])
def test_foreign_origins_cannot_submit_classifier_work(origin):
    with TestClient(app(), base_url="http://127.0.0.1") as client:
        result = client.post("/v1/classify", headers={"Origin": origin},
            json={"kind": "report", "messages": [{"role": "player", "text": "PRIVATE"}]})
        assert result.status_code == 403 and "PRIVATE" not in result.text
        UUID(result.headers["x-request-id"])


def test_same_origin_and_cli_classification_are_available_without_generation():
    with TestClient(app(), base_url="http://127.0.0.1") as client:
        for headers in ({}, {"Origin": "http://127.0.0.1"}):
            result = client.post("/v1/classify", headers=headers,
                json={"kind": "report", "messages": [{"role": "player", "text": "hello"}]})
            assert result.status_code == 200 and result.json() == response()


def test_evaluation_requires_synthetic_provenance_and_classifier_metrics_omit_cases(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    private = tmp_path / ".private"
    private.mkdir()
    directory = private / "classifier"
    directory.mkdir()
    metrics = {"schema_version": 1, "human_gold": False, "label_kind": "weak_and_synthetic_not_human_gold",
               "weak_label_holdout_agreement": {"n": 5}, "authored_synthetic_challenge": {"n": 3},
               "cases": [{"text": "PRIVATE-TICKET"}], "path": "PRIVATE-PATH"}
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf8")
    evaluation = private / "evaluation.json"
    evaluation.write_text(json.dumps({"schema_version": 1, "human_gold": True}), encoding="utf8")
    settings = replace(Settings(), classifier_path=directory / "model", evaluation_report_path=evaluation)
    with TestClient(app(settings), base_url="http://127.0.0.1") as client:
        result = client.get("/v1/classifier/metrics")
        assert result.status_code == 200 and "PRIVATE" not in result.text and "cases" not in result.json()
        assert client.get("/v1/evaluation").status_code == 404
