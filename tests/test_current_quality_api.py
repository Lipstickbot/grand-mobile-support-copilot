"""Public comparison must never expose private raw evaluation records."""
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

import support_copilot.api as api
from support_copilot.config import Settings
from support_copilot.quality_report import PublicQualityReport


def fixture_report():
    stats = {"cases": 10, "category_correct": 3, "category_cases": 6,
        "category_macro_f1": .3, "priority_macro_f1": .4, "p0_recall": .5,
        "required_citation_hits": 4, "required_citation_cases": 5,
        "pseudo_citation_cases": 2, "intake_template_cases": 1, "fallback_cases": 3,
        "latency_p50_seconds": 1.0, "latency_p95_seconds": 2.0}
    return {"schema_version": 1, "label_kind": "authored_synthetic_quality_comparison_not_human_gold",
        "human_gold": False, "status": "complete", "comparison": {"before": stats.copy(), "after": stats.copy()},
        "limitations": ["synthetic_not_human_gold", "templates_are_not_model_training"]}


def client(tmp_path, monkeypatch, report):
    monkeypatch.setattr(api, "ROOT", tmp_path)
    p = tmp_path / ".private/quality-v5/dashboard-quality.json"
    p.parent.mkdir(parents=True)
    if report is not None:
        p.write_text(json.dumps(report), encoding="utf8")
    engine = SimpleNamespace(load=lambda: None, close=lambda: None)
    return TestClient(api.create_app(Settings(), lambda settings: engine), base_url="http://127.0.0.1")


def test_only_complete_numeric_synthetic_comparison_is_public(tmp_path, monkeypatch):
    report = fixture_report()
    with client(tmp_path, monkeypatch, report) as c:
        response = c.get("/v1/quality")
        assert response.status_code == 200
        assert response.json()["comparison"] == report["comparison"]
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("field,value", [("private_case", "PRIVATE-TEXT"),
    ("human_gold", True), ("status", "running"), ("limitations", ["PRIVATE-PATH"])])
def test_private_or_unsupported_fields_fail_closed(tmp_path, monkeypatch, field, value):
    report = fixture_report()
    report[field] = value
    with client(tmp_path, monkeypatch, report) as c:
        response = c.get("/v1/quality")
        assert response.status_code == 404 and "PRIVATE" not in response.text


@pytest.mark.parametrize("field,value", [("category_correct", 7), ("required_citation_hits", 6),
    ("pseudo_citation_cases", 11), ("p0_recall", 1.1), ("cases", True)])
def test_invalid_counts_and_ratios_are_not_published(field, value):
    report = fixture_report()
    report["comparison"]["after"][field] = value
    with pytest.raises(ValueError):
        PublicQualityReport.model_validate(report)


def test_missing_report_is_pending_not_a_fake_zero_score(tmp_path, monkeypatch):
    with client(tmp_path, monkeypatch, None) as c:
        response = c.get("/v1/quality")
        assert response.status_code == 404 and response.json()["error"]["code"] == "quality_unavailable"
