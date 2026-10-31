import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient as _TestClient
from functools import partial
TestClient = partial(_TestClient, base_url="http://127.0.0.1")

from support_copilot.api import MAX_REQUEST_BYTES, create_app
from support_copilot.cli import main
from support_copilot.config import Settings
from support_copilot.schemas import TriageRequest, TriageResponse


PRIVATE = "PRIVATE-player-complaint-123"


def record(text="hello"):
    return {"kind": "complaint", "messages": [{"role": "player", "text": text}]}


def response(text="hello"):
    return TriageResponse(
        lang="ru", category="other", priority="p2", is_spam_or_duplicate=False,
        extracted={"nickname": None, "rule_refs": [], "has_evidence": False, "server": None},
        draft_reply={"lang": "ru", "text": text, "cited": []},
        needs_human=True, confidence=0.5, reasons=["heuristic"],
    )


class FakeEngine:
    def __init__(self):
        self.loads = self.closes = 0
        self.requests = []
        self.ready = True
        self.failure = None

    def load(self):
        self.loads += 1

    def close(self):
        self.closes += 1

    def health(self):
        return {"ready": self.ready}

    def triage(self, request):
        assert isinstance(request, TriageRequest)
        self.requests.append(request)
        if self.failure:
            raise self.failure
        return response(request.messages[0].text)


def app_for(engine):
    return create_app(Settings(), lambda settings: engine)


def run_cli(engine, *args):
    return main(list(args), settings=Settings(), engine_factory=lambda settings: engine)


def test_api_loads_once_and_closes_and_validates_schema():
    engine = FakeEngine()
    with TestClient(app_for(engine)) as client:
        assert client.get("/health").json() == {"ready": True}
        payload = record("first") | {"category": "administrator", "confidence": 1.0}
        assert client.post("/triage", json=payload).json()["category"] == "other"
        assert client.post("/triage", json=record("second")).status_code == 200
        assert engine.loads == 1
        assert not hasattr(engine.requests[0], "category")
    assert engine.closes == 1


@pytest.mark.parametrize("payload", [
    {"kind": PRIVATE, "messages": []},
    {"kind": "complaint", "messages": [{"role": "admin", "text": PRIVATE}]},
    {"kind": "complaint", "messages": [{"role": "player", "text": PRIVATE * 1000}]},
])
def test_api_validation_never_echoes_input(payload):
    with TestClient(app_for(FakeEngine())) as client:
        result = client.post("/triage", json=payload)
        assert result.status_code == 422
        assert PRIVATE not in result.text
        assert result.json()["error"]["code"] == "invalid_request"


def test_api_malformed_json_and_oversized_chunked_body():
    with TestClient(app_for(FakeEngine())) as client:
        invalid = client.post("/triage", content='{"private":"' + PRIVATE)
        assert invalid.status_code == 422
        assert PRIVATE not in invalid.text
        large = client.post("/triage", content=iter([b"a" * MAX_REQUEST_BYTES, b"b"]))
        assert large.status_code == 413
        assert large.json()["error"]["code"] == "request_too_large"


def test_api_missing_lifespan_and_unready_health():
    engine = FakeEngine()
    client = TestClient(app_for(engine))
    assert client.get("/health").status_code == 503
    assert client.post("/triage", json=record()).status_code == 503
    engine.ready = False
    with TestClient(app_for(engine)) as client:
        assert client.get("/health").status_code == 503


@pytest.mark.parametrize("failure,status,code", [
    (TimeoutError(PRIVATE), 504, "timeout"),
    (RuntimeError(PRIVATE), 500, "inference_failed"),
])
def test_api_inference_errors_are_safe(failure, status, code):
    engine = FakeEngine()
    engine.failure = failure
    with TestClient(app_for(engine)) as client:
        result = client.post("/triage", json=record(PRIVATE))
        assert result.status_code == status
        assert result.json()["error"]["code"] == code
        assert PRIVATE not in result.text


def test_api_load_failure_closes_engine_and_hides_exception():
    engine = FakeEngine()

    def broken_load():
        raise RuntimeError(PRIVATE)

    engine.load = broken_load
    with pytest.raises(RuntimeError, match="Engine initialization failed") as caught:
        with TestClient(app_for(engine)):
            pass
    assert PRIVATE not in str(caught.value)
    assert engine.closes == 1


def test_api_health_and_invalid_response_errors_hide_private_values():
    engine = FakeEngine()

    def broken_health():
        raise RuntimeError(PRIVATE)

    engine.health = broken_health
    engine.triage = lambda request: {"private": PRIVATE}
    with TestClient(app_for(engine)) as client:
        result = client.get("/health")
        assert result.status_code == 503
        assert PRIVATE not in result.text
        result = client.post("/triage", json=record())
        assert result.status_code == 500
        assert PRIVATE not in result.text


def test_api_inference_does_not_block_health_and_rejects_overload():
    engine = FakeEngine()
    entered, release = threading.Event(), threading.Event()

    def slow(request):
        entered.set()
        assert release.wait(5)
        return response()

    engine.triage = slow
    with TestClient(app_for(engine)) as client, ThreadPoolExecutor() as pool:
        pending = pool.submit(client.post, "/triage", json=record())
        try:
            assert entered.wait(3)
            assert client.get("/health").status_code == 200
            assert client.post("/triage", json=record()).json()["error"]["code"] == "busy"
        finally:
            release.set()
        assert pending.result(timeout=3).status_code == 200


@pytest.mark.parametrize("compressed", [False, True])
def test_predict_streams_ordered_schema_filtered_records(tmp_path, compressed):
    source = tmp_path / ("records.jsonl.gz" if compressed else "records.jsonl")
    destination = tmp_path / "result.jsonl"
    rows = [record("Первый") | {"outcome": "unban", "category": "override"}, record("Second")]
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    if compressed:
        with gzip.open(source, "wt", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        source.write_text(payload, encoding="utf-8")
    engine = FakeEngine()
    assert run_cli(engine, "predict", "--input", str(source), "--output", str(destination)) == 0
    results = [json.loads(line) for line in destination.read_text(encoding="utf-8").splitlines()]
    assert [row["draft_reply"]["text"] for row in results] == ["Первый", "Second"]
    assert all(row["category"] == "other" for row in results)
    assert engine.loads == engine.closes == 1
    assert not list(tmp_path.glob(".copilot-*.tmp"))


@pytest.mark.parametrize("invalid", [PRIVATE, "", json.dumps(record(PRIVATE * 1000))])
def test_predict_failure_is_atomic_and_reports_safe_line_number(tmp_path, capsys, invalid):
    source, destination = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(record()) + "\n" + invalid + "\n", encoding="utf-8")
    engine = FakeEngine()
    assert run_cli(engine, "predict", "--input", str(source), "--output", str(destination)) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".copilot-*.tmp"))
    error = capsys.readouterr().err
    assert "Line 2:" in error
    assert PRIVATE not in error
    assert engine.closes == 1


def test_predict_inference_failure_is_atomic(tmp_path, capsys):
    source, destination = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(record(PRIVATE)) + "\n", encoding="utf-8")
    engine = FakeEngine()
    engine.failure = RuntimeError(PRIVATE)
    assert run_cli(engine, "predict", "--input", str(source), "--output", str(destination)) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".copilot-*.tmp"))
    error = capsys.readouterr().err
    assert "Line 1:" in error
    assert PRIVATE not in error


@pytest.mark.parametrize("corrupt_gzip", [False, True])
def test_predict_oversized_row_or_corrupt_gzip_is_safe(tmp_path, capsys, corrupt_gzip):
    source = tmp_path / ("in.jsonl.gz" if corrupt_gzip else "in.jsonl")
    destination = tmp_path / "out.jsonl"
    if corrupt_gzip:
        source.write_bytes(PRIVATE.encode())
    else:
        source.write_bytes(PRIVATE.encode() * MAX_REQUEST_BYTES)
    assert run_cli(FakeEngine(), "predict", "--input", str(source), "--output", str(destination)) == 1
    assert not destination.exists()
    assert not list(tmp_path.glob(".copilot-*.tmp"))
    error = capsys.readouterr().err
    assert "Line 1:" in error
    assert PRIVATE not in error


def test_predict_refuses_existing_output_and_input_path_before_load(tmp_path):
    source, destination = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(record()), encoding="utf-8")
    destination.write_text("preserve", encoding="utf-8")
    engine = FakeEngine()
    for output in (source, destination):
        assert run_cli(engine, "predict", "--input", str(source), "--output", str(output)) == 1
    assert engine.loads == 0
    assert destination.read_text(encoding="utf-8") == "preserve"
    assert source.read_text(encoding="utf-8") == json.dumps(record())


def test_predict_output_race_preserves_other_writers_file(tmp_path):
    source, destination = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    engine = FakeEngine()
    original = engine.triage

    def concurrent_write(request):
        destination.write_text("other writer", encoding="utf-8")
        return original(request)

    engine.triage = concurrent_write
    assert run_cli(engine, "predict", "--input", str(source), "--output", str(destination)) == 1
    assert destination.read_text(encoding="utf-8") == "other writer"
    assert not list(tmp_path.glob(".copilot-*.tmp"))


def test_predict_environment_paths(tmp_path, monkeypatch):
    source, destination = tmp_path / "in.jsonl", tmp_path / "out.jsonl"
    source.write_text(json.dumps(record()) + "\n", encoding="utf-8")
    monkeypatch.setenv("COPILOT_INPUT", str(source))
    monkeypatch.setenv("COPILOT_OUTPUT", str(destination))
    assert run_cli(FakeEngine(), "predict") == 0


def test_check_returns_readiness_and_closes(capsys):
    engine = FakeEngine()
    assert run_cli(engine, "check") == 0
    assert json.loads(capsys.readouterr().out) == {"ready": True}
    engine.ready = False
    assert run_cli(engine, "check") == 1
    assert engine.loads == engine.closes == 2


def test_check_error_does_not_echo_private_exception(capsys):
    engine = FakeEngine()

    def broken_health():
        raise RuntimeError(PRIVATE)

    engine.health = broken_health
    assert run_cli(engine, "check") == 1
    assert PRIVATE not in capsys.readouterr().err
    assert engine.closes == 1


def test_serve_loopback_default(monkeypatch):
    import uvicorn

    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: calls.append(kwargs))
    assert run_cli(FakeEngine(), "serve") == 0
    assert calls == [{"host": "127.0.0.1", "port": 8000}]


def test_module_entry_point_help_without_model_import():
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1] / "src")
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [source_root, environment.get("PYTHONPATH")]))
    completed = subprocess.run(
        [sys.executable, "-m", "support_copilot.cli", "--help"],
        capture_output=True, text=True, timeout=10, env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    assert "predict" in completed.stdout
