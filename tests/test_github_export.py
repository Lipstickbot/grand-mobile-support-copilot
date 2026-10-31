"""Publication privacy boundary: synthetic files, no network or credentials."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("publisher", Path(__file__).resolve().parents[1] / "scripts/publish_github.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def write(root, path, text):
    file = root / path
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(text, encoding="utf8")


def test_only_source_is_exported_not_data_weights_runtime_or_credentials(tmp_path):
    for p in ("README.md", "src/demo.py", "scripts/run.ps1", ".codex/agents/example.toml", "team/HANDOFF.md"):
        write(tmp_path, p, "synthetic source")
    excluded = (".private/data/players.json", ".venv-soup/Lib/code.py", "team/runs/run/notes.md",
        "src/demo.egg-info/PKG-INFO", "src/__pycache__/code.py", "src/weights.safetensors",
        "scripts/runtime.log", ".env", "handout.zip", "personal.docx", ".git/config")
    for p in excluded:
        write(tmp_path, p, "PRIVATE_CONTENT")
    files = publisher.collect_source(tmp_path)
    assert set(files) == {"README.md", "src/demo.py", "scripts/run.ps1", ".codex/agents/example.toml", "team/HANDOFF.md"}
    assert all(b"PRIVATE_CONTENT" not in data for data in files.values())


def test_credential_like_literal_is_blocked_without_echoing_it(tmp_path):
    write(tmp_path, "README.md", "demo")
    token = "ghp_" + "A" * 40
    write(tmp_path, "src/app.py", "TOKEN='" + token + "'")
    with pytest.raises(ValueError) as error:
        publisher.collect_source(tmp_path)
    assert token not in str(error.value)


def test_existing_repository_is_never_changed(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(publisher, "credential", lambda owner: "fake-memory-only-token")
    calls = []
    transport = httpx.MockTransport(lambda req: (
        calls.append((req.method, req.url.path)) or
        httpx.Response(200, json={"login": "Lipstickbot"} if req.url.path == "/user" else {"id": 1})))
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real(transport=transport, **kwargs))
    with pytest.raises(FileExistsError):
        publisher.publish({"README.md": b"demo"}, {}, "Lipstickbot", "demo")
    assert calls == [("GET", "/user"), ("GET", "/repos/Lipstickbot/demo")]


def test_wrong_account_stops_before_repository_access(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setattr(publisher, "ROOT", tmp_path)
    monkeypatch.setattr(publisher, "credential", lambda owner: "fake-memory-only-token")
    calls = []
    transport = httpx.MockTransport(lambda req: (
        calls.append(req.url.path) or httpx.Response(200, json={"login": "someone_else"})))
    real = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real(transport=transport, **kwargs))
    with pytest.raises(ValueError):
        publisher.publish({"README.md": b"demo"}, {}, "Lipstickbot", "demo")
    assert calls == ["/user"]
