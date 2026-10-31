"""Dependency-free HTTP service (stdlib only) for the v3 pipeline + admin queue UI.

Endpoints
  POST /triage            case contract: request -> triage JSON (also queued for the admin UI)
  GET  /ready             health + model manifest
  GET  /                  admin UI: queue, draft, send / edit / escalate
  GET  /api/queue         queued triage results
  POST /api/queue/demo    triage N tickets from tickets_valid.jsonl into the queue (demo)
  POST /api/feedback      {id, action: send|edit|escalate, text} -> appended to feedback.jsonl
                          (admin edits are kept as a training signal)
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROLES = {"player", "reported_player", "admin"}
UI = Path(__file__).with_name("admin_ui.html")


def validate(req) -> str | None:
    if not isinstance(req, dict):
        return "body must be a JSON object"
    if req.get("kind") not in ("complaint", "report"):
        return "kind must be 'complaint' or 'report'"
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not 1 <= len(msgs) <= 50:
        return "messages must be a list of 1..50 items"
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ROLES or not isinstance(m.get("text", ""), str) or len(m.get("text", "")) > 8000:
            return "each message needs role in player|reported_player|admin and text <= 8000 chars"
        if not isinstance(m.get("attachments", 0), int) or m.get("attachments", 0) < 0:
            return "attachments must be a non-negative integer"
    if not any(m.get("role") == "player" and str(m.get("text", "")).strip() for m in msgs):
        return "a non-empty player message is required"
    if not isinstance(req.get("images_count", 0) or 0, int):
        return "images_count must be an integer"
    return None


class State:
    def __init__(self, engine, feedback_path: Path, data_dir: Path, max_queue: int = 1000):
        self.engine, self.feedback_path, self.data_dir = engine, Path(feedback_path), Path(data_dir)
        self.queue: OrderedDict[str, dict] = OrderedDict()
        self.max_queue = max_queue
        self.lock = threading.Lock()
        self.latencies: list[float] = []

    def triage(self, req: dict, source: str = "api") -> dict:
        started = time.monotonic()
        result = self.engine.triage(req)
        elapsed = time.monotonic() - started
        with self.lock:
            self.latencies = (self.latencies + [elapsed])[-2000:]
            item_id = uuid.uuid4().hex[:12]
            self.queue[item_id] = {"id": item_id, "source": source, "status": "new", "request": req, "result": result,
                                   "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            while len(self.queue) > self.max_queue:
                self.queue.popitem(last=False)
        return result

    def feedback(self, body: dict) -> dict:
        action = body.get("action")
        if action not in ("send", "edit", "escalate"):
            raise ValueError("action must be send|edit|escalate")
        with self.lock:
            item = self.queue.get(body.get("id"))
            if item is None:
                raise KeyError("unknown queue id")
            item["status"] = {"send": "sent", "edit": "edited", "escalate": "escalated"}[action]
            final = body.get("text") if isinstance(body.get("text"), str) else item["result"]["draft_reply"]["text"]
            record = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "action": action,
                      "ticket_id": item["request"].get("ticket_id"), "kind": item["request"].get("kind"),
                      "category": item["result"]["category"], "priority": item["result"]["priority"],
                      "lang": item["result"]["lang"], "draft": item["result"]["draft_reply"]["text"],
                      "final_text": final, "edited": final.strip() != item["result"]["draft_reply"]["text"].strip(),
                      "corrected_category": body.get("category") or None}
            self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
            with self.feedback_path.open("a", encoding="utf8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True, "status": item["status"]}


def make_handler(state: State):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SupportCopilotV3/1.0"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, payload, ctype="application/json; charset=utf-8"):
            body = payload if isinstance(payload, bytes) else json.dumps(payload, ensure_ascii=False).encode("utf8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n > 2_000_000:
                raise ValueError("body too large")
            return json.loads(self.rfile.read(n).decode("utf8") or "null")

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/ready":
                lat = sorted(state.latencies)
                return self._send(200, {"ready": state.engine.ready, "pipeline": "v3", "model": state.engine.bundle["manifest"].get("sha256"),
                                        "queue": len(state.queue), "p95_ms": round(lat[int(.95 * (len(lat) - 1))] * 1000, 2) if lat else None,
                                        "llm": state.engine.llm.stats if state.engine.llm else None})
            if path == "/api/queue":
                with state.lock:
                    return self._send(200, list(state.queue.values())[::-1])
            if path in ("/", "/admin"):
                return self._send(200, UI.read_bytes(), "text/html; charset=utf-8")
            self._send(404, {"detail": "not found"})

        def do_POST(self):
            url = urlparse(self.path)
            try:
                body = self._body()
            except Exception as exc:
                return self._send(400, {"detail": f"invalid JSON: {exc}"})
            try:
                if url.path == "/triage":
                    error = validate(body)
                    if error:
                        return self._send(422, {"detail": error})
                    return self._send(200, state.triage(body))
                if url.path == "/api/feedback":
                    return self._send(200, state.feedback(body or {}))
                if url.path == "/api/queue/demo":
                    n = min(int(parse_qs(url.query).get("n", ["40"])[0]), 300)
                    path = state.data_dir / "tickets_valid.jsonl"
                    rows = [json.loads(l) for l in path.open(encoding="utf8") if l.strip()][:n]
                    for r in sorted(rows, key=lambda r: r.get("created_at") or ""):
                        state.triage(r, source="demo:tickets_valid")
                    return self._send(200, {"queued": len(rows)})
            except (ValueError, KeyError) as exc:
                return self._send(400, {"detail": str(exc)})
            except Exception as exc:  # pragma: no cover
                return self._send(500, {"detail": f"internal error: {type(exc).__name__}"})
            self._send(404, {"detail": "not found"})

    return Handler


def serve(engine, host="127.0.0.1", port=8000, feedback_path=None, data_dir=None):
    state = State(engine, feedback_path or engine.model_dir / "feedback.jsonl", data_dir or engine.data_dir)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    print(f"Support Copilot v3: http://{host}:{port}/  (POST /triage, admin UI at /)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
