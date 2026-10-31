"""Real local HTTP smoke with the real corpus and E5; --llm also loads the 8B GPU runtime.

Synthetic requests only. Default mode explicitly uses templates, not fake LLM output.
Writes private evidence once; never overwrites an earlier result.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llm", action="store_true")
    parser.add_argument("--adapter", action="store_true", help="Use the existing converted LoRA")
    parser.add_argument("--all", action="store_true", help="Exercise all four synthetic records even in LLM mode")
    parser.add_argument("--test-native-exit", action="store_true", help="Verify health503 after terminating only our own native child")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.test_native_exit and not args.llm:
        parser.error("--test-native-exit requires --llm")
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".private").resolve()) or output.exists():
        parser.error("Use a new output path under .private")
    from fastapi.testclient import TestClient
    from support_copilot.api import create_app
    from support_copilot.config import Settings
    settings = replace(Settings.from_env(), generation_enabled=args.llm, use_adapter=args.adapter)
    records = [json.loads(line) for line in (ROOT / "experiments/rag/synthetic-input.jsonl").read_text(encoding="utf8").splitlines()]
    rows = []
    started = time.monotonic()
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        load_seconds = time.monotonic() - started
        health = client.get("/health")
        assert health.status_code == 200 and health.json()["ready"]
        operations = {}
        for route in ("/live", "/ready", "/v1/meta", "/v1/rag/rules/30/2.1"):
            response = client.get(route)
            assert response.status_code == 200
            UUID(response.headers["x-request-id"], version=4)
            operations[route] = response.json()
        assert operations["/v1/meta"]["corpus_version"] == health.json()["corpus_version"]
        assert operations["/v1/rag/rules/30/2.1"]["canonical_id"] == "rules:30:2.1"
        assert client.get("/v1/rag/rules/999/9.9").status_code == 404
        searches = []
        for query, language in (("Игрок оскорбил меня", "ru"),
                                ("Ойыншы мені қорлады", "kk"),
                                ("U meni haqorat qildi", "uz"),
                                ("Оплатил донат, бонус не пришёл", "ru")):
            response = client.post("/v1/rag/search", json={"query": query, "limit": 3})
            assert response.status_code == 200
            UUID(response.headers["x-request-id"], version=4)
            found = response.json()
            assert found["query_lang"] == language
            if "донат" in query:
                assert found["results"] == [] and "no_verified_applicable_rule" in found["warnings"]
            else:
                assert found["results"][0]["canonical_id"] == "rules:30:2.1"
            searches.append(found)
        # Observe concurrent real HTTP work, rather than delaying/stubbing the engine.
        concurrent = None
        if args.llm:
            generator = client.app.state.engine.generator
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(client.post, "/triage", json=records[0])
                deadline = time.monotonic() + 15
                while not client.app.state.inference_gate.locked() and not pending.done() and time.monotonic() < deadline:
                    time.sleep(.01)
                assert client.app.state.inference_gate.locked(), "Could not observe in-flight inference"
                start = time.monotonic()
                concurrent_search = client.post("/v1/rag/search", json={"query": "U meni haqorat qildi"})
                concurrent_live = client.get("/live")
                busy = client.post("/triage", json=records[0])
                concurrent_seconds = time.monotonic() - start
                assert concurrent_search.status_code == concurrent_live.status_code == 200
                assert busy.status_code == 503 and busy.json()["error"]["code"] == "busy"
                completed = pending.result(timeout=50)
                assert completed.status_code == 200
                concurrent = {"retrieval_status": concurrent_search.status_code,
                              "liveness_status": concurrent_live.status_code, "second_triage_status": busy.status_code,
                              "retrieval_and_liveness_seconds": concurrent_seconds,
                              "triage_generation": dict(generator.last_generation)}
        for record in records[:1] if args.llm and not args.all else records:
            start = time.monotonic()
            response = client.post("/triage", json=record)
            elapsed = time.monotonic() - start
            assert response.status_code == 200
            data = response.json()
            assert data["needs_human"] is True and data["draft_reply"]["text"]
            assert data["category"].startswith(record["kind"] + ".")
            generator = client.app.state.engine.generator
            rows.append({"request": record, "response": data, "seconds": elapsed,
                         "generation": dict(generator.last_generation) if generator else None})
        invalid = client.post("/triage", json={"kind": "PRIVATE", "messages": []})
        assert invalid.status_code == 422 and "PRIVATE" not in invalid.text
        exit_status = None
        if args.test_native_exit:
            if settings.inference_backend != "llamacpp":
                raise ValueError("Native exit test requires llamacpp")
            owned = client.app.state.engine.generator._process
            owned.terminate()
            owned.wait(timeout=10)
            exited = client.get("/health")
            assert exited.status_code == 503 and exited.json()["ready"] is False
            exit_status = exited.status_code
    result = {"real_llm_attempted": args.llm, "adapter_requested": args.adapter, "real_dense_retrieval": True,
              "load_seconds": load_seconds, "health": health.json(),
              "transport": "in_process_ASGI_HTTP_TestClient", "synthetic_only": True,
              "operations": operations, "searches": searches, "concurrent_work": concurrent,
              "quality_benchmark": False, "requests": rows, "malformed_input_status": invalid.status_code,
              "native_exit_health_status": exit_status}
    with output.open("x", encoding="utf8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"path": str(output), "llm": args.llm, "requests": len(rows),
                      "seconds": [round(row["seconds"], 3) for row in rows]}))


if __name__ == "__main__":
    main()
