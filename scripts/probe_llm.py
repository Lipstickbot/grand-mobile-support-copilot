"""Diagnose the LLM endpoint: one tiny request, then one real draft. Prints errors verbatim.

Usage: COPILOT_V3_LLM_URL=http://127.0.0.1:8081 python scripts/probe_llm.py
"""
from __future__ import annotations
import json
import time
import urllib.request

from support_copilot.v3.llm import LLMPolisher


def main() -> None:
    llm = LLMPolisher.from_env()
    if llm is None:
        print("COPILOT_V3_LLM_URL is not set")
        return
    print("url:", llm.url, "| system proxies seen by Python:", {k: v for k, v in urllib.request.getproxies().items() if k in ("http", "https", "no")} or "none")
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(llm.url + "/health", timeout=5) as r:
            print("health:", r.status, r.read()[:200].decode("utf8", "replace"))
    except Exception as exc:
        print("health FAILED:", type(exc).__name__, exc)
    started = time.monotonic()
    try:
        data = llm._post({"model": llm.model, "max_tokens": 20, "temperature": 0,
                          "messages": [{"role": "user", "content": "Say OK\n/no_think"}]})
        print(f"tiny request OK in {time.monotonic() - started:.2f}s:",
              json.dumps(data.get("choices", [{}])[0].get("message", {}), ensure_ascii=False)[:300],
              "| usage:", data.get("usage"))
    except Exception as exc:
        body = ""
        if hasattr(exc, "read"):
            try:
                body = exc.read().decode("utf8", "replace")[:400]
            except Exception:
                pass
        print(f"tiny request FAILED in {time.monotonic() - started:.2f}s:", type(exc).__name__, exc, body)


if __name__ == "__main__":
    main()
