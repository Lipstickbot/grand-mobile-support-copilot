"""python -m support_copilot.v3 {train,predict,serve,eval}"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = Path(os.getenv("COPILOT_V3_DATA", ROOT / ".private/data"))
MODEL = Path(os.getenv("COPILOT_V3_MODEL", ROOT / ".private/v3"))


def _engine():
    from .pipeline import TriageV3
    from .llm import LLMPolisher
    return TriageV3(DATA, MODEL, llm=LLMPolisher.from_env()).load()


def cmd_train(a):
    from .model import train
    train(Path(a.data or DATA), Path(a.out or MODEL))


def cmd_predict(a):
    eng = _engine()
    rows = [json.loads(l) for l in open(a.input, encoding="utf8") if l.strip()]
    order = sorted(range(len(rows)), key=lambda i: rows[i].get("created_at") or "")  # duplicates need time order
    out = [None] * len(rows)
    lat = []
    for i in order:
        t = time.monotonic()
        res = eng.triage(rows[i])
        lat.append(time.monotonic() - t)
        out[i] = {"ticket_id": rows[i].get("ticket_id"), **res}
    with open(a.output, "w", encoding="utf8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    lat.sort()
    p95 = lat[int(.95 * (len(lat) - 1))] if lat else 0
    print(json.dumps({"n": len(rows), "p50_ms": round(lat[len(lat) // 2] * 1000, 1) if lat else 0,
                      "p95_ms": round(p95 * 1000, 1), "needs_human_share": round(sum(r["needs_human"] for r in out) / max(1, len(out)), 3),
                      "llm": eng.llm.stats if eng.llm else None}, ensure_ascii=False), file=sys.stderr)


def cmd_serve(a):
    from .server import serve
    serve(_engine(), a.host, a.port)


def main(argv=None):
    p = argparse.ArgumentParser(prog="support_copilot.v3")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data"); t.add_argument("--out"); t.set_defaults(fn=cmd_train)
    pr = sub.add_parser("predict"); pr.add_argument("--input", required=True); pr.add_argument("--output", required=True); pr.set_defaults(fn=cmd_predict)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8000); s.set_defaults(fn=cmd_serve)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
