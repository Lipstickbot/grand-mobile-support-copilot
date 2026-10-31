"""Side-by-side review: template draft vs LLM-polished draft vs the real admin reply.

Run on a machine with an LLM server (e.g. llama.cpp `llama-server --port 8081` with Qwen):
    COPILOT_V3_LLM_URL=http://127.0.0.1:8081 python -X utf8 scripts/compare_llm.py --n 30
Writes .private/v3/llm_compare.html (contains player texts -> keep private) and prints:
how many LLM drafts passed the guard, latency p50/p95, and token usage for cost.
Decide by reading the page: keep the LLM on only if its drafts are clearly better.
"""
import argparse, html, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.v3.pipeline import TriageV3
from support_copilot.v3.llm import LLMPolisher


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / ".private/data/tickets_valid.jsonl"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--exclude-seed", type=int, default=None)
    ap.add_argument("--exclude-n", type=int, default=30)
    ap.add_argument("--out", default=str(ROOT / ".private/v3/llm_compare.html"))
    a = ap.parse_args()
    llm = LLMPolisher.from_env()
    if llm is None:
        sys.exit("Set COPILOT_V3_LLM_URL (OpenAI-compatible server, e.g. http://127.0.0.1:8081)")
    base = TriageV3(ROOT / ".private/data", ROOT / ".private/v3").load()
    with_llm = TriageV3(ROOT / ".private/data", ROOT / ".private/v3", llm=llm).load()
    rows = [json.loads(l) for l in open(a.input, encoding="utf8") if l.strip()]
    seen = set()
    if a.exclude_seed is not None:  # keep a fresh sample disjoint from an earlier reviewed one
        old = list(rows); random.Random(a.exclude_seed).shuffle(old)
        seen = {r["ticket_id"] for r in old[:a.exclude_n]}
    random.Random(a.seed).shuffle(rows)
    rows = [r for r in rows if r["ticket_id"] not in seen][:a.n]
    cards, lat, used, pipe_rej = [], [], 0, {}
    for r in rows:
        t0 = base.triage(r)
        s = time.monotonic(); t1 = with_llm.triage(r); lat.append(time.monotonic() - s)
        used += "draft=llm" in t1["reasons"]
        for x in t1["reasons"]:
            if x.startswith("llm_draft_rejected="):
                pipe_rej[x.split("=", 1)[1]] = pipe_rej.get(x.split("=", 1)[1], 0) + 1
        player = "\n".join(m["text"] for m in r["messages"] if m["role"] == "player")
        admin = "\n".join(m["text"] for m in r["messages"] if m["role"] == "admin")
        cards.append(f"""<section><h3>{html.escape(t0['category'])} · {t0['lang']} · {t0['priority']}</h3>
<p class=p><b>Игрок:</b> {html.escape(player)}</p>
<div class=g><div><b>Шаблон</b><p>{html.escape(t0['draft_reply']['text'])}</p></div>
<div><b>LLM {'✅' if 'draft=llm' in t1['reasons'] else '⛔ отклонён проверкой → шаблон'}</b><p>{html.escape(t1['draft_reply']['text'])}</p></div>
<div><b>Реальный ответ админа</b><p>{html.escape(admin) or '—'}</p></div></div></section>""")
    lat.sort()
    stats = {"n": len(rows), "llm_attempts": llm.stats["calls"], "llm_responses_received": llm.stats["responses"],
             "llm_drafts_accepted_after_guard": used, "rejected_by_llm_guard": llm.stats["rejected"],
             "rejected_by_pipeline_guard": pipe_rej, "transport_errors": llm.stats["errors"],
             "template_fallback_share": round(1 - used / max(1, len(rows)), 3),
             "scope": "whole triage() request incl. LLM call, local llama.cpp, API cost $0 (hardware cost unknown)",
             "p50_s": round(lat[len(lat) // 2], 2), "p95_s": round(lat[int(.95 * (len(lat) - 1))], 2), "llm": llm.stats}
    page = f"""<!doctype html><meta charset=utf-8><title>Шаблон vs LLM</title>
<style>body{{font:14px/1.45 system-ui;margin:24px;max-width:1300px}}section{{border-top:1px solid #ddd;padding:12px 0}}
.g{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}}.g div{{background:#f6f5f2;padding:8px 12px;border-radius:8px}}
h3{{margin:0 0 6px;font-size:14px;color:#555}}.p{{white-space:pre-wrap}}@media(max-width:800px){{.g{{grid-template-columns:1fr}}}}</style>
<h1>Шаблон vs LLM</h1><pre>{html.escape(json.dumps(stats, ensure_ascii=False, indent=1))}</pre>{''.join(cards)}"""
    Path(a.out).write_text(page, encoding="utf8")
    print(json.dumps(stats, ensure_ascii=False), "\n->", a.out)


if __name__ == "__main__":
    main()
