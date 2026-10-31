"""Derived numeric evaluation, with explicit unmeasured semantic/human quality."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from build_quality_comparison import read_run,response
from evaluate_http_quality import CITATION,classification_metrics,latency
from summarize_http_quality import markers


def aggregate(run):
    raw=[json.loads(x) for x in (ROOT/run["source"]).read_text(encoding="utf8").splitlines() if x.strip()]
    classified=[r for r in raw if r.get("record_type")=="classifier_case"]
    taxonomy=json.loads((ROOT/".private/data/taxonomy.json").read_text(encoding="utf8"))
    labels=[r["code"] for r in taxonomy["categories"]]
    source=json.loads((ROOT/".private/data/rules.json").read_text(encoding="utf8"))
    authority={f"rules:{s['id']}:{p['ref']}" for s in source["rules"] for p in s["points"]}
    cited=known=relevant=needs_human=0
    attempted=[]
    for r in run["triage"]:
        d=response(r);draft=d.get("draft_reply") or {}
        ids=set(CITATION.findall(draft.get("text", "")))|set(draft.get("cited", []))
        cited+=len(ids);known+=len(ids&authority);relevant+=len(ids&set(r["case"]["allowed_citation_ids"]))
        needs_human+=d.get("needs_human") is True
        reasons=d.get("reasons",[])
        if r["case"]["expected_language"]!="und" and "generation=skipped_no_verified_rule" not in reasons:
            attempted.append(r["attempt"]["wall_seconds"])
    return {"human_gold":False,"source_sha256":run["sha256"],"triage":run["summary"]["triage"],
        "classifier":run["summary"]["classifier"],
        "classifier_by_kind":{kind:{f:classification_metrics([r for r in classified if r["case"]["request"]["kind"]==kind],f,ls)
            for f,ls in (("category",labels),("priority",["p0","p1","p2","p3"]))} for kind in ("complaint","report")},
        "citations":{"unique_ids_per_ticket_total":cited,"ids_existing_in_rules":known,
            "existence_precision":known/cited if cited else None,"ids_in_authored_allowlist":relevant,
            "authored_allowlist_precision_proxy":relevant/cited if cited else None,
            "pseudo_marker_cases":sum(bool(markers((response(r).get("draft_reply") or {}).get("text", ""))) for r in run["triage"]),
            "semantic_entailment_precision":None},
        "needs_human":{"count":needs_human,"cases":len(run["triage"]),"fraction":needs_human/len(run["triage"])},
        "latency_actual_generation_attempts":latency(attempted),
        "cost":{"external_api_usd_per_ticket":0,"total_usd_per_ticket":None,"reason":"local hardware electricity and amortization not measured"},
        "not_measured":{"human_answer_quality":None,"native_kk_uz_quality":None,"semantic_hallucination_fraction":None,
            "independent_real_category_accuracy":None,"independent_real_priority_f1":None}}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before",type=Path,required=True)
    p.add_argument("--after",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to((ROOT/".private").resolve()) or out.exists():raise ValueError("New private output required")
    before,after=read_run(args.before),read_run(args.after)
    if before["manifest"]["fixture_sha256"]!=after["manifest"]["fixture_sha256"]:
        raise ValueError("Identical fixtures required")
    result={"schema_version":1,"human_gold":False,"label_kind":"known_authored_development_not_independent",
        "before":aggregate(before),"after":aggregate(after),
        "limitations":["known development cases","one sequential HTTP run per version, not load SLO",
            "ID existence and authored allowlist agreement are not semantic entailment","human review unavailable"]}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps({k:{"citation_existence_precision":v["citations"]["existence_precision"],
        "needs_human":v["needs_human"]["fraction"],"actual_LLM_latency":v["latency_actual_generation_attempts"]} for k,v in result.items() if k in {"before","after"}}))
    return 0


if __name__=="__main__":raise SystemExit(main())
