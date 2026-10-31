"""Evaluate the serving classifier only after the independent human review gate.

Refuses incomplete packets and train-text overlap. Never fills missing gold labels.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))
from validate_review_packet import validate_record,validate_selection
from train_classifier_candidate import view
from compare_classifiers import report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    private=(ROOT/".private").resolve()
    if not args.packet.resolve().is_relative_to(private) or not args.output.resolve().is_relative_to(private) or args.output.exists():
        raise ValueError("Use private paths and a new output file")
    taxonomy=json.loads((private/"data/taxonomy.json").read_text(encoding="utf8"))
    labels=[r["code"] for r in taxonomy["categories"]]
    packet=[json.loads(x) for x in args.packet.read_text(encoding="utf8").splitlines() if x.strip()]
    if not packet:raise ValueError("Empty packet")
    validate_selection(packet,json.loads((args.packet.parent/"review-selection.json").read_text(encoding="utf8")))
    for r in packet:
        try:validate_record(r,set(labels))
        except ValueError:
            print(json.dumps({"status":"not_ready","reason":"independent_human_review_incomplete","human_metrics":None}))
            return 2
    rows=[{"text":"\n".join(m["text"] for m in r["messages"] if m["role"]=="player"),"kind":r["kind"],
        "category":r["category"],"priority":r["priority"],"lang":r["verified_language"],
        "server_lang":r.get("server_lang"),"region":r.get("region")} for r in packet]
    train=[json.loads(x) for x in (private/"finetune-v3/head-candidate/training.jsonl").read_text(encoding="utf8").splitlines() if x.strip()]
    keys={" ".join(view(r).casefold().split()) for r in train}
    if any(" ".join(view(r).casefold().split()) in keys for r in rows):raise ValueError("Authored training text overlap")
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    from support_copilot.config import Settings
    from support_copilot.engine import TriageEngine
    from support_copilot.schemas import TriageRequest
    engine=TriageEngine(replace(Settings.from_env(),generation_enabled=False))
    engine.load()
    try:
        predictions=[engine.classify_request(TriageRequest.model_validate({"kind":r["kind"],
            "messages":[{"role":"player","text":r["text"]}],"server_lang":r["server_lang"],"region":r["region"]})) for r in rows]
        result={"schema_version":1,"label_kind":"two_human_reviews_attested","human_gold":True,
            "n":len(rows),"metrics":report(rows,predictions,labels),
            "needs_human_fraction":sum(p["needs_human"] for p in predictions)/len(rows),
            "identity_verified_by_software":False,"representative_population_estimate":False,
            "sampling":"stratified diagnostic packet; see private audit manifest; no population weighting"}
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
        print(json.dumps({"status":"complete","n":len(rows)}))
    finally:engine.close()
    return 0


if __name__=="__main__":raise SystemExit(main())
