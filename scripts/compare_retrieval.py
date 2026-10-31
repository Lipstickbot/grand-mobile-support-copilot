"""Frozen retrieval ablation on known authored cases, without LLM generation."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
sys.path.insert(0,str(ROOT/"scripts"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    out=args.output.resolve()
    if not out.is_relative_to((ROOT/".private").resolve()) or out.exists():
        raise ValueError("Choose a new private output")
    os.environ.update(HF_HUB_OFFLINE="1",HF_HUB_DISABLE_TELEMETRY="1",TOKENIZERS_PARALLELISM="false")
    import numpy as np
    import torch
    from evaluate_http_quality import load_inputs,sha
    from support_copilot.knowledge import HybridRetriever,_query_terms,_EVENTS,_normalize
    from support_copilot.text import analyze_language,normalize_retrieval_query
    from support_copilot.policy import select_relevant_rules
    torch.set_num_threads(4)
    cases,_,authority=load_inputs(challenge=ROOT/".private/quality-live/challenge24.jsonl")
    r=HybridRetriever(ROOT/".private/data/rules.json",ROOT/".private/models/multilingual-e5-small",ROOT/".private/index",True,query_cache_size=0)
    r.load()
    result={"human_gold":False,"cases":len(cases),"required_cases":sum(bool(c["required_citation_ids"]) for c in cases),
        "rules_sha256":sha(ROOT/".private/data/rules.json"),"encoder_sha256":r._model_identity(),
        "limitations":["known authored development labels","allowlists may be incomplete","retrieval latency not HTTP latency","no independent human relevance judgments"],"modes":{}}
    for mode in ("exact_reference_only","bm25","dense","hybrid_rrf","hybrid_existing_reranking"):
        hits_raw=hits_gated=required_n=known_ids=total_ids=allowed_ids=0
        by_lang={}
        start=time.perf_counter()
        for c in cases:
            text="\n".join(m["text"] for m in c["request"]["messages"] if m["role"]=="player")
            a=analyze_language(text,c["request"].get("server_lang"),c["request"].get("region"))
            query,_=normalize_retrieval_query(a["normalized_text"])
            refs=re.findall(r"\b\d+(?:\.\d+){1,3}\b",c["request"].get("rule_refs_raw", "") or "")
            refs+=re.findall(r"(?:пункт|правил|п\.|rule)\s*(\d+(?:\.\d+){1,3})",query,re.I)
            lexical_query,cues=_query_terms(query)
            events=[sid for sid,patt in _EVENTS.items() if patt.search(_normalize(query))]
            if a["language"]=="und": candidates=[]
            elif mode=="hybrid_existing_reranking": candidates=r.search_with_diagnostics(query,refs,10)["results"]
            elif mode=="exact_reference_only": candidates=[d for d in r._docs if d.ref in refs or d.canonical_id in refs][:10]
            else:
                lexical=r._bm25(lexical_query)
                dense=r._vectors@r.encode_queries([query])[0] if mode!="bm25" else None
                scores=np.zeros(len(r._docs))
                if mode in {"bm25","hybrid_rrf"}:
                    for rank,i in enumerate(np.argsort(-lexical,kind="stable")):
                        if lexical[i]>0: scores[i]+=lexical[i] if mode=="bm25" else 1/(61+rank)
                if dense is not None:
                    for rank,i in enumerate(np.argsort(-dense,kind="stable")):
                        scores[i]+=dense[i] if mode=="dense" else 1/(61+rank)
                candidates=[replace(r._docs[i],score=float(scores[i])) for i in np.argsort(-scores,kind="stable")[:10] if scores[i]>0]
            selected=select_relevant_rules(query,candidates,cues=cues,events=events)[:5]
            raw_ids={d.canonical_id for d in candidates}
            ids={d.canonical_id for d in selected if d.source_type=="rule"}
            total_ids+=len(ids);known_ids+=len(ids&authority.keys());allowed_ids+=len(ids&set(c["allowed_citation_ids"]))
            if c["required_citation_ids"]:
                required_n+=1
                hraw=set(c["required_citation_ids"])<=raw_ids
                hgate=set(c["required_citation_ids"])<=ids
                hits_raw+=hraw;hits_gated+=hgate
                lang=by_lang.setdefault(c["expected_language"],{"cases":0,"raw_hits":0,"selected_hits":0})
                lang["cases"]+=1;lang["raw_hits"]+=hraw;lang["selected_hits"]+=hgate
        result["modes"][mode]={"required_cases":required_n,"all_required_in_top10":hits_raw,"all_required_after_gate_top5":hits_gated,
            "selected_numbered_ids":total_ids,"existing_numbered_ids":known_ids,"authored_allowlist_supported_ids":allowed_ids,
            "allowlist_precision_proxy":allowed_ids/total_ids if total_ids else None,"by_language":by_lang,
            "seconds_all100_excluding_load":time.perf_counter()-start}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result))
    return 0


if __name__=="__main__":raise SystemExit(main())
