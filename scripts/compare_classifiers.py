"""Fixed CPU baselines on identical private authored data, never human accuracy.

No model selection or parameter search on evaluation. Experimental dependencies
live in .private/eval-deps; the serving environment is not modified. No pickle.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from train_classifier_candidate import validate_rows, view


def extra_features(text):
    """Allowlisted text-only features; no verdict, reply, nickname or identity."""
    n = max(len(text), 1)
    return [min(n, 4000)/4000, min(len(text.split()), 512)/512,
        sum(c.isdigit() for c in text)/n, sum(c.isupper() for c in text)/n,
        len(re.findall(r"[а-яА-Я]", text))/n, len(re.findall(r"[a-zA-Z]", text))/n,
        len(re.findall(r"[әғқңөұүһі]", text.casefold()))/n,
        min(text.count("?"),10)/10, min(text.count("!"),10)/10,
        min(text.count("\n"),10)/10, float("'" in text), float("\ufffd" in text)]


def validate_eval(rows, labels, *, human_gold=False):
    if not rows or any(not isinstance(r.get("text"), str) or not r["text"].strip()
        or r.get("kind") not in {"report", "complaint"} or r.get("category") not in labels
        or not r["category"].startswith(r["kind"]+".") or r.get("priority") not in {"p0","p1","p2","p3"}
        or r.get("lang") not in {"ru","kk","uz","en","es","pt","lat","und"} or r.get("human_gold") is not human_gold for r in rows):
        raise ValueError("Invalid explicitly non-gold evaluation rows")


def report(rows, predictions, labels):
    from support_copilot.classifier import metrics_for
    def one(indices):
        if not indices:
            return {"n": 0, "category": None, "priority": None}
        out = {"n": len(indices)}
        for name, classes in (("category", labels), ("priority", ["p0","p1","p2","p3"])):
            m = metrics_for([rows[i][name] for i in indices], [predictions[i][name] for i in indices], classes)
            m["macro_f1_all_taxonomy_classes"] = sum(v["f1"] for v in m["per_class"].values())/len(classes)
            out[name] = m
        return out
    return {"overall": one(list(range(len(rows)))),
        "by_kind": {k:one([i for i,r in enumerate(rows) if r["kind"]==k]) for k in ("complaint","report")},
        "by_language": {k:one([i for i,r in enumerate(rows) if r["lang"]==k]) for k in ("ru","kk","uz","en","es","pt","lat","und")}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--training", type=Path, default=ROOT/".private/finetune-v3/head-candidate/training.jsonl")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--review-packet", type=Path, help="Optional completed two-human-review packet; refuses incomplete labels")
    args = p.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/".private").resolve()) or output.exists():
        raise ValueError("Choose a new private experiment output")
    os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", TOKENIZERS_PARALLELISM="false",
                      OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", DO_NOT_TRACK="1")
    sys.path.insert(0,str(ROOT/".private/eval-deps"))
    import numpy as np
    import torch
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import FeatureUnion
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from support_copilot.classifier import TextClassifier
    from support_copilot.knowledge import HybridRetriever
    torch.set_num_threads(4)
    def read(path):
        return [json.loads(x) for x in Path(path).read_text(encoding="utf8").splitlines() if x.strip()]
    taxonomy_path = ROOT/".private/data/taxonomy.json"
    labels = [r["code"] for r in json.loads(taxonomy_path.read_text(encoding="utf8"))["categories"]]
    train = read(args.training)
    keys = validate_rows(train,labels)
    challenge = read(ROOT/".private/classifier-v1/synthetic-rows.jsonl")
    stress = [{"text": "\n".join(m["text"] for m in r["request"]["messages"] if m["role"]=="player"),
        "kind": r["request"]["kind"], "category":r["expected_category"], "priority":"p0",
        "lang":r["expected_language"], "human_gold":False}
        for r in read(ROOT/".private/quality-v5/critical16-fresh.jsonl")]
    reviewed=[]
    if args.review_packet:
        from validate_review_packet import validate_record,validate_selection
        if not args.review_packet.resolve().is_relative_to((ROOT/".private").resolve()):
            raise ValueError("Review packet must stay private")
        raw=read(args.review_packet)
        if not raw:raise ValueError("Empty review packet")
        validate_selection(raw,json.loads((args.review_packet.parent/"review-selection.json").read_text(encoding="utf8")))
        for r in raw:
            try:validate_record(r,set(labels))
            except ValueError:
                print(json.dumps({"status":"not_ready","human_metrics":None,"reason":"two_human_reviews_required"}))
                return 2
            reviewed.append({"text":"\n".join(m["text"] for m in r["messages"] if m["role"]=="player"),
                "kind":r["kind"],"category":r["category"],"priority":r["priority"],"lang":r["verified_language"],"human_gold":True,
                "server_lang":r.get("server_lang"),"region":r.get("region")})
        validate_eval(reviewed,labels,human_gold=True)
    for rows in (challenge,stress,reviewed):
        if not rows:continue
        validate_eval(rows,labels,human_gold=rows is reviewed)
        if any(" ".join(view(r).casefold().split()) in keys for r in rows):
            raise ValueError("Training/evaluation overlap")
    # Freeze single configurations before training, no evaluation-driven search.
    output.mkdir(parents=True)
    plan = {"seed":42,"human_gold":False,"training_rows":len(train),
        "parameters":{"svm_C":1.,"logistic_C":1.,"catboost_iterations":200,"catboost_depth":4},
        "selection":"none; known development metrics do not authorize promotion",
        "source_training_sha256":hashlib.sha256(args.training.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "taxonomy_sha256":hashlib.sha256(taxonomy_path.read_bytes()).hexdigest(),
        "review_packet_sha256":hashlib.sha256(args.review_packet.read_bytes()).hexdigest() if args.review_packet else None,
        "versions":{n:version(n) for n in ("scikit-learn","catboost","scipy","numpy")}}
    (output/"plan.json").write_text(json.dumps(plan,indent=2)+"\n",encoding="utf8")
    all_rows=train+challenge+stress+reviewed
    texts=[view(r) for r in all_rows]
    encoder=HybridRetriever(ROOT/".private/data/rules.json",ROOT/".private/models/multilingual-e5-small",ROOT/".private/index",dense_enabled=True)
    encoder.load()
    t=time.perf_counter()
    vectors=encoder.encode_queries(texts)
    embedding_seconds=time.perf_counter()-t
    features=np.column_stack((vectors,np.asarray([extra_features(t) for t in texts],dtype=np.float32)))
    splits={"known_authored42":(len(train),len(train)+len(challenge),challenge),
            "known_critical16":(len(train)+len(challenge),len(train)+len(challenge)+len(stress),stress)}
    if reviewed:splits["two_human_reviewed_packet"]=(len(all_rows)-len(reviewed),len(all_rows),reviewed)
    result={"schema_version":1,"human_gold":False,"plan":plan,"embedding_batch_seconds":embedding_seconds,
        "embedding_rows":len(all_rows),"models":{},"not_measured":["independent_human_accuracy","native_kk_uz_quality","end_to_end_HTTP_latency"],
        "training_fairness":"new baselines use same164; old reference also used693 weak targets; no tuning or selection on tests"}
    for name,artifact in (("reference_v1_ridge",ROOT/".private/classifier-v1/model"),
                          ("current_authored_ridge",ROOT/".private/classifier-v2-authored/model")):
        classifier=TextClassifier(artifact,encoder.encode_queries,expected_encoder_sha256=encoder._model_identity(),
            expected_taxonomy_sha256=plan["taxonomy_sha256"])
        classifier.load()
        result["models"][name]={"head_only":True,"evaluation":{}}
        for split,(a,b,rows) in splits.items():
            predictions=[]
            for row,vector in zip(rows,vectors[a:b]):
                classifier.encode=lambda ts,v=vector:np.repeat(v[None,:],len(ts),axis=0)
                predictions.append(classifier.predict("fixed player text",row["kind"]))
            result["models"][name]["evaluation"][split]=report(rows,predictions,labels)
    for name in ("tfidf_linear_svm","e5_logistic_regression","e5_features_catboost"):
        started=time.perf_counter()
        models={}
        transform=None
        if name=="tfidf_linear_svm":
            transform=FeatureUnion([("word",TfidfVectorizer(ngram_range=(1,2),sublinear_tf=True,max_features=20000)),
                ("char",TfidfVectorizer(analyzer="char_wb",ngram_range=(3,5),sublinear_tf=True,max_features=30000))])
            # Fit vocabulary/IDF on training only.
            transform.fit(texts[:len(train)])
            x=transform.transform(texts)
        else:
            x=vectors if name=="e5_logistic_regression" else features
        for task in ("category_complaint","category_report","priority"):
            indices=[i for i,r in enumerate(train) if task=="priority" or r["kind"]==task.removeprefix("category_")]
            y=[train[i]["priority" if task=="priority" else "category"] for i in indices]
            if name=="tfidf_linear_svm": model=LinearSVC(C=1.,class_weight="balanced",random_state=42,max_iter=5000)
            elif name=="e5_logistic_regression": model=LogisticRegression(C=1.,class_weight="balanced",random_state=42,max_iter=1000)
            else: model=CatBoostClassifier(iterations=200,depth=4,learning_rate=.05,l2_leaf_reg=3.,
                loss_function="MultiClass",auto_class_weights="Balanced",random_seed=42,thread_count=4,
                task_type="CPU",allow_writing_files=False,verbose=False)
            model.fit(x[indices],y)
            models[task]=model
        training_seconds=time.perf_counter()-started
        evaluations={}
        for split,(a,b,rows) in splits.items():
            predictions=[]
            timings=[]
            for i,row in enumerate(rows,a):
                t=time.perf_counter()
                category=str(np.asarray(models["category_"+row["kind"]].predict(x[i:i+1])).reshape(-1)[0])
                priority=str(np.asarray(models["priority"].predict(x[i:i+1])).reshape(-1)[0])
                timings.append(time.perf_counter()-t)
                predictions.append({"category":category,"priority":priority})
            evaluations[split]=report(rows,predictions,labels)
            evaluations[split]["head_prediction_seconds"]={"p50":float(np.quantile(timings,.5)),"p95":float(np.quantile(timings,.95)),
                "excludes":"normalization, TFIDF/E5 features, HTTP, generation"}
        result["models"][name]={"head_only":True,"training_seconds":training_seconds,"evaluation":evaluations}
        # Optional inspection artifacts are local; never load arbitrary pickle/joblib.
        if name=="e5_features_catboost":
            for task,model in models.items(): model.save_model(str(output/(task+".cbm")))
        elif name=="e5_logistic_regression":
            for task,model in models.items():
                np.savez_compressed(output/(task+".npz"),coef=model.coef_,intercept=model.intercept_,classes=model.classes_)
        print(json.dumps({"model":name,"category42":evaluations["known_authored42"]["overall"]["category"]["accuracy"],
            "priority42":evaluations["known_authored42"]["overall"]["priority"]["macro_f1_supported_classes"],"human_gold":False}),flush=True)
    (output/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
