"""Validate completion of a private, two-reviewer classification packet.

This checks the supplied human attestation and agreement, not reviewers' identity.
No automatic labels, historical administrator replies or model predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def selection_hash(row):
    fields={k:row[k] for k in ("review_id","kind","messages","server_lang","region","images_count","source_language_heuristic")}
    return hashlib.sha256(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode("utf8")).hexdigest()


def validate_selection(rows, manifest):
    """Freeze membership and input text before annotation, not the gold labels."""
    expected=manifest.get("request_sha256_by_review_id",{})
    ids=[r.get("review_id") for r in rows]
    if (manifest.get("schema_version")!=1 or not rows or len(set(ids))!=len(ids)
        or set(ids)!=set(expected) or any(selection_hash(r)!=expected[r["review_id"]] for r in rows)):
        raise ValueError("Review selection changed or contains duplicated records")


def validate_record(row, categories):
    if row.get("approved_for_evaluation") is not True or row.get("human_gold") is not True:
        raise ValueError("Review is not approved")
    reviews=[row.get("reviewer_1"),row.get("reviewer_2")]
    for review in reviews:
        if (not isinstance(review,dict) or set(review)!={"reviewer_id","human_attestation","category","priority","language"}
            or not isinstance(review["reviewer_id"],str) or not review["reviewer_id"].strip()
            or review["human_attestation"] is not True or review["category"] not in categories
            or not review["category"].startswith(row["kind"]+".")
            or review["priority"] not in {"p0","p1","p2","p3"}
            or review["language"] not in {"ru","kk","uz","en","es","pt","lat","und"}):
            raise ValueError("Two complete human reviews are required")
    if reviews[0]["reviewer_id"]==reviews[1]["reviewer_id"]:
        raise ValueError("Distinct reviewers required")
    expected=(row.get("category"),row.get("priority"),row.get("verified_language"))
    votes=[(r["category"],r["priority"],r["language"]) for r in reviews]
    if votes[0]==votes[1]:
        if expected!=votes[0]:raise ValueError("Final labels disagree with reviewers")
    else:
        adjudicator=row.get("adjudicator")
        if (not isinstance(adjudicator,dict) or not adjudicator.get("reviewer_id")
            or adjudicator["reviewer_id"] in {r["reviewer_id"] for r in reviews}
            or adjudicator.get("human_attestation") is not True
            or expected!=(adjudicator.get("category"),adjudicator.get("priority"),adjudicator.get("language"))
            or expected[0] not in categories or not expected[0].startswith(row["kind"]+".")
            or expected[1] not in {"p0","p1","p2","p3"} or expected[2] not in {"ru","kk","uz","en","es","pt","lat","und"}):
            raise ValueError("Disagreement requires a third human adjudicator")
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packet",type=Path,required=True)
    args=p.parse_args()
    packet=args.packet.resolve()
    if not packet.is_relative_to((ROOT/".private").resolve()):raise ValueError("Private packet required")
    categories={c["code"] for c in json.loads((ROOT/".private/data/taxonomy.json").read_text(encoding="utf8"))["categories"]}
    rows=[json.loads(x) for x in packet.read_text(encoding="utf8").splitlines() if x.strip()]
    if not rows:raise ValueError("Empty review packet")
    validate_selection(rows,json.loads((packet.parent/"review-selection.json").read_text(encoding="utf8")))
    complete=0
    for row in rows:
        try:validate_record(row,categories);complete+=1
        except ValueError:pass
    print(json.dumps({"records":len(rows),"complete_two_human_reviews":complete,
        "ready_for_evaluation":complete==len(rows),"identity_verified_by_software":False}))
    return 0 if complete==len(rows) else 2


if __name__=="__main__":raise SystemExit(main())
