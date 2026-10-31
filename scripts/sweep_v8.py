"""Dev-only selection of category-head hyperparameters (TEST is not read here).

Same features as v3 (model.feature_text, same TF-IDF), fit once on split-TRAIN; heads re-fit per setting.
Selection metric: macro-F1 on DEV rows whose weak label comes from admin outcome (least circular),
tie-break by overall DEV accuracy. Output: .private/v8/sweep_dev.json
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from compare_v8 import OUT, load_rows, macro_f1  # noqa: E402
from support_copilot.v3 import model as modelmod  # noqa: E402


def main():
    from sklearn.svm import LinearSVC
    rows = load_rows()
    train = [r for r in rows if r["_split"] == "train"]
    dev = [r for r in rows if r["_split"] == "dev"]
    bundle = modelmod.load(OUT / "model_split")
    vec = bundle["vectorizer"]
    ft = lambda r: modelmod.feature_text(r["kind"], modelmod.player_text(r), r.get("rule_refs_raw"))  # noqa: E731
    Xtr, Xdv = vec.transform([ft(r) for r in train]), vec.transform([ft(r) for r in dev])
    res = []
    for kind in ("complaint", "report"):
        ti = [i for i, r in enumerate(train) if r["kind"] == kind]
        di = [i for i, r in enumerate(dev) if r["kind"] == kind]
        y = [dev[i]["_weak"] for i in di]
        out_mask = [dev[i]["_weak_kind"] == "outcome" for i in di]
        labels = sorted(set(train[i]["_weak"] for i in ti))
        for c, cw in itertools.product((0.25, 0.5, 1.0), (None, "balanced")):
            clf = LinearSVC(C=c, class_weight=cw).fit(Xtr[ti], [train[i]["_weak"] for i in ti])
            p = list(clf.predict(Xdv[di]))
            yo = [a for a, m in zip(y, out_mask) if m]; po = [a for a, m in zip(p, out_mask) if m]
            res.append({"kind": kind, "C": c, "class_weight": cw or "none", "n_dev": len(di), "n_dev_outcome": len(yo),
                        "acc": round(float(np.mean([a == b for a, b in zip(y, p)])), 4),
                        "macro_f1": round(macro_f1(y, p, labels), 4),
                        "acc_outcome": round(float(np.mean([a == b for a, b in zip(yo, po)])), 4) if yo else None,
                        "macro_f1_outcome": round(macro_f1(yo, po, sorted(set(yo))), 4) if yo else None,
                        "current": (c == 0.5 and cw == (None if kind == "complaint" else "balanced"))})
            print(res[-1], flush=True)
    (OUT / "sweep_dev.json").write_text(json.dumps(res, indent=1), encoding="utf8")


if __name__ == "__main__":
    main()
