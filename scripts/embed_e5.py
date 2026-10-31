"""Compute multilingual-E5 embeddings of the player text for every ticket (train + valid), GPU if available.

Same encoder recipe as the legacy pipeline (knowledge.py): "query: " prefix, masked mean pooling, L2.
Only player messages are encoded (no admin text, verdict, violation, punishment, labels).
Output (private): <out>/emb.npy (float16, N x 384), <out>/ids.json, <out>/manifest.json

  python scripts/embed_e5.py --data ..\\.private\\data --model ..\\.private\\models\\multilingual-e5-small --out .private\\v8\\e5
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def player_text(t):
    return " \n ".join(m.get("text", "") for m in t.get("messages", []) if m.get("role") == "player")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--batch", type=int, default=128)
    a = ap.parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in gzip.open(data / "tickets_train.jsonl.gz", "rt", encoding="utf8") if l.strip()]
    rows += [json.loads(l) for l in open(data / "tickets_valid.jsonl", encoding="utf8") if l.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModel.from_pretrained(a.model, local_files_only=True, use_safetensors=True).to(device).eval()
    if device == "cuda":
        model = model.half()
    texts = ["query: " + player_text(r) for r in rows]
    order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
    emb = np.zeros((len(texts), model.config.hidden_size), dtype=np.float16)
    started = time.time()
    with torch.inference_mode():
        for s in range(0, len(order), a.batch):
            idx = order[s:s + a.batch]
            enc = tok([texts[i] for i in idx], padding=True, truncation=True, max_length=a.max_length, return_tensors="pt").to(device)
            h = model(**enc).last_hidden_state.float()
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            v = torch.nn.functional.normalize(v, dim=-1)
            emb[idx] = v.cpu().numpy().astype(np.float16)
            if (s // a.batch) % 100 == 0:
                print(f"{s}/{len(order)} {time.time() - started:.0f}s", flush=True)
    np.save(out / "emb.npy", emb)
    (out / "ids.json").write_text(json.dumps([r["ticket_id"] for r in rows]), encoding="utf8")
    manifest = {"model": str(a.model), "device": device, "max_length": a.max_length, "pooling": "masked-mean-l2",
                "prefix": "query: ", "n": len(rows), "seconds": round(time.time() - started, 1),
                "emb_sha256": hashlib.sha256((out / "emb.npy").read_bytes()).hexdigest(),
                "input_fields": "player messages only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
