"""Fresh64-step local adapter with strict target masking and private provenance.

Uses installed Soup0.77. No data upload, downloads or base-weight overwrites.
Inference quality, including untouched language regressions, decides acceptance.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import soup_experiment as helper

CONFIG = ROOT / "experiments/finetune-v3/train64.yaml"
DATA = ROOT / ".private/finetune-v3/prepared"
EVIDENCE = ROOT / ".private/finetune-v3"
RUN = ROOT / ".private/soup-runs/qwen3-8b-grounded-v3"


def identity():
    return {"config_sha256": helper.file_sha(CONFIG), "train_sha256": helper.file_sha(DATA/"train.jsonl"),
        "heldout_sha256": helper.file_sha(DATA/"heldout.jsonl"),
        "data_manifest_sha256": helper.file_sha(DATA/"manifest.json")}


def components():
    if importlib.metadata.version("soup-cli") != "0.77.0":
        raise RuntimeError("This experiment is verified for installed Soup0.77")
    import torch
    from transformers import AutoTokenizer
    from soup_cli.config.loader import load_config
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = load_config(CONFIG, raise_unreadable=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT/cfg.base), local_files_only=True, trust_remote_code=False)
    return cfg, tokenizer, torch


def preflight():
    cfg, tokenizer, torch = components()
    train, heldout = (helper.read_rows(DATA/(name+".jsonl")) for name in ("train", "heldout"))
    if len(train) != 64 or len(heldout) != 24:
        raise ValueError("Expected64training/24heldout records")
    stats = helper.format_and_check(train, tokenizer, cfg)
    validation = helper.format_and_check(heldout, tokenizer, cfg)
    target_lengths = [len(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False)) for row in train+heldout]
    if max(target_lengths) > 192:
        raise ValueError("Target too long for serving224-token limit")
    # A string check alone does not establish split independence; the recipe and
    # authoring limitations remain recorded separately in the data manifest.
    questions = [[row["messages"][1]["content"] for row in rows] for rows in (train, heldout)]
    if set(questions[0]) & set(questions[1]):
        raise ValueError("Training/heldout input overlap")
    result = {"status": "passed", **identity(), "human_gold": False, "fresh_adapter": True,
        "train_rows": 64, "heldout_rows": 24, "input_labels_masked": True, "targets_not_truncated": True,
        "max_train_tokens": max(x["tokens"] for x in stats),
        "max_heldout_tokens": max(x["tokens"] for x in validation), "max_target_tokens": max(target_lengths),
        "versions": {p: importlib.metadata.version(p) for p in ("soup-cli", "torch", "transformers", "peft", "trl", "bitsandbytes")}}
    path = EVIDENCE/"preflight.json"
    if path.exists():
        raise FileExistsError("Preflight evidence exists; preserve previous experiment")
    helper.write_json(path, result)
    print(json.dumps(result), flush=True)


def train():
    from peft import get_peft_model_state_dict
    from safetensors.torch import load_file
    from soup_cli.trainer.sft import SFTTrainerWrapper
    prior = json.loads((EVIDENCE/"preflight.json").read_text(encoding="utf8"))
    if prior.get("status") != "passed" or any(prior.get(k) != v for k, v in identity().items()):
        raise ValueError("Current exact-mask preflight required")
    if RUN.exists():
        raise FileExistsError("Adapter output exists; never overwrite or retry training automatically")
    cfg, tokenizer, torch = components()
    if not torch.cuda.is_available():
        raise RuntimeError("LocalCUDA required; accidental CPU8B training disabled")
    for p, version in prior["versions"].items():
        if importlib.metadata.version(p) != version:
            raise ValueError("Dependency changed after preflight")
    rows = helper.read_rows(DATA/"train.jsonl")
    wrapper = SFTTrainerWrapper(cfg, device="cuda", report_to="none", trust_remote_code=False)
    start = time.monotonic()
    raw_losses = []
    try:
        wrapper.setup({"train": rows})
        initial = {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(wrapper.model).items()}
        step = wrapper.trainer.training_step
        wrapper.trainer.args.logging_nan_inf_filter = False
        def checked_step(*args, **kwargs):
            loss = step(*args, **kwargs)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Nonfinite step loss")
            for parameter in wrapper.model.parameters():
                if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Nonfinite adapter gradient")
            raw_losses.append(float(loss.detach().float().cpu()))
            return loss
        wrapper.trainer.training_step = checked_step
        result = wrapper.train()
        # Soup releases its streaming runtime in train(); do not forward again.
        if wrapper.trainer.state.global_step != 64 or len(raw_losses) != 64:
            raise RuntimeError("Unexpected actual optimization-step count")
        saved = load_file(str(RUN/"adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(wrapper.model)
        if set(saved) != set(initial) or set(saved) != set(actual):
            raise RuntimeError("Adapter key mismatch")
        if any(not torch.equal(v, actual[k].detach().cpu().to(v.dtype)) or not bool(torch.isfinite(v).all()) for k,v in saved.items()):
            raise RuntimeError("Saved adapter mismatch or nonfinite values")
        changed = sum(not torch.equal(v, initial[k].to(v.dtype)) for k,v in saved.items())
        if not changed:
            raise RuntimeError("No adapter tensors changed")
        wrapper.trainer.save_state()
        evidence = {"status": "complete", **identity(), "fresh_adapter": True, "human_gold": False,
            "observed_steps": len(raw_losses), "raw_losses": raw_losses, "training_result": result,
            "changed_tensors": changed, "saved_tensors": len(saved), "finite_gradients_and_saved_tensors": True,
            "wall_seconds": time.monotonic()-start, "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "adapter_sha256": helper.file_sha(RUN/"adapter_model.safetensors"),
            "promotion": "not_approved_until_matched_answer_evaluation"}
        helper.write_json(RUN/"experiment.json", evidence)
        helper.write_json(EVIDENCE/"training-result.json", evidence)
        print(json.dumps({k: v for k,v in evidence.items() if k not in {"raw_losses", "training_result"}}), flush=True)
    finally:
        wrapper._close_stream_runtime()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["preflight", "train"])
    args = parser.parse_args()
    helper.environment()
    {"preflight": preflight, "train": train}[args.action]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
