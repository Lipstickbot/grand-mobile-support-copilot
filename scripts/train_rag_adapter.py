"""Bounded108-step local LoRA continuation, exact-mask preflight and private evidence.

Uses the existing pinned Soup runtime/cache; fresh output, no cloud or base-weight writes.
"""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import soup_experiment as helpers

DATA = ROOT / ".private/rag-training-v2/prepared-reviewed"
CONFIG = ROOT / "experiments/rag-training/continue108.yaml"
EVIDENCE = ROOT / ".private/rag-training-v2"
RUN = ROOT / ".private/soup-runs/qwen3-8b-rag-v2"
INITIAL = ROOT / ".private/soup-runs/qwen3-8b-smoke"


def identity():
    return {"train_sha256": helpers.file_sha(DATA / "train108.jsonl"),
            "heldout_sha256": helpers.file_sha(DATA / "holdout.jsonl"),
            "evaluation_sha256": helpers.file_sha(DATA / "evaluation.jsonl"),
            "config_sha256": helpers.file_sha(CONFIG),
            "initial_adapter_sha256": helpers.file_sha(INITIAL / "adapter_model.safetensors")}


def components():
    import torch
    from transformers import AutoTokenizer
    from soup_cli.config.loader import load_config
    if importlib.metadata.version("soup-cli") != "0.77.0":
        raise RuntimeError("This wrapper is verified only for Soup0.77")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    cfg = load_config(CONFIG, raise_unreadable=True)
    tokenizer = AutoTokenizer.from_pretrained(str(ROOT / cfg.base), local_files_only=True, trust_remote_code=False)
    return cfg, tokenizer, torch


def preflight():
    cfg, tokenizer, torch = components()
    train = helpers.read_rows(DATA / "train108.jsonl")
    heldout = helpers.read_rows(DATA / "holdout.jsonl")
    if len(train) != 108 or len(heldout) != 36:
        raise RuntimeError("Expected fixed108/36 synthetic-silver rows")
    stats = helpers.format_and_check(train, tokenizer, cfg)
    val = helpers.format_and_check(heldout, tokenizer, cfg)
    result = {"status": "passed", "synthetic_silver_not_gold": True, **identity(),
              "train_rows": len(train), "heldout_rows": len(heldout),
              "train_max_tokens": max(row["tokens"] for row in stats),
              "heldout_max_tokens": max(row["tokens"] for row in val),
              "all_assistant_targets_complete": True, "all_system_user_prefix_labels_masked": True,
              "versions": {name: importlib.metadata.version(name) for name in
                           ("soup-cli", "torch", "transformers", "peft", "trl", "bitsandbytes")}}
    helpers.write_json(EVIDENCE / "training-preflight.json", result)
    print(json.dumps(result), flush=True)


def losses(wrapper, torch, rows):
    from soup_cli.data.sft_format import build_format_row
    formatter = build_format_row(wrapper.tokenizer, wrapper.config.data, training_cfg=wrapper.config.training)
    model = wrapper.model
    was_training = model.training
    model.eval()
    values = []
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                formatted = formatter(row)
                batch = {key: torch.tensor([formatted[key]], device="cuda") for key in ("input_ids", "attention_mask", "labels")}
                result = model(**batch, use_cache=False)
                value = float(result.loss.detach().float().cpu())
                if not math.isfinite(value):
                    raise RuntimeError("Nonfinite heldout loss")
                values.append(value)
                del result, batch
        return {"values": values, "mean": sum(values)/len(values), "scope": "silver target imitation only"}
    finally:
        model.train(was_training)


def train():
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from soup_cli.trainer.sft import SFTTrainerWrapper
    prior = json.loads((EVIDENCE / "training-preflight.json").read_text(encoding="utf8"))
    if prior.get("status") != "passed" or any(prior.get(key) != value for key, value in identity().items()):
        raise RuntimeError("Current exact-mask preflight required")
    if RUN.exists():
        raise FileExistsError("Training output exists; preserve it, no automatic retry/rotation")
    cfg, tokenizer, torch = components()
    for name, version in prior["versions"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("Dependency changed after preflight")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; no accidental CPU8B training")
    rows = helpers.read_rows(DATA / "train108.jsonl")
    heldout = helpers.read_rows(DATA / "holdout.jsonl")
    # Fixed six probes, two per language, chosen before training; evaluation.jsonl not used for loss.
    probes = [heldout[index] for index in (0, 1, 2, 18, 19, 20)]
    wrapper = SFTTrainerWrapper(cfg, device="cuda", report_to="none", trust_remote_code=False)
    started = time.monotonic()
    try:
        wrapper.setup({"train": rows})
        original = load_file(str(INITIAL / "adapter_model.safetensors"), device="cpu")
        set_peft_model_state_dict(wrapper.model, original, adapter_name="default")
        actual = get_peft_model_state_dict(wrapper.model)
        if set(actual) != set(original) or any(not torch.equal(actual[k].detach().cpu().to(v.dtype), v) for k, v in original.items()):
            raise RuntimeError("Initial adapter reload mismatch")
        before = losses(wrapper, torch, probes)
        helpers.write_json(EVIDENCE / "training-start.json", {**identity(), "before_silver_loss": before,
            "initial_adapter_reloaded_exactly": True})
        raw_steps = []
        step = wrapper.trainer.training_step
        wrapper.trainer.args.logging_nan_inf_filter = False

        def checked_step(*args, **kwargs):
            loss = step(*args, **kwargs)
            if not bool(torch.isfinite(loss).all()):
                raise RuntimeError("Nonfinite raw step loss")
            for parameter in wrapper.model.parameters():
                if parameter.requires_grad and parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("Nonfinite training gradient")
            raw_steps.append(float(loss.detach().float().cpu()))
            return loss

        wrapper.trainer.training_step = checked_step
        training_result = wrapper.train()
        if wrapper.trainer.state.global_step != 108 or len(raw_steps) != 108:
            raise RuntimeError("Actual step count differs from fixed108")
        # Soup.train() closes its streaming context. Never forward through it again.
        saved = load_file(str(RUN / "adapter_model.safetensors"), device="cpu")
        actual = get_peft_model_state_dict(wrapper.model)
        if set(saved) != set(actual) or any(not torch.equal(actual[k].detach().cpu().to(v.dtype), v) for k, v in saved.items()):
            raise RuntimeError("Saved adapter differs from trained state")
        if any(not bool(torch.isfinite(value).all()) for value in saved.values()):
            raise RuntimeError("Nonfinite saved adapter")
        changed = sum(not torch.equal(saved[key], original[key]) for key in saved)
        if not changed:
            raise RuntimeError("Adapter did not change")
        wrapper.trainer.save_state()
        helpers.write_json(RUN / "experiment.json", {"status": "complete", **identity(),
            "initial_adapter_reloaded_exactly": True, "expected_steps": 108, "observed_steps": len(raw_steps),
            "all_raw_losses_and_observed_gradients_finite": True, "raw_step_losses": raw_steps,
            "before_silver_loss": before, "training_result": training_result,
            "changed_adapter_tensors": changed, "all_saved_tensors_exact": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
            "wall_seconds": time.monotonic()-started, "human_gold": False,
            "production_promotion": "not_approved_until_full_answer_evaluation"})
        print(json.dumps({"status": "complete", "steps": len(raw_steps), "changed_tensors": changed,
                          "before_silver_loss": before["mean"], "next": "verify using fresh streaming runtime"}), flush=True)
    finally:
        wrapper._close_stream_runtime()


def verify():
    """Recover/check saved108-step artifacts and measure losses in a fresh runtime.

    No optimizer step or change to the saved run. Also supports the first run whose
    post-training forward incorrectly reused Soup's already released runtime.
    """
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from safetensors.torch import load_file
    from soup_cli.trainer.sft import SFTTrainerWrapper
    report_path = EVIDENCE / "saved-adapter-verification.json"
    audit_output = EVIDENCE / "verification-runtime"
    if report_path.exists() or audit_output.exists():
        raise FileExistsError("Verification evidence/runtime already exists; preserve it")
    cfg, tokenizer, torch = components()
    state = json.loads((RUN / "checkpoint-108/trainer_state.json").read_text(encoding="utf8"))
    history = state["log_history"]
    steps = [entry for entry in history if "loss" in entry]
    rewind = [entry for entry in helpers.read_rows(RUN / "rewind.jsonl") if entry.get("kind") == "batch"]
    if state["global_step"] != 108 or len(steps) != 108 or len(rewind) != 108:
        raise RuntimeError("Saved step evidence does not cover108 steps")
    if any(not math.isfinite(entry[key]) for entry in steps for key in ("loss", "grad_norm")):
        raise RuntimeError("Nonfinite recorded training loss/gradient norm")
    saved = load_file(str(RUN / "adapter_model.safetensors"), device="cpu")
    checkpoint = load_file(str(RUN / "checkpoint-108/adapter_model.safetensors"), device="cpu")
    original = load_file(str(INITIAL / "adapter_model.safetensors"), device="cpu")
    if set(saved) != set(original) or set(saved) != set(checkpoint) or len(saved) != 144:
        raise RuntimeError("Saved adapter keys differ")
    if any(not torch.equal(saved[k], checkpoint[k]) for k in saved):
        raise RuntimeError("Final adapter differs from saved108-step checkpoint")
    if any(not bool(torch.isfinite(value).all()) for value in saved.values()):
        raise RuntimeError("Nonfinite saved tensors")
    changed = sum(not torch.equal(saved[k], original[k]) for k in saved)
    if not changed:
        raise RuntimeError("No changed adapter tensors")
    # Only this fresh scratch wrapper may write setup bookkeeping, never the run.
    cfg = cfg.model_copy(update={"output": str(audit_output.relative_to(ROOT))}, deep=True)
    heldout = helpers.read_rows(DATA / "holdout.jsonl")
    probes = [heldout[index] for index in (0, 1, 2, 18, 19, 20)]
    wrapper = SFTTrainerWrapper(cfg, device="cuda", report_to="none", trust_remote_code=False)
    try:
        wrapper.setup({"train": helpers.read_rows(DATA / "train108.jsonl")[:1]})
        measurements = {}
        for label, weights in (("before", original), ("after", saved)):
            set_peft_model_state_dict(wrapper.model, weights, adapter_name="default")
            actual = get_peft_model_state_dict(wrapper.model)
            if set(actual) != set(weights) or any(not torch.equal(actual[k].detach().cpu().to(v.dtype), v) for k, v in weights.items()):
                raise RuntimeError("Reloaded adapter state mismatch")
            measurements[label] = losses(wrapper, torch, probes)
        report = {"status": "passed", **identity(), "observed_steps": 108,
                  "saved_checkpoint_global_step": state["global_step"],
                  "recorded_losses_and_gradient_norms_finite": True,
                  "saved_tensors_finite": True, "tensor_count": len(saved),
                  "changed_tensors": changed, "checkpoint_and_final_tensors_exact": True,
                  "old_and_new_reload_exact": True, "heldout_probes_per_language": 2,
                  "silver_loss": measurements, "adapter_sha256": helpers.file_sha(RUN / "adapter_model.safetensors"),
                  "post_training_check_recovered_with_fresh_runtime": not (RUN / "experiment.json").exists(),
                  "human_gold": False, "production_promotion": "not_approved_until_full_answer_evaluation"}
        helpers.write_json(report_path, report)
        print(json.dumps({"status": "passed", "observed_steps": 108, "changed_tensors": changed,
                          "before_silver_loss": measurements["before"]["mean"],
                          "after_silver_loss": measurements["after"]["mean"]}), flush=True)
    finally:
        wrapper._close_stream_runtime()


def main():
    helpers.environment()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "train", "verify"))
    args = parser.parse_args()
    {"preflight": preflight, "train": train, "verify": verify}[args.stage]()


if __name__ == "__main__":
    main()
