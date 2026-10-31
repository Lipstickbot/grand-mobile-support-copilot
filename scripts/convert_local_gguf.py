"""Convert our existing pinned HF base and LoRA with the verified official converter.

CPU-only conversion; no training or network. Refuses to overwrite any conversion output.
"""
from __future__ import annotations
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from prepare_llamacpp import COMMIT, TAG, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume-base", action="store_true", help="Reuse a completed, validated BF16 intermediate; never overwrite it")
    args = parser.parse_args()
    source = ROOT / ".private/tools" / ("llama-source-" + TAG)
    runtime = ROOT / ".private/tools" / ("llama-" + TAG)
    base = ROOT / ".private/models/qwen3-8b"
    adapter = ROOT / ".private/soup-runs/qwen3-8b-smoke"
    destination = ROOT / ".private/models/qwen3-8b-gguf-local"
    destination.mkdir(parents=True, exist_ok=True)
    intermediate = destination / "Qwen3-8B-BF16.gguf"
    quantized = destination / "Qwen3-8B-Q4_K_M.gguf"
    lora = ROOT / ".private/models/qwen3-8b-smoke-lora.gguf"
    for output in ((quantized, lora) if args.resume_base else (intermediate, quantized, lora)):
        if output.exists():
            raise FileExistsError("Conversion output exists; preserve and inspect before rerunning: " + output.name)
    if shutil.disk_usage(destination).free < 30 * 1024**3:
        raise RuntimeError("Need at least 30 GiB free for reversible local conversion")
    manifest = json.loads((ROOT / ".private/llamacpp-setup/manifest.json").read_text(encoding="utf8"))
    if manifest["commit"] != COMMIT:
        raise ValueError("Converter pin mismatch")
    if args.resume_base:
        sys.path.insert(0, str(source / "gguf-py"))
        import gguf
        reader = gguf.GGUFReader(str(intermediate))
        if (len(reader.tensors) != 399 or intermediate.stat().st_size < 16_000_000_000
                or int(reader.get_field("general.file_type").contents()) != 32):
            raise ValueError("Existing intermediate is not the completed Qwen3 BF16 conversion")
        del reader
    env = dict(os.environ)
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=str(source / "gguf-py"), PYTHONUTF8="1")
    torch_lib = ROOT / ".venv-soup/Lib/site-packages/torch/lib"
    env["PATH"] = str(torch_lib) + os.pathsep + str(runtime) + os.pathsep + env.get("PATH", "")
    runs = []
    commands = [
        ("base", [sys.executable, "-X", "utf8", str(source / "convert_hf_to_gguf.py"),
                  "--outtype", "bf16", "--outfile", str(intermediate), str(base)]),
        # This Windows build's narrow argv cannot open absolute Cyrillic paths.
        # Relative ASCII paths from the Unicode working directory are supported.
        ("quantize", [str(runtime / "llama-quantize.exe"), str(intermediate.relative_to(ROOT)), str(quantized.relative_to(ROOT)), "Q4_K_M", "4"]),
        ("lora", [sys.executable, "-X", "utf8", str(source / "convert_lora_to_gguf.py"),
                  "--base", str(base), "--outtype", "bf16", "--outfile", str(lora), str(adapter)]),
    ]
    if args.resume_base:
        commands = commands[1:]
    for name, command in commands:
        log = ROOT / ".private/llamacpp-setup" / ("convert-" + name + "-" + str(time.time_ns()) + ".log")
        print("Running " + name + " conversion; log=" + str(log), flush=True)
        started = time.monotonic()
        with log.open("xb") as handle:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        runs.append({"stage": name, "exit_code": result.returncode, "seconds": time.monotonic()-started, "log": str(log)})
        if result.returncode:
            raise RuntimeError(name + " conversion failed; inspect private log (outputs preserved)")
    result = {"source_hf_manifest": json.loads((base / "download-manifest.json").read_text(encoding="utf8")),
              "converter_commit": COMMIT, "training_performed": False,
              "adapter_source_sha256": digest(adapter / "adapter_model.safetensors"),
              "runs": runs, "outputs": [{"path": str(path), "bytes": path.stat().st_size, "sha256": digest(path)}
                                          for path in (intermediate, quantized, lora)]}
    (ROOT / ".private/llamacpp-setup/local-conversion.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("Local Q4_K_M base and BF16 LoRA converted and hashed; original artifacts preserved.", flush=True)


if __name__ == "__main__":
    main()
