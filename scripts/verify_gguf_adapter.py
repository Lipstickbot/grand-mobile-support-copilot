"""Verify all 144 mapped LoRA tensors byte-for-byte against original BF16 safetensors."""
import json
import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / ".private/tools/llama-source-b11960/gguf-py"))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-path", type=Path, default=ROOT / ".private/soup-runs/qwen3-8b-smoke")
    parser.add_argument("--gguf-path", type=Path, default=ROOT / ".private/models/qwen3-8b-smoke-lora.gguf")
    parser.add_argument("--output", type=Path, default=ROOT / ".private/llamacpp-setup/adapter-verification.json")
    args = parser.parse_args()
    path = args.output.resolve()
    if not path.is_relative_to((ROOT / ".private").resolve()) or path.exists():
        parser.error("Use a new output file under .private")
    import gguf
    import torch
    from safetensors.torch import load_file
    from prepare_llamacpp import digest
    source_path = args.adapter_path.resolve() / "adapter_model.safetensors"
    target_path = args.gguf_path.resolve()
    saved = load_file(str(source_path), device="cpu")
    reader = gguf.GGUFReader(str(target_path))
    if len(saved) != 144 or len(reader.tensors) != len(saved):
        raise ValueError("Adapter tensor coverage mismatch")
    alpha = float(reader.get_field("adapter.lora.alpha").contents())
    if alpha != 16:
        raise ValueError("Adapter alpha mismatch")
    mapped = set()
    for tensor in reader.tensors:
        match = re.fullmatch(r"blk\.(\d+)\.attn_([qv])\.weight\.lora_([ab])", tensor.name)
        if not match:
            raise ValueError("Unexpected adapter tensor name: " + tensor.name)
        layer, projection, part = match.groups()
        key = f"base_model.model.model.layers.{layer}.self_attn.{projection}_proj.lora_{part.upper()}.weight"
        value = saved[key].contiguous()
        if value.dtype != torch.bfloat16 or value.view(torch.uint8).numpy().tobytes() != tensor.data.tobytes():
            raise ValueError("Converted adapter tensor differs: " + tensor.name)
        mapped.add(key)
    if mapped != set(saved):
        raise ValueError("Adapter keys missing after conversion")
    result = {"tensor_count": len(mapped), "pairs": len(mapped)//2, "alpha": alpha,
              "all_tensor_bytes_equal": True, "source_sha256": digest(source_path),
              "gguf_sha256": digest(target_path), "training_performed": False}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
