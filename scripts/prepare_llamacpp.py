"""Install pinned official llama.cpp CUDA12.4 binaries/source privately; optional author GGUF.

No admin install, PATH/global edits or input uploads. Existing differing artifacts fail closed.
"""
from __future__ import annotations
import hashlib
import argparse
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / ".private/llamacpp-setup"
COMMIT = "8642541b255bfb5dc39cd351a178be09013a55d6"
TAG = "b11960"
REVISION = "7c41481f57cb95916b40956ab2f0b139b296d974"
MODEL_FILE = "Qwen3-8B-Q4_K_M.gguf"
MODEL_SHA = "d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785"
ASSETS = {
    "llama-b11960-bin-win-cuda-12.4-x64.zip": "8d9d819261de10cc2f03921b8b79c779c5c458e21edb60c170ee30a7d462d684",
    "cudart-llama-bin-win-cuda-12.4-x64.zip": "8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, destination, expected=None):
    if destination.exists():
        if expected and digest(destination) != expected:
            raise ValueError("Existing asset differs; preserving: " + destination.name)
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    if temporary.exists():
        raise FileExistsError("Partial asset exists; inspect it before retrying: " + temporary.name)
    with urllib.request.urlopen(url, timeout=120) as response, temporary.open("xb") as output:
        while chunk := response.read(4 * 1024 * 1024):
            output.write(chunk)
    if expected and digest(temporary) != expected:
        raise ValueError("Downloaded asset hash mismatch; partial preserved")
    temporary.rename(destination)
    return destination


def write_exact(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Existing extracted file differs: " + path.name)
    else:
        with path.open("xb") as handle:
            handle.write(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-torch-cuda", action="store_true", help="Reuse already installed CUDA12 libraries instead of downloading cudart")
    model_choice = parser.add_mutually_exclusive_group()
    model_choice.add_argument("--skip-model", dest="skip_model", action="store_true", help="Prepare runtime/converter only (default)")
    model_choice.add_argument("--download-model", dest="skip_model", action="store_false", help="Explicit optional download of author's GGUF instead of local conversion")
    parser.set_defaults(skip_model=True)
    args = parser.parse_args()
    SETUP.mkdir(parents=True, exist_ok=True)
    runtime = ROOT / ".private/tools" / ("llama-" + TAG)
    runtime.mkdir(parents=True, exist_ok=True)
    manifest = {"tag": TAG, "commit": COMMIT, "assets": []}
    for filename, expected in ASSETS.items():
        if args.use_torch_cuda and filename.startswith("cudart-"):
            torch_lib = ROOT / ".venv-soup/Lib/site-packages/torch/lib"
            if not all((torch_lib / name).is_file() for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")):
                raise FileNotFoundError("Installed PyTorch CUDA12 libraries not found")
            manifest["cuda_libraries"] = {"source": "existing_project_PyTorch_CUDA12", "files": [
                {"name": name, "sha256": digest(torch_lib / name)} for name in ("cudart64_12.dll", "cublas64_12.dll", "cublasLt64_12.dll")]}
            continue
        print("Preparing " + filename, flush=True)
        archive_path = fetch(f"https://github.com/ggml-org/llama.cpp/releases/download/{TAG}/{filename}", SETUP / filename, expected)
        with zipfile.ZipFile(archive_path) as archive:
            for entry in archive.infolist():
                target = (runtime / entry.filename).resolve()
                if not target.is_relative_to(runtime.resolve()) or entry.file_size > 1024**3:
                    raise ValueError("Unsafe runtime archive entry")
                if not entry.is_dir():
                    write_exact(target, archive.read(entry))
        manifest["assets"].append({"name": filename, "sha256": expected, "verified": True})
    print("Preparing pinned converter source", flush=True)
    request = urllib.request.Request(f"https://api.github.com/repos/ggml-org/llama.cpp/git/trees/{COMMIT}?recursive=1", headers={"User-Agent": "local-hackathon-setup"})
    tree = json.load(urllib.request.urlopen(request, timeout=90))
    if tree.get("truncated"):
        raise ValueError("Incomplete source tree")
    blobs = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    source_archive = fetch(f"https://codeload.github.com/ggml-org/llama.cpp/zip/{COMMIT}", SETUP / (COMMIT + ".zip"))
    source = ROOT / ".private/tools" / ("llama-source-" + TAG)
    verified = []
    with zipfile.ZipFile(source_archive) as archive:
        for entry in archive.infolist():
            relative = entry.filename.split("/", 1)[-1]
            if entry.is_dir() or not (relative.startswith(("gguf-py/", "conversion/", "requirements/"))
                                      or relative in {"convert_lora_to_gguf.py", "convert_hf_to_gguf.py"}):
                continue
            target = (source / relative).resolve()
            if not target.is_relative_to(source.resolve()) or entry.file_size > 32 * 1024**2:
                raise ValueError("Unsafe source archive entry")
            data = archive.read(entry)
            actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
            if blobs.get(relative) != actual:
                raise ValueError("Pinned source blob mismatch: " + relative)
            write_exact(target, data)
            verified.append(relative)
    manifest["source_verified_git_blobs"] = verified
    if args.skip_model:
        (SETUP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
        print("Pinned runtime/source verified; model download explicitly skipped.", flush=True)
        return
    print("Downloading author's Q4_K_M model", flush=True)
    from huggingface_hub import hf_hub_download
    model_dir = ROOT / ".private/models/qwen3-8b-gguf"
    model_path = Path(hf_hub_download("Qwen/Qwen3-8B-GGUF", MODEL_FILE, revision=REVISION, local_dir=str(model_dir)))
    actual = digest(model_path)
    if actual != MODEL_SHA:
        raise ValueError("Author GGUF failed SHA-256 verification")
    manifest["model"] = {"repo": "Qwen/Qwen3-8B-GGUF", "revision": REVISION,
                         "filename": MODEL_FILE, "sha256": actual, "verified": True}
    (SETUP / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf8")
    print("All pinned assets verified. No training or global installation performed.", flush=True)


if __name__ == "__main__":
    main()
