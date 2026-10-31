"""Extract only the two knowledge files; optionally download pinned author E5 assets.

Never uploads tickets or executes code from either archive or model repository.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
E5_FILES = {
    "config.json": "69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959",
    "model.safetensors": "1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477",
    "sentencepiece.bpe.model": "cfc8146abe2a0488e9e2a0c56de7952f7c11ab059eca145a0a727afce0db2865",
    "special_tokens_map.json": "d05497f1da52c5e09554c0cd874037a083e1dc1b9cfd48034d1c717f1afc07a7",
    "tokenizer.json": "0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39",
    "tokenizer_config.json": "a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b",
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handout", type=Path, default=Path.home() / "Downloads/handout.zip")
    parser.add_argument("--download-embedding", action="store_true")
    args = parser.parse_args()
    destination = ROOT / ".private/data"
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.handout) as archive:
        for name in ("rules.json", "taxonomy.json"):
            info = archive.getinfo("handout/" + name)
            if info.file_size > 8 * 1024 * 1024:
                raise ValueError("Unexpected knowledge file size")
            raw = archive.read(info)
            json.loads(raw)
            target = destination / name
            if target.exists():
                if target.read_bytes() != raw:
                    raise FileExistsError("Knowledge file differs; preserve it and select an explicit update procedure")
            else:
                with target.open("xb") as handle:
                    handle.write(raw)
    model = ROOT / ".private/models/multilingual-e5-small"
    if args.download_embedding:
        from huggingface_hub import snapshot_download
        snapshot_download("intfloat/multilingual-e5-small", revision=REVISION,
                          allow_patterns=list(E5_FILES), local_dir=str(model), max_workers=3)
    if not model.is_dir():
        raise FileNotFoundError("Run once with --download-embedding to install pinned E5 assets")
    files = []
    for name, expected in E5_FILES.items():
        actual = digest(model / name)
        if actual != expected:
            raise ValueError("Embedding asset failed integrity check: " + name)
        files.append({"file": name, "sha256": actual, "verified": True})
    (model / "download-manifest.json").write_text(
        json.dumps({"revision": REVISION, "files": files}, indent=2), encoding="utf-8")
    print("Rules and taxonomy ready; all six pinned E5 files verified.")


if __name__ == "__main__":
    main()
