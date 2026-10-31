"""Prepare a source-only snapshot; --execute creates a NEW PRIVATE GitHub repo.

No local/global Git configuration changes. Credentials stay in memory and come
from Git Credential Manager or existing GH_TOKEN/GITHUB_TOKEN, never arguments.
Existing repositories are never changed. No request is automatically retried.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP = {"README.md", "RAG.md", "AGENTS.md", "STATUS.md", "PUBLISH.md", "pyproject.toml", ".gitignore"}
EXTRA_SOURCE = {"experiments/inference/requirements-lock.txt", "experiments/rag/requirements-lock.txt",
                "experiments/soup/requirements-lock.txt", "experiments/soup/smoke.yaml",
                "experiments/rag-training/continue108.yaml"}
DIRECTORIES = {"src", "tests", "scripts", "team", "experiments", ".codex"}
SUFFIXES = {".py", ".md", ".ps1", ".toml"}
EXCLUDED_PARTS = {".private", ".git", ".venv", ".venv-soup", "__pycache__", ".pytest_cache", "runs"}
SECRETS = re.compile(r"\b(?:github_pat_[A-Za-z0-9_]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{24,})\b")


def collect_source(root=ROOT):
    root = root.resolve()
    files = {}
    candidates = [root / name for name in TOP if (root / name).is_file()]
    for name in sorted(DIRECTORIES):
        directory = root / name
        if not directory.is_dir():
            continue
        for current, dirs, names in os.walk(directory, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_PARTS and not d.endswith(".egg-info"))
            candidates.extend(Path(current) / n for n in sorted(names))
    for path in sorted(candidates):
        rel = path.relative_to(root)
        if (set(rel.parts) & EXCLUDED_PARTS or any(p.endswith(".egg-info") for p in rel.parts)
                or not path.is_file()):
            continue
        if not (rel.as_posix() in TOP or rel.as_posix() in EXTRA_SOURCE
                or (rel.parts[0] in DIRECTORIES and path.suffix in SUFFIXES)):
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("Source symlinks are excluded")
        data = path.read_bytes()
        if len(data) > 4 * 1024 * 1024:
            raise ValueError("Oversized source file: " + rel.as_posix())
        text = data.decode("utf8")
        if SECRETS.search(text):
            raise ValueError("Credential-like text requires review in " + rel.as_posix())
        files[rel.as_posix()] = data
    if "README.md" not in files or sum(map(len, files.values())) > 32 * 1024 * 1024:
        raise ValueError("Invalid source snapshot")
    return files


def prepare(files, owner, name):
    manifest = {"owner": owner, "repository": name, "visibility": "private", "raw_data_included": False,
        "files": [{"path": p, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for p, b in files.items()]}
    directory = ROOT / ".private/github-export"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / (name + "-source.zip")
    # This derived archive can be regenerated; original source files are untouched.
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for p, data in files.items():
            output.writestr(p, data)
        output.writestr("SOURCE_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2))
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n", encoding="utf8")
    return archive, manifest


def credential(owner):
    token = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0", GCM_INTERACTIVE="Never", GCM_GUI_PROMPT="0")
    result = subprocess.run(["git", "-c", "credential.interactive=never", "credential", "fill"],
        input=f"protocol=https\nhost=github.com\nusername={owner}\n\n", capture_output=True,
        text=True, encoding="utf8", errors="replace", env=env, timeout=20)
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    if result.returncode or not fields.get("password"):
        raise RuntimeError("GitHub login required; do not paste tokens into chat")
    return fields["password"]


def publish(files, manifest, owner, name):
    import httpx
    token = credential(owner)
    state_file = ROOT / ".private/github-export/published.json"
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2026-03-10"}
    with httpx.Client(base_url="https://api.github.com", headers=headers, timeout=40,
                      trust_env=False, follow_redirects=False) as client:
        def call(method, path, payload=None, expected=(200, 201)):
            response = client.request(method, path, json=payload)
            if response.status_code not in expected:
                raise RuntimeError("GitHub returned HTTP" + str(response.status_code))
            return response.json()
        who = call("GET", "/user")
        if who.get("login", "").casefold() != owner.casefold():
            raise ValueError("Authenticated account does not match requested owner")
        existing = client.get(f"/repos/{owner}/{name}")
        if existing.status_code != 404:
            raise FileExistsError("Target repository exists or cannot be verified absent; no changes made")
        repo = call("POST", "/user/repos", {"name": name, "private": True, "auto_init": True,
            "description": "Decentrathon 6.0: local Grand Mobile support RAG research prototype and12-agent team"})
        state = {"repository": repo["full_name"], "url": repo["html_url"], "repository_id": repo["id"],
                 "phase": "created", "private": repo["private"]}
        state_file.write_text(json.dumps(state, indent=2)+"\n", encoding="utf8")
        if repo["private"] is not True:
            raise ValueError("Repository privacy was not confirmed; source not uploaded")
        prefix = f"/repos/{owner}/{name}"
        branch = repo["default_branch"]
        head = call("GET", prefix + "/git/ref/heads/" + branch)["object"]["sha"]
        base_tree = call("GET", prefix + "/git/commits/" + head)["tree"]["sha"]
        content = dict(files)
        content["SOURCE_MANIFEST.json"] = json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf8")
        tree = call("POST", prefix + "/git/trees", {"base_tree": base_tree, "tree": [
            {"path": p, "mode": "100644", "type": "blob", "content": b.decode("utf8")} for p, b in content.items()]})
        commit = call("POST", prefix + "/git/commits", {"message": "Add Support Copilot research prototype and Decentrathon agent team",
            "tree": tree["sha"], "parents": [head]})
        call("PATCH", prefix + "/git/refs/heads/" + branch, {"sha": commit["sha"], "force": False})
        if call("GET", prefix + "/git/ref/heads/" + branch)["object"]["sha"] != commit["sha"]:
            raise RuntimeError("Remote branch verification failed")
        uploaded = call("GET", prefix + "/git/trees/" + tree["sha"] + "?recursive=1")
        blobs = {e["path"]: e["sha"] for e in uploaded["tree"] if e["type"] == "blob"}
        for path, data in content.items():
            expected_sha = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
            if blobs.get(path) != expected_sha:
                raise RuntimeError("Uploaded source checksum mismatch")
        state.update(phase="verified", commit=commit["sha"], branch=branch, source_files=len(files),
                     total_files=len(content), tree_sha=tree["sha"])
        state_file.write_text(json.dumps(state, indent=2)+"\n", encoding="utf8")
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--owner", default="Lipstickbot")
    p.add_argument("--name", default="grand-mobile-support-copilot")
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    if args.owner != "Lipstickbot" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", args.name):
        raise ValueError("Unexpected target")
    files = collect_source()
    archive, manifest = prepare(files, args.owner, args.name)
    if args.execute:
        print(json.dumps(publish(files, manifest, args.owner, args.name)))
    else:
        print(json.dumps({"mode": "prepared_only", "archive": str(archive), "files": len(files),
            "bytes": sum(map(len, files.values())), "remote_changes": False}))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # Never log request bodies, auth headers or credential-helper output.
        print("Publication not completed: " + type(exc).__name__ + ". Inspect local state; no automatic retry.", file=sys.stderr)
        raise SystemExit(1)
