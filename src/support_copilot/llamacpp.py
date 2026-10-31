"""Managed, loopback-only llama.cpp b11960 inference, without importing torch.

Only existing local assets are used. The original HF chat template is rendered
before calling the native /completion API; server-side chat templating is unused.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import secrets
import socket
import subprocess
import tempfile
import time

import httpx

from .generation import (GenerationError, GenerationTimeout, LocalGenerator, _GPU_LOCK,
                         _MAX_PROMPT_TOKENS, _MAX_RULE_CHARS, build_messages, normalize_draft)

ROOT = Path(__file__).resolve().parents[2]


class LlamaCppGenerator(LocalGenerator):
    """One owned server, one inference slot, explicit per-request adapter selection."""

    def __init__(self, model_path, adapter_path=None, runtime_dir=None, max_new_tokens=224,
                 timeout_seconds=45, device="cuda", dtype="bfloat16", *, gguf_path=None,
                 gguf_adapter_path=None, binary_path=None, port=8081):
        super().__init__(model_path, adapter_path, runtime_dir, max_new_tokens,
                         timeout_seconds, device, dtype)
        if type(port) is not int or not 1024 <= port <= 65535:
            raise ValueError("port must be an integer from 1024 through 65535")
        self.port = port
        self.gguf_path = Path(gguf_path or ROOT / ".private/models/qwen3-8b-gguf-local/Qwen3-8B-Q4_K_M.gguf").resolve()
        self.gguf_adapter_path = Path(gguf_adapter_path or ROOT / ".private/models/qwen3-8b-smoke-lora.gguf").resolve()
        self.binary_path = Path(binary_path or ROOT / ".private/tools/llama-b11960/llama-server.exe").resolve()
        self.use_cache = True
        self._process = None
        self._client = None
        self._log_handle = None
        self._log_path = None
        self._key_path = None
        self._adapter_id = None

    def _check_port_free(self):
        # Preflight only; the private key verifies ownership during the remaining bind race.
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("127.0.0.1", self.port))
        except OSError as exc:
            raise GenerationError("llamacpp_port_unavailable") from exc

    def _command(self):
        def native_path(path):
            # Windows b11960 narrow argv mishandles absolute Cyrillic paths.
            # Relative names avoid the shared Unicode workspace prefix.
            relative = os.path.relpath(path, self.binary_path.parent)
            if not relative.isascii():
                raise GenerationError("llamacpp_native_path_requires_ascii_relative_name")
            return relative
        command = [str(self.binary_path), "-m", native_path(self.gguf_path), "--host", "127.0.0.1",
                   "--port", str(self.port), "--gpu-layers", "all", "-c", "1536", "-np", "1",
                   "-t", "4", "-tb", "4", "-b", "512", "-ub", "128", "--jinja",
                   "--cache-ram", "0", "--flash-attn", "on", "--fit", "off", "--no-webui", "--no-agent",
                   "--api-key-file", native_path(self._key_path)]
        if self.adapter_path:
            command.extend(["--lora", native_path(self.gguf_adapter_path), "--lora-init-without-apply"])
        return command

    def _startup_ready(self):
        response = self._client.get("/health", timeout=2)
        if response.status_code != 200:
            return False
        # /health is public. An authenticated request rules out a different process
        # winning the port between our preflight and our own server's bind.
        models = self._client.get("/v1/models", timeout=2)
        models.raise_for_status()
        payload = models.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list) or not payload["data"]:
            raise GenerationError("llamacpp_invalid_model_list")
        if self.adapter_path:
            adapters = self._client.get("/lora-adapters", timeout=2)
            adapters.raise_for_status()
            entries = adapters.json()
            if (not isinstance(entries, list) or len(entries) != 1
                    or type(entries[0].get("id")) is not int):
                raise GenerationError("llamacpp_adapter_not_loaded")
            self._adapter_id = entries[0]["id"]
        return True

    def load(self):
        with _GPU_LOCK:
            if self.ready:
                return
            if self.device != "cuda":
                raise GenerationError("llamacpp_cuda_required")
            for path in (self.binary_path, self.gguf_path, self.model_path / "config.json"):
                if not path.is_file():
                    raise GenerationError("llamacpp_local_asset_missing")
            if self.adapter_path and not self.gguf_adapter_path.is_file():
                raise GenerationError("llamacpp_local_adapter_missing")
            started = time.monotonic()
            self._check_port_free()
            os.environ.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                              HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True,
                                                                trust_remote_code=False)
                self.runtime_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile("w", encoding="ascii", dir=self.runtime_dir,
                                                 prefix="llamacpp-key-", suffix=".txt", delete=False) as key_file:
                    self._key_path = Path(key_file.name)
                    secret = secrets.token_urlsafe(32)
                    key_file.write(secret + "\n")
                self._log_handle = tempfile.NamedTemporaryFile("wb", dir=self.runtime_dir,
                                                               prefix="llamacpp-startup-", suffix=".log", delete=False)
                self._log_path = Path(self._log_handle.name)
                # Ignore ambient llama.cpp tuning/tools/download/logging environment.
                env = {key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")}
                env["PATH"] = str(self.binary_path.parent) + os.pathsep + env.get("PATH", "")
                torch_lib = ROOT / ".venv-soup/Lib/site-packages/torch/lib"
                if (torch_lib / "cudart64_12.dll").is_file():
                    env["PATH"] = str(torch_lib) + os.pathsep + env["PATH"]
                self._process = subprocess.Popen(
                    self._command(), cwd=self.binary_path.parent, stdin=subprocess.DEVNULL,
                    stdout=self._log_handle, stderr=subprocess.STDOUT, env=env,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self._client = httpx.Client(base_url=f"http://127.0.0.1:{self.port}",
                                            headers={"Authorization": "Bearer " + secret},
                                            trust_env=False, timeout=2)
                while time.monotonic() - started < 120:
                    if self._process.poll() is not None:
                        raise GenerationError("llamacpp_startup_failed")
                    try:
                        if self._startup_ready():
                            if self._process.poll() is not None:
                                raise GenerationError("llamacpp_startup_failed")
                            self.ready = True
                            self._load_seconds = time.monotonic() - started
                            return
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    time.sleep(.1)
                raise GenerationError("llamacpp_startup_timeout")
            except Exception as exc:
                self._close_owned()
                if isinstance(exc, GenerationError):
                    raise
                raise GenerationError("llamacpp_startup_failure") from exc

    @contextmanager
    def disable_adapter(self):
        with _GPU_LOCK:
            if not self.ready:
                raise GenerationError("not_ready")
            self._base_depth += 1
            try:
                yield self
            finally:
                self._base_depth -= 1

    def _prepare_prompt(self, text, lang, rules, category, decision_context=None):
        selected = list(rules[:5])
        while True:
            messages = build_messages(text, lang, selected, category, decision_context=decision_context)
            prompt = self._tokenizer.apply_chat_template(messages, tokenize=False,
                                                         add_generation_prompt=True, enable_thinking=False)
            tokens = self._tokenizer(prompt, add_special_tokens=False, truncation=False)["input_ids"]
            if len(tokens) <= _MAX_PROMPT_TOKENS:
                break
            if not selected:
                raise GenerationError("context_limit")
            selected.pop()
        included = {rule.canonical_id for rule in selected}
        self.last_generation.update(prompt_tokens=len(tokens), prompt_messages=messages,
            included_rule_ids=[rule.canonical_id for rule in selected],
            omitted_rule_ids=[rule.canonical_id for rule in rules if rule.canonical_id not in included],
            excerpted_rule_ids=[rule.canonical_id for rule in selected if len(rule.text) > _MAX_RULE_CHARS])
        return prompt, included

    def generate(self, text, lang, rules, category=None, *, decision_context=None):
        started = time.monotonic()
        if not _GPU_LOCK.acquire(timeout=self.timeout_seconds):
            raise GenerationTimeout()
        try:
            adapter = bool(self.adapter_path and not self._base_depth)
            self.last_generation = {"raw_text": "", "decoded_text": "", "draft": None,
                "generated_tokens": 0, "finish_reason": "error", "seed": self.seed,
                "adapter_enabled": adapter, "use_cache": True, "cache_prompt": False,
                "enable_thinking": False, "backend": "llamacpp-b11960-cuda-q4km"}
            if not self.ready or self._process is None or self._process.poll() is not None:
                raise GenerationError("not_ready")
            prompt, allowed_refs = self._prepare_prompt(text, lang, rules, category, decision_context)
            remaining = self.timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise GenerationTimeout()
            # b11960 treats [] as global defaults; explicit scale 0 is required
            # to disable a loaded adapter for an isolated base comparison.
            lora = ([{"id": self._adapter_id, "scale": 1.0 if adapter else 0.0}]
                    if self._adapter_id is not None else [])
            self.last_generation["requested_lora"] = lora
            response = self._client.post("/completion", json={
                "prompt": prompt, "n_predict": self.max_new_tokens, "temperature": 0,
                "seed": self.seed, "cache_prompt": False, "return_tokens": True,
                "stream": False, "lora": lora,
            }, timeout=httpx.Timeout(remaining, connect=min(2, remaining)))
            response.raise_for_status()
            result = response.json()
            if not isinstance(result, dict) or not isinstance(result.get("content"), str):
                raise GenerationError("llamacpp_invalid_response")
            ids = result.get("tokens", [])
            if not isinstance(ids, list) or not all(type(token) is int for token in ids):
                raise GenerationError("llamacpp_invalid_tokens")
            raw = result["content"]
            provider_lora = result.get("generation_settings", {}).get("lora")
            self.last_generation["provider_lora"] = provider_lora
            if provider_lora is not None and provider_lora != lora:
                raise GenerationError("llamacpp_adapter_selection_mismatch")
            self.last_generation.update(raw_text=raw, decoded_text=raw,
                generated_tokens=result.get("tokens_predicted", len(ids)), generated_token_ids=ids,
                provider_timings=result.get("timings"), provider_stop_type=result.get("stop_type"),
                provider_tokens_cached=result.get("tokens_cached"),
                provider_tokens_evaluated=result.get("tokens_evaluated"))
            if time.monotonic() - started >= self.timeout_seconds:
                raise GenerationTimeout()
            if result.get("truncated") is True:
                self.last_generation["finish_reason"] = "context_truncated"
                raise GenerationError("truncated_output")
            if result.get("stop_type") != "eos" and result.get("stopped_eos") is not True:
                self.last_generation["finish_reason"] = "max_tokens" if result.get("stop_type") == "limit" else "missing_eos"
                raise GenerationError("truncated_output")
            self.last_generation["finish_reason"] = "eos"
            draft = normalize_draft(raw, allowed_refs)
            self.last_generation.update(draft=draft, status="ok")
            return draft
        except GenerationError as exc:
            self.last_generation.update(status="error", error_code=exc.code)
            if exc.code == "timeout":
                self.last_generation["finish_reason"] = "timeout"
            raise
        except httpx.TimeoutException as exc:
            self.last_generation.update(status="error", error_code="timeout", finish_reason="timeout")
            raise GenerationTimeout() from exc
        except Exception as exc:
            self.last_generation.update(status="error", error_code="provider_failure", exception_type=type(exc).__name__)
            raise GenerationError("provider_failure") from exc
        finally:
            self.last_generation["duration_seconds"] = time.monotonic() - started
            _GPU_LOCK.release()

    def stats(self):
        return {"ready": bool(self.ready and self._process is not None and self._process.poll() is None),
                "backend": "llamacpp-b11960-cuda-q4km", "use_cache": True, "cache_prompt": False,
                "enable_thinking": False, "adapter_loaded": self._adapter_id is not None,
                "load_seconds": self._load_seconds, "max_prompt_tokens": _MAX_PROMPT_TOKENS,
                "max_new_tokens": self.max_new_tokens, "timeout_seconds": self.timeout_seconds,
                "deadline": "HTTP_client_timeout;server_cancellation_unverified",
                "parallel_slots": 1, "context_size": 1536,
                "last_status": self.last_generation.get("status"),
                "last_error_code": self.last_generation.get("error_code"),
                "last_duration_seconds": self.last_generation.get("duration_seconds")}

    def _close_owned(self):
        self.ready = False
        process, self._process = self._process, None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            if self._client is not None:
                self._client.close()
                self._client = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            if self._key_path is not None:
                self._key_path.unlink(missing_ok=True)
                self._key_path = None
            self._tokenizer = None
            self._adapter_id = None

    def close(self):
        with _GPU_LOCK:
            self._close_owned()
