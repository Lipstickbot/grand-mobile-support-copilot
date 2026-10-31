"""No sockets, model, subprocess or GPU work: deterministic native adapter doubles."""
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pytest

from support_copilot.coretypes import RetrievedRule
from support_copilot.generation import GenerationError, GenerationTimeout
from support_copilot.llamacpp import LlamaCppGenerator


RULE = RetrievedRule("rules:30:2.1", 30, "Rules", "2.1", "Synthetic insults rule")


class Tokenizer:
    def __init__(self, budget="normal"):
        self.budget = budget

    def apply_chat_template(self, messages, **kwargs):
        self.options = kwargs
        self.messages = messages
        return messages[-1]["content"]

    def __call__(self, rendered, **kwargs):
        self.encode_options = kwargs
        payload = json.loads(rendered)
        size = 1025 if self.budget == "always_long" or (self.budget == "drop_rules" and payload["retrieved_passages"]) else 32
        return {"input_ids": [1] * size}


def result(**changes):
    return {"content": "Уточните время и сервер для проверки человеком.", "tokens": [40, 41, 2],
            "tokens_predicted": 3, "stop_type": "eos", "truncated": False,
            "timings": {"predicted_per_second": 20}, "tokens_cached": 0,
            "tokens_evaluated": 32, **changes}


def response(body, status=200):
    return httpx.Response(status, json=body, request=httpx.Request("POST", "http://127.0.0.1:8081/completion"))


def generator(adapter=True, body=None):
    g = LlamaCppGenerator("unused-hf", "unused-lora" if adapter else None)
    g.ready = True
    g._process = Mock(poll=Mock(return_value=None))
    g._tokenizer = Tokenizer()
    g._client = Mock()
    g._client.post.return_value = response(result() if body is None else body)
    g._adapter_id = 0 if adapter else None
    return g


def test_original_template_no_thinking_exact_prompt_and_private_metadata():
    g = generator()
    draft = g.generate("PRIVATE player message", "ru", [RULE])
    assert draft == result()["content"]
    assert g._tokenizer.options == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
    payload = g._client.post.call_args.kwargs["json"]
    assert payload["prompt"] == g._tokenizer.messages[-1]["content"]
    assert payload["cache_prompt"] is False and payload["return_tokens"] is True
    assert payload["stream"] is False and payload["temperature"] == 0
    assert payload["lora"] == [{"id": 0, "scale": 1.0}]
    assert g.last_generation["finish_reason"] == "eos"
    assert g.last_generation["generated_tokens"] == 3
    assert g.last_generation["provider_tokens_cached"] == 0
    assert "PRIVATE" not in json.dumps(g.stats())
    assert "prompt_messages" not in json.dumps(g.stats())


def test_per_request_adapter_toggle_and_nested_failure_restore():
    g = generator()
    with pytest.raises(GenerationError), g.disable_adapter():
        with g.disable_adapter():
            g.generate("text", "ru", [])
            assert g._client.post.call_args.kwargs["json"]["lora"] == [{"id": 0, "scale": 0.0}]
            assert not g.last_generation["adapter_enabled"]
        g._client.post.return_value = response(result(stop_type="limit"))
        g.generate("text", "ru", [])
    assert g._base_depth == 0
    g._client.post.return_value = response(result())
    g.generate("text", "ru", [])
    assert g._client.post.call_args.kwargs["json"]["lora"] == [{"id": 0, "scale": 1.0}]


def test_no_adapter_sends_explicit_empty_lora():
    g = generator(adapter=False)
    g.generate("text", "ru", [])
    assert g._client.post.call_args.kwargs["json"]["lora"] == []


def test_loaded_adapter_base_uses_explicit_zero_and_checks_provider_acknowledgement():
    g = generator(body=result(generation_settings={"lora": [{"id": 0, "scale": 1.0}]}))
    with g.disable_adapter(), pytest.raises(GenerationError, match="adapter_selection_mismatch"):
        g.generate("player", "ru", [])
    assert g._client.post.call_args.kwargs["json"]["lora"] == [{"id": 0, "scale": 0.0}]


@pytest.mark.parametrize("changes,finish", [
    ({"stop_type": "limit"}, "max_tokens"),
    ({"stop_type": "word"}, "missing_eos"),
    ({"stop_type": "none"}, "missing_eos"),
    ({"truncated": True}, "context_truncated"),
])
def test_complete_looking_partial_text_cannot_pass_without_eos(changes, finish):
    g = generator(body=result(**changes))
    with pytest.raises(GenerationError) as exc:
        g.generate("text", "ru", [])
    assert exc.value.code == "truncated_output"
    assert g.last_generation["finish_reason"] == finish
    assert g.last_generation["raw_text"] == result()["content"]
    assert g.last_generation["draft"] is None


@pytest.mark.parametrize("body", [{}, {"content": 42}, {"content": "x", "tokens": ["bad"]}])
def test_malformed_provider_response_is_sanitized(body):
    g = generator(body=body)
    with pytest.raises(GenerationError):
        g.generate("PRIVATE", "ru", [])
    assert g.last_generation["status"] == "error"


def test_timeout_and_provider_error_release_lock_and_keep_private_input_out_of_errors():
    g = generator()
    g._client.post.side_effect = httpx.ReadTimeout("PRIVATE echoed text")
    with pytest.raises(GenerationTimeout) as exc:
        g.generate("PRIVATE", "ru", [])
    assert "PRIVATE" not in str(exc.value)
    assert g.last_generation["finish_reason"] == "timeout"
    g._client.post.side_effect = None
    g._client.post.return_value = response({"error": "PRIVATE"}, 500)
    with pytest.raises(GenerationError) as exc:
        g.generate("PRIVATE", "ru", [])
    assert "PRIVATE" not in str(exc.value)
    g._client.post.return_value = response(result())
    assert g.generate("retry", "ru", [])


def test_context_removes_only_rules_never_player_and_records_omissions():
    g = generator()
    g._tokenizer.budget = "drop_rules"
    g.generate("Entire player message", "ru", [RULE])
    assert g.last_generation["omitted_rule_ids"] == [RULE.canonical_id]
    assert g.last_generation["included_rule_ids"] == []
    assert json.loads(g._client.post.call_args.kwargs["json"]["prompt"])["player_text"] == "Entire player message"
    g._tokenizer.budget = "always_long"
    g._client.post.reset_mock()
    with pytest.raises(GenerationError) as exc:
        g.generate("Entire player message", "ru", [])
    assert exc.value.code == "context_limit"
    g._client.post.assert_not_called()


def test_dead_process_is_not_ready_and_disable_requires_ready():
    g = generator()
    g._process.poll.return_value = 7
    assert not g.stats()["ready"]
    with pytest.raises(GenerationError) as exc:
        g.generate("text", "ru", [])
    assert exc.value.code == "not_ready"
    g.ready = False
    with pytest.raises(GenerationError), g.disable_adapter():
        pass


def test_close_only_owned_process_kills_if_termination_times_out(tmp_path):
    g = generator()
    process = g._process
    process.wait.side_effect = [subprocess.TimeoutExpired("own", 10), 0]
    key = tmp_path / "owned-key.txt"
    key.write_text("local key")
    g._key_path = key
    log = tmp_path / "owned-log.log"
    g._log_handle = log.open("wb")
    g._log_path = log
    client = g._client
    g.close()
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    client.close.assert_called_once()
    assert not key.exists() and log.exists()
    assert g._process is None and not g.ready
    g.close()  # idempotent, cannot terminate any unowned process


def test_load_owned_hidden_loopback_server_and_cleanup(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    gguf, adapter, binary = [tmp_path / name for name in ("base.gguf", "adapter.gguf", "llama-server.exe")]
    for path in (gguf, adapter, binary):
        path.write_bytes(b"fixture")
    g = LlamaCppGenerator(hf, "original-adapter", tmp_path / "runtime", gguf_path=gguf,
                         gguf_adapter_path=adapter, binary_path=binary, port=8181)
    process = Mock(poll=Mock(return_value=None))
    client = Mock()
    client.get.side_effect = [response({"status": "ok"}), response({"data": [{"id": "local"}]}),
                              response([{"id": 0, "scale": 0}])]
    tokenizer = Tokenizer()
    auto = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    with patch.object(g, "_check_port_free") as port_check, patch.dict(sys.modules, {"transformers": SimpleNamespace(AutoTokenizer=auto)}), \
            patch("support_copilot.llamacpp.subprocess.Popen", return_value=process) as popen, \
            patch("support_copilot.llamacpp.httpx.Client", return_value=client) as new_client:
        g.load()
        assert g.ready and g._adapter_id == 0
        port_check.assert_called_once()
        command = popen.call_args.args[0]
        assert command[command.index("--host") + 1] == "127.0.0.1"
        assert command[command.index("--gpu-layers") + 1] == "all"
        assert "--lora-init-without-apply" in command and "--api-key-file" in command
        assert command[command.index("--cache-ram") + 1] == "0"
        assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
        assert new_client.call_args.kwargs["trust_env"] is False
        assert auto.from_pretrained.call_args.kwargs["local_files_only"] is True
        g.load()  # no second process
        assert popen.call_count == 1
        key = g._key_path
        assert key.exists()
        g.close()
        assert not key.exists()


def test_port_conflict_never_starts_or_attaches_to_foreign_process(tmp_path):
    g = LlamaCppGenerator(tmp_path)
    with patch.object(Path, "is_file", return_value=True), \
            patch.object(g, "_check_port_free", side_effect=GenerationError("llamacpp_port_unavailable")), \
            patch("support_copilot.llamacpp.subprocess.Popen") as popen:
        with pytest.raises(GenerationError):
            g.load()
        popen.assert_not_called()
        assert g._process is None


def test_native_file_arguments_avoid_shared_unicode_workspace_prefix(tmp_path):
    root = tmp_path / "проект"
    g = LlamaCppGenerator(root / "hf", root / "adapter", root / "runtime",
                         binary_path=root / "tools/bin/llama-server.exe",
                         gguf_path=root / "models/base.gguf",
                         gguf_adapter_path=root / "models/lora.gguf")
    g._key_path = root / "runtime/key.txt"
    command = g._command()
    for flag, expected in (("-m", g.gguf_path), ("--lora", g.gguf_adapter_path), ("--api-key-file", g._key_path)):
        value = command[command.index(flag) + 1]
        assert value.isascii()
        assert (g.binary_path.parent / value).resolve() == expected.resolve()


def test_unsupported_nonascii_relative_file_name_fails_before_subprocess(tmp_path):
    g = LlamaCppGenerator(tmp_path, binary_path=tmp_path / "llama-server.exe",
                         gguf_path=tmp_path / "модель.gguf")
    g._key_path = tmp_path / "key.txt"
    with pytest.raises(GenerationError, match="ascii_relative"):
        g._command()
