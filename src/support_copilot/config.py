from dataclasses import dataclass
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    rules_path: Path = ROOT / ".private/data/rules.json"
    taxonomy_path: Path = ROOT / ".private/data/taxonomy.json"
    model_path: Path = ROOT / ".private/models/qwen3-8b"
    adapter_path: Path = ROOT / ".private/soup-runs/qwen3-8b-smoke"
    runtime_dir: Path = ROOT / ".private/runtime"
    embedding_path: Path = ROOT / ".private/models/multilingual-e5-small"
    index_dir: Path = ROOT / ".private/index"
    classifier_path: Path = ROOT / ".private/classifier-v1/model"
    evaluation_report_path: Path = ROOT / ".private/rag-v3/metrics.json"
    classifier_enabled: bool = True
    query_cache_enabled: bool = True
    gguf_path: Path = ROOT / ".private/models/qwen3-8b-gguf-local/Qwen3-8B-Q4_K_M.gguf"
    gguf_adapter_path: Path = ROOT / ".private/models/qwen3-8b-smoke-lora.gguf"
    llama_binary_path: Path = ROOT / ".private/tools/llama-b11960/llama-server.exe"
    llama_port: int = 8081
    inference_backend: str = "llamacpp"
    dense_enabled: bool = True
    generation_enabled: bool = True
    use_adapter: bool = False
    max_new_tokens: int = 224
    model_dtype: str = "bfloat16"
    device: str = "cuda"
    num_threads: int = 4
    generation_timeout_seconds: float = 45.0

    @classmethod
    def from_env(cls):
        defaults = cls()
        def flag(name, default):
            value = os.getenv(name, default)
            if value not in {"0", "1"}:
                raise ValueError(name + " must be 0 or 1")
            return value == "1"
        backend = os.getenv("COPILOT_BACKEND", defaults.inference_backend)
        if backend not in {"stream", "resident", "llamacpp"}:
            raise ValueError("COPILOT_BACKEND must be stream, resident or llamacpp")
        paths = {k: Path(os.getenv("COPILOT_" + k.upper(), str(getattr(defaults, k))))
                 for k in ("rules_path", "taxonomy_path", "model_path", "adapter_path", "runtime_dir", "embedding_path", "index_dir", "gguf_path", "gguf_adapter_path", "llama_binary_path", "classifier_path", "evaluation_report_path")}
        return cls(**paths,
                   inference_backend=backend,
                   llama_port=int(os.getenv("COPILOT_LLAMA_PORT", "8081")),
                   dense_enabled=flag("COPILOT_DENSE", "1"),
                   classifier_enabled=flag("COPILOT_CLASSIFIER", "1"),
                   query_cache_enabled=flag("COPILOT_QUERY_CACHE", "1"),
                   generation_enabled=flag("COPILOT_GENERATION", "1"),
                   use_adapter=flag("COPILOT_USE_ADAPTER", "0"),
                   max_new_tokens=int(os.getenv("COPILOT_MAX_NEW_TOKENS", "224")),
                   model_dtype=os.getenv("COPILOT_DTYPE", "bfloat16"),
                   device=os.getenv("COPILOT_DEVICE", "cuda"),
                   num_threads=int(os.getenv("COPILOT_NUM_THREADS", "4")),
                   generation_timeout_seconds=float(os.getenv("COPILOT_GENERATION_TIMEOUT", "45")))
