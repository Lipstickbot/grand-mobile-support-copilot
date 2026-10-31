"""Matched CPU-only cache ablation; identical models/requests, no LLM calls."""
from dataclasses import replace
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def timings(values):
    ordered = sorted(values)
    return {"n": len(values), "p50_ms": statistics.median(ordered)*1000,
            "p95_ms": ordered[math.ceil(.95*len(ordered))-1]*1000}


def stable(response):
    data = response.model_dump(mode="json")
    data["reasons"] = [r for r in data["reasons"] if not r.startswith("triage_seconds=")]
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=ROOT / ".private/input-v4/robustness-cases.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".private/input-v4").resolve()) or output.exists():
        parser.error("Use a new report under .private/input-v4")
    from support_copilot.config import Settings
    from support_copilot.engine import TriageEngine
    from support_copilot.schemas import TriageRequest
    from support_copilot.text import analyze_language
    cases = [json.loads(line) for line in args.cases.read_text(encoding="utf8").splitlines()]
    if any(case.get("human_gold") is not False or not case["id"].startswith("robust-v4:") for case in cases):
        raise ValueError("Synthetic scenarios required")
    requests = [TriageRequest.model_validate(case["request"]) for case in cases]
    engine = TriageEngine(replace(Settings.from_env(), generation_enabled=False))
    try:
        engine.load()
        # Warm CPU kernels once; exclude model load and this request from timings.
        engine.triage(requests[0])
        encode = engine.retriever._embed
        actual_calls = [0]
        def counted(texts, prefix):
            if prefix == "query":
                actual_calls[0] += len(texts)
            return encode(texts, prefix)
        engine.retriever._embed = counted
        results = {"disabled": [], "enabled": []}
        vectors = {"disabled": 0, "enabled": 0}
        identical = 0
        for index, request in enumerate(requests):
            values = {}
            order = ("disabled", "enabled") if index % 2 == 0 else ("enabled", "disabled")
            for mode in order:
                engine.retriever.clear_query_cache()
                engine.retriever._query_cache_size = 128 if mode == "enabled" else 0
                before = actual_calls[0]
                start = time.perf_counter()
                values[mode] = stable(engine.triage(request))
                results[mode].append(time.perf_counter()-start)
                vectors[mode] += actual_calls[0]-before
            if values["enabled"] != values["disabled"]:
                raise RuntimeError("Cache changed observable output")
            identical += 1
        engine.retriever._query_cache_size = 128
        engine.retriever.clear_query_cache()
        engine.triage(requests[0])
        before = actual_calls[0]
        engine.triage(requests[0])
        repeated_vectors = actual_calls[0]-before
        language = []
        for _ in range(10):
            for request in requests:
                text = "\n".join(m.text for m in request.messages if m.role == "player")
                start = time.perf_counter()
                analyze_language(text, request.server_lang, request.region)
                language.append(time.perf_counter()-start)
        report = {"scope": "CPU_input_classifier_retrieval_only_generation_disabled",
            "cases_sha256": hashlib.sha256(args.cases.read_bytes()).hexdigest(),
            "code_sha256": {name: hashlib.sha256((ROOT/"src/support_copilot"/name).read_bytes()).hexdigest()
                for name in ("text.py", "policy.py", "engine.py", "knowledge.py", "classifier.py")},
            "human_gold": False, "cases": len(cases), "identical_cached_uncached_responses": identical,
            "alternating_mode_order": True, "model_load_and_first_kernel_warmup_excluded": True,
            "timings": {mode: timings(values) for mode, values in results.items()},
            "encoded_query_vectors": vectors, "repeated_identical_request_new_vectors": repeated_vectors,
            "language_analysis": timings(language),
            "cache_capacity": 128, "vector_bytes_maximum": 128*384*4,
            "cache_storage": "RAM_vectors_and_exact_text_hashes_only_300secondTTL_no_response_cache",
            "not_measured": ["full_LLM_latency", "human_quality", "production_SLO"]}
        with output.open("x", encoding="utf8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        print(json.dumps(report))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
