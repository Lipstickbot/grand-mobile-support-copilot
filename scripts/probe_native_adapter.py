"""Check actual native adapter influence with first-token probabilities, not claimed mode labels."""
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from dataclasses import replace
    from support_copilot.config import Settings
    from support_copilot.providers import create_generator
    g = create_generator(replace(Settings.from_env(), inference_backend="llamacpp", use_adapter=True))
    output = ROOT / ".private/native-adapter-probe-02.json"
    if output.exists():
        raise FileExistsError("Preserve existing probe output")
    try:
        g.load()
        prompt, _ = g._prepare_prompt("Игрок оскорбил меня в чате. Что написать в жалобе?", "ru", [], "complaint.insult")
        results = []
        for scale in (0.0, 1.0, 4.0, 0.0):
            started = time.monotonic()
            response = g._client.post("/completion", json={"prompt": prompt, "n_predict": 1,
                "temperature": 0, "seed": 42, "n_probs": 5, "cache_prompt": False,
                "lora": [{"id": g._adapter_id, "scale": scale}],
                "return_tokens": True}, timeout=15)
            response.raise_for_status()
            results.append({"requested_scale": scale, "seconds": time.monotonic()-started,
                            "result": response.json(), "reported_adapters": g._client.get("/lora-adapters").json()})
        probabilities = [{item["id"]: item["logprob"] for item in row["result"]["completion_probabilities"][0]["top_logprobs"]} for row in results]
        common = set(probabilities[0]) & set(probabilities[1])
        influence = max(abs(probabilities[0][key] - probabilities[1][key]) for key in common)
        repeat_drift = max(abs(probabilities[0][key] - probabilities[3][key]) for key in probabilities[0])
        with output.open("x", encoding="utf8") as handle:
            json.dump({"log": str(g._log_path), "results": results, "max_common_logprob_delta_0_vs_1": influence,
                       "repeat_base_max_logprob_drift": repeat_drift}, handle, ensure_ascii=False, indent=2)
        assert influence > 1e-5 and repeat_drift < 1e-5
        print(json.dumps([{"scale": r["requested_scale"], "probabilities": r["result"].get("completion_probabilities"),
                           "provider_lora": r["result"].get("generation_settings",{}).get("lora"),
                           "reported_adapters": [{"id": a.get("id"),"scale": a.get("scale")} for a in r["reported_adapters"]]}
                          for r in results], ensure_ascii=False))
    finally:
        g.close()


if __name__ == "__main__":
    main()
