"""Bonus: propose reply templates from clusters of similar historical admin replies.

Groups normalized admin replies (nicks/ids masked) by weak category and server
language, merges near-duplicates by 4-gram Jaccard, and prints the largest
clusters as template candidates for review. Output: .private/v3/template_candidates.json
"""
import gzip, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from support_copilot.v3.labels import weak_category
from support_copilot.v3.model import clean_admin

def grams(t):
    t = re.sub(r"[^\w{}]+", " ", t.casefold())
    return {t[i:i + 4] for i in range(max(1, len(t) - 3))}

def main(data=ROOT / ".private/data", out=ROOT / ".private/v3/template_candidates.json", per_group=8):
    by = defaultdict(Counter)
    with gzip.open(Path(data) / "tickets_train.jsonl.gz", "rt", encoding="utf8") as f:
        for line in f:
            t = json.loads(line)
            admin = [m["text"] for m in t["messages"] if m["role"] == "admin"]
            if admin:
                by[(weak_category(t)[0], t.get("server_lang"))][clean_admin(admin[0])[:500]] += 1
    result = {}
    for key, counter in sorted(by.items()):
        clusters = []  # [representative, grams, count, variants]
        for text, n in counter.most_common(3000):
            g = grams(text)
            for c in clusters:
                if len(g & c[1]) / max(1, len(g | c[1])) >= .6:
                    c[2] += n; c[3] += 1; break
            else:
                clusters.append([text, g, n, 1])
        clusters.sort(key=lambda c: -c[2])
        total = sum(counter.values())
        result[f"{key[0]}|{key[1]}"] = [{"template": c[0], "tickets": c[2], "share": round(c[2] / total, 3), "variants": c[3]}
                                       for c in clusters[:per_group]]
    Path(out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf8")
    for k, v in result.items():
        if v and v[0]["tickets"] >= 50:
            print(k, "->", v[0]["share"], v[0]["template"][:110])

if __name__ == "__main__":
    main()
