"""Rule knowledge base helpers: section aliases, ref parsing, taxonomy priors, gold extraction."""
from __future__ import annotations
import json
import re
from pathlib import Path

from .labels import norm

SECTION_ALIAS = {"оп": 30, "op": 30, "gz": 32, "гз": 32, "жб": 31}
CODE = re.compile(r"(?<![\d.])(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)\s*(оп|op|gz|гз|жб)?(?![\d])", re.I)


def load_rule_ids(rules_path: Path) -> set[str]:
    payload = json.loads(Path(rules_path).read_text(encoding="utf-8-sig"))
    return {f"rules:{s['id']}:{p['ref']}" for s in payload["rules"] for p in s.get("points", [])}


def _section_for(code: str, suffix: str, tail: str) -> int:
    if suffix:
        return SECTION_ALIAS[suffix]
    if re.search(r"жалоб", tail) or (code.count(".") == 2 and code.startswith("1.")):
        return 31
    return 30


def parse_refs(text: str, known: set[str]) -> list[str]:
    """Canonical ids for rule numbers a text mentions ('2.1 ОП', '1.2 GZ', '1.2.7 правил жалоб')."""
    t = norm(text)
    out = []
    for m in CODE.finditer(t):
        code, suffix = m.group(1), (m.group(2) or "").lower()
        cid = f"rules:{_section_for(code, suffix, t[m.end():m.end() + 30])}:{code}"
        if cid in known and cid not in out:
            out.append(cid)
    return out


def admin_rule_ids(ticket: dict, known: set[str]) -> list[str]:
    """Rules the admin applied (training/eval gold only, never used at inference)."""
    text = "\n".join([ticket.get("violation") or "", ticket.get("punishment") or ""] +
                     [m["text"] for m in ticket.get("messages", []) if m.get("role") == "admin"])
    return parse_refs(text, known)


def taxonomy_prior(taxonomy: dict, known: set[str]) -> dict[str, list[str]]:
    out = {}
    for cat in taxonomy["categories"]:
        ids = []
        for ref in cat.get("rules", []):
            if ref == "GZ":
                ids += ["rules:32:1.1", "rules:32:1.2"]
            elif ref.startswith("ЖБ"):
                ids.append("rules:31:" + ref.split()[1])
            else:
                ids.append("rules:30:" + ref)
        out[cat["code"]] = [i for i in ids if i in known]
    return out
