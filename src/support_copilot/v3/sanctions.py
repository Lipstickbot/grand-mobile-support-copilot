"""Extract the sanction a rule point prescribes and render it in the 6 game languages.

Only what the rule text literally says is rendered, as a *conditional* statement
("if the violation is confirmed, the rules provide ..."); it is never phrased as a
decision. Anything the parser does not fully understand is dropped, never guessed.
"""
from __future__ import annotations
import re

_TYPE = r"(MUTE REPORT|MUTE|DEMORGAN|JAIL|WARN|[Бб]локировка персонажа|[Бб]локировка аккаунта|устное предупреждение|предупреждение)"
_UNIT = r"(минут\w*|час\w*|дн\w*|день)"
_RANGE = (rf"(?:от\s+(\d+)(?:\s+{_UNIT})?\s+до\s+(\d+|навсегда)(?:\s+{_UNIT})?"
          rf"|до\s+(\d+)\s+{_UNIT}|на\s+(\d+)\s+{_UNIT}|(\d+)\s+{_UNIT}|(навсегда))?")
_PART = re.compile(rf"{_TYPE}\s*[-–—]?\s*{_RANGE}")
_LABELS = {
    "Нарушение": None, "Оскорбление игрока": "insult_player", "Упоминание родных": "mention_relatives",
    "Оскорбление родных": "insult_relatives", "Грубое нарушение": "gross", "Багоюз": "bug_abuse",
    "Распространение, использование стороннего ПО": "third_party_software", "Многочисленное нарушение": "repeated",
}
L = {  # per-language vocabulary
    "ru": {"MUTE": "MUTE", "MUTE REPORT": "MUTE REPORT", "DEMORGAN": "DEMORGAN", "JAIL": "JAIL", "WARN": "WARN",
           "char": "блокировка персонажа", "acc": "блокировка аккаунта", "warn_oral": "устное предупреждение", "warning": "предупреждение",
           "min": "мин", "h": "ч", "d": "дн.", "forever": "навсегда", "upto": "до {n} {u}", "range": "от {a} до {b} {u}",
           "range_forever": "от {a} {u} до навсегда", "exact": "{n} {u}", "or": " или ", "sep": "; ",
           "insult_player": "оскорбление игрока", "mention_relatives": "упоминание родных", "insult_relatives": "оскорбление родных",
           "gross": "грубое нарушение", "bug_abuse": "багоюз", "third_party_software": "стороннее ПО", "repeated": "многократное нарушение"},
    "kk": {"char": "кейіпкерді бұғаттау", "acc": "аккаунтты бұғаттау", "warn_oral": "ауызша ескерту", "warning": "ескерту",
           "min": "мин", "h": "сағ", "d": "күн", "forever": "мәңгі", "upto": "{n} {u} дейін", "range": "{a}–{b} {u}",
           "range_forever": "{a} {u} бастап мәңгіге дейін", "exact": "{n} {u}", "or": " немесе ", "sep": "; ",
           "insult_player": "ойыншыны қорлау", "mention_relatives": "туыстарын атау", "insult_relatives": "туыстарын қорлау",
           "gross": "өрескел бұзушылық", "bug_abuse": "баг пайдалану", "third_party_software": "бөгде бағдарлама", "repeated": "қайталанған бұзушылық"},
    "uz": {"char": "personajni bloklash", "acc": "akkauntni bloklash", "warn_oral": "og‘zaki ogohlantirish", "warning": "ogohlantirish",
           "min": "daqiqa", "h": "soat", "d": "kun", "forever": "butunlay", "upto": "{n} {u}gacha", "range": "{a}–{b} {u}",
           "range_forever": "{a} {u}dan butunlaygacha", "exact": "{n} {u}", "or": " yoki ", "sep": "; ",
           "insult_player": "o‘yinchini haqorat qilish", "mention_relatives": "qarindoshlarni tilga olish", "insult_relatives": "qarindoshlarni haqorat qilish",
           "gross": "qo‘pol qoidabuzarlik", "bug_abuse": "bagdan foydalanish", "third_party_software": "begona dastur", "repeated": "takroriy qoidabuzarlik"},
    "en": {"char": "character ban", "acc": "account ban", "warn_oral": "verbal warning", "warning": "warning",
           "min": "min", "h": "h", "d": "days", "forever": "permanent", "upto": "up to {n} {u}", "range": "{a}–{b} {u}",
           "range_forever": "from {a} {u} to permanent", "exact": "{n} {u}", "or": " or ", "sep": "; ",
           "insult_player": "insulting a player", "mention_relatives": "mentioning relatives", "insult_relatives": "insulting relatives",
           "gross": "gross violation", "bug_abuse": "bug abuse", "third_party_software": "third-party software", "repeated": "repeated violation"},
    "es": {"char": "bloqueo del personaje", "acc": "bloqueo de la cuenta", "warn_oral": "advertencia verbal", "warning": "advertencia",
           "min": "min", "h": "h", "d": "días", "forever": "permanente", "upto": "hasta {n} {u}", "range": "de {a} a {b} {u}",
           "range_forever": "de {a} {u} a permanente", "exact": "{n} {u}", "or": " o ", "sep": "; ",
           "insult_player": "insultar a un jugador", "mention_relatives": "mencionar a familiares", "insult_relatives": "insultar a familiares",
           "gross": "infracción grave", "bug_abuse": "abuso de bugs", "third_party_software": "software de terceros", "repeated": "infracción reiterada"},
    "pt": {"char": "bloqueio do personagem", "acc": "bloqueio da conta", "warn_oral": "advertência verbal", "warning": "advertência",
           "min": "min", "h": "h", "d": "dias", "forever": "permanente", "upto": "até {n} {u}", "range": "de {a} a {b} {u}",
           "range_forever": "de {a} {u} a permanente", "exact": "{n} {u}", "or": " ou ", "sep": "; ",
           "insult_player": "insultar um jogador", "mention_relatives": "mencionar familiares", "insult_relatives": "insultar familiares",
           "gross": "infração grave", "bug_abuse": "abuso de bugs", "third_party_software": "software de terceiros", "repeated": "infração repetida"},
}


def _unit(u: str | None) -> str | None:
    if not u:
        return None
    u = u.casefold()
    return "min" if u.startswith("минут") else "h" if u.startswith("час") else "d"


def _parse_part(m: re.Match) -> dict:
    kind = m.group(1)
    k = kind.casefold()
    kind = "char" if "персонаж" in k else "acc" if "аккаунт" in k else "warn_oral" if "устное" in k else "warning" if "предупр" in k else kind
    lo, u1, hi, u2, upto, u3, at, u4, ex, u5, forever = m.groups()[1:]
    if lo:
        unit = _unit(u2 or u1)
        if hi == "навсегда":
            return {"type": kind, "range_forever": lo, "unit": unit or "d"}
        return {"type": kind, "range": (lo, hi), "unit": unit}
    if upto:
        return {"type": kind, "upto": upto, "unit": _unit(u3)}
    if at or ex:
        return {"type": kind, "exact": at or ex, "unit": _unit(u4 or u5)}
    if forever:
        return {"type": kind, "forever": True}
    return {"type": kind}


def parse(rule_text: str, max_items: int = 2) -> list[dict]:
    """[{label, parts:[...]}] for the first sanction lines of a rule point."""
    out = []
    for seg in rule_text.split("|")[1:]:
        seg = re.split(r"\s+[IVX]+\.\s|Примечание", seg)[0].strip().rstrip(".")
        label, sep, body = seg.partition(":")
        if not sep:
            continue
        label = label.strip()
        if label not in _LABELS:
            continue
        parts = [_parse_part(m) for m in _PART.finditer(body)]
        if not parts:
            continue
        # body must be fully explained by recognised parts (otherwise do not render)
        leftover = _PART.sub(" ", body)
        if re.search(r"[а-яё]{3,}", re.sub(r"\b(?:или|и)\b", " ", leftover, flags=re.I), re.I):
            if not out:
                return []  # the base sanction is not understood: show nothing rather than only the gross one
            continue
        out.append({"label": _LABELS[label], "parts": parts, "joiner": "or" if " или " in body or "/" in body else "and"})
        if len(out) >= max_items:
            break
    return out


def render(items: list[dict], lang: str) -> str:
    v = L[lang]
    base = L["ru"]
    def word(key):
        return v.get(key, base.get(key, key))
    lines = []
    for it in items:
        rendered = []
        for p in it["parts"]:
            t = word(p["type"])
            u = word(p["unit"]) if p.get("unit") else ""
            if "range" in p:
                rendered.append(f"{t} " + v["range"].format(a=p["range"][0], b=p["range"][1], u=u).strip())
            elif "range_forever" in p:
                rendered.append(f"{t} " + v["range_forever"].format(a=p["range_forever"], u=u).strip())
            elif "upto" in p:
                rendered.append(f"{t} " + v["upto"].format(n=p["upto"], u=u).strip())
            elif "exact" in p:
                rendered.append(f"{t} " + v["exact"].format(n=p["exact"], u=u).strip())
            elif p.get("forever"):
                rendered.append(f"{t} " + v["forever"])
            else:
                rendered.append(t)
        joined = (v["or"] if it["joiner"] == "or" else " + ").join(rendered)
        lines.append(f"{word(it['label'])} — {joined}" if it["label"] else joined)
    return v["sep"].join(lines)


_SEGMENT_CUES = {
    "insult_relatives": r"оск\w*\s*(?:род|мам|мат|семь|отц|пап)|родн|мамк|мать|матер[иь]\b|семь|ona(?:ng|si|mni|dan)|onam|dadam|отбас|анам|mother|family|madre|m[aã]e|família",
    "mention_relatives": r"упом\w*\s*род|упомина",
    "third_party_software": r"софт|чит|cheat|chit|aim|аим|wh\b|вх\b|макрос|скрипт|hack|soft|сборк",
    "bug_abuse": r"баг|bug|bag\b",
}


def _pick_segments(items: list[dict], text: str) -> list[dict]:
    """Keep the single segment that matches the complaint; otherwise the base one.
    'Gross'/'repeated' escalations are never shown on their own."""
    import re as _re
    t = (text or "").casefold()
    rel = [x for x in items if x["label"] in ("mention_relatives", "insult_relatives")]
    if rel and _re.search(_SEGMENT_CUES["insult_relatives"] + "|" + _SEGMENT_CUES["mention_relatives"], t):
        return rel
    for it in items:
        cue = _SEGMENT_CUES.get(it["label"] or "")
        if cue and _re.search(cue, t):
            if it["label"] == "insult_relatives":
                # 'оск родни' vs 'упоминание родных': the rule distinguishes them; show both relative lines
                rel = [x for x in items if x["label"] in ("mention_relatives", "insult_relatives")]
                return rel or [it]
            return [it]
    base = [it for it in items if it["label"] not in ("gross", "repeated", "mention_relatives", "insult_relatives")]
    return base[:1]


def sanction_text(doc, lang: str, text: str = "") -> str:
    try:
        return render(_pick_segments(parse(doc.text, max_items=4), text), lang)
    except Exception:  # never let a parsing quirk break a draft
        return ""
