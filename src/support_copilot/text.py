"""Cheap, explicit input views: legacy Kazakh, domain typos and language evidence.

These heuristics are not a language model or calibrated confidence. Returned
normalization views never become verified facts and never modify the request.
"""
import re
import unicodedata

_LEGACY = "ќґѕєјўї"
_RESTORED = "қғңәөұү"
_TABLE = str.maketrans(_LEGACY + _LEGACY.upper(), _RESTORED + _RESTORED.upper())
_APOSTROPHES = str.maketrans({char: "'" for char in "‘’ʻʼ`"})
_WORDS = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)
_LANGUAGES = ("ru", "kk", "uz", "en", "es", "pt")
_KAZAKH = re.compile(r"[әғқңөұүһі]")
_RUSSIAN = re.compile(r"[а-яё]")
_LATIN = re.compile(r"[a-z]")
# None of these rules edits reference numbers, negations or arbitrary text.
# This is deliberately an exact vocabulary, not edit-distance spell correction.
_TYPO = {
    "оскарбил": "оскорбил", "оскарбляет": "оскорбляет", "аскорбил": "оскорбил",
    "оскабил": "оскорбил", "читерр": "читер", "читор": "читер",
    "аккаут": "аккаунт", "акаунт": "аккаунт", "пороль": "пароль",
    "доннат": "донат", "застрялл": "застрял", "памагите": "помогите",
    "помагите": "помогите", "помогити": "помогите", "обманулл": "обманул",
    "комек": "көмек", "комектес": "көмектес", "балагаттады": "балағаттады",
    "yordm": "yordam", "haqort": "haqorat", "shkoyat": "shikoyat",
    "hlep": "help", "acount": "account", "paymnt": "payment",
}
_TYPO_STEMS = (("оскарбл", "оскорбл"), ("радител", "родител"),
               ("балагат", "балағат"), ("корла", "қорла"))


def _correct_word(word):
    if word in _TYPO:
        return _TYPO[word]
    for incorrect, corrected in _TYPO_STEMS:
        if word.startswith(incorrect) and len(word) > len(incorrect):
            return corrected + word[len(incorrect):]
    return word
_METADATA = re.compile(
    r"https?://\S+|www\.\S+|[\w.+-]+@[\w.-]+\.[^\W\d_]+|"
    r"(?<!\w)@[\w.-]+|\b(?:rules:\d+:[\d.]+|internal:[\w:.-]+)\b|"
    r"\b[^\W\d_][\w.-]*_[\w.-]+\b|"
    r"(?:ник(?:нейм)?|nickname|nick)\s*[:=]\s*[^\s,;]+|"
    r"<\|[^>\n]{0,100}\|>|</?(?:think|system|assistant|user|developer)[^>\n]*>|"
    r"<(?:url|email|handle|digits|nick)>|```[\s\S]*?```",
    re.I,
)
_QUOTED_SOURCE = re.compile(
    r"(?:правил\w*|пункт\w*|источник\w*|ереж\w*|qoid\w*|source|rule|quote)"
    r"[\s:№\d.=-]{0,30}(?:«[^»]{1,2000}»|\"[^\"]{1,2000}\")|(?m:^\s*>[^\n]*)",
    re.I,
)

_LEXICAL = {
    "ru": "пожалуйста привет здравствуйте помогите помоги помощь спасибо уточните почему где как кто когда меня мне мое моя мой мои мы вы вам вас это сегодня вчера нельзя можно пожалуйста доказательства не нет после чужой другой именно правильно запрет вход доступ бан",
    "kk": "маған мені менде менің мен ойын ойыншы көмек көмектес көмектесіңіз бүгін шағым тіркелгі жаза уақыт серверді жағдай қолдау болды келмеді сәлем осы бұл үшін бірақ және қалай қайда емес жоқ еді алдым сауыт бронь рұқсат нақты",
    "uz": "iltimos yordam bugun men meni menga ammo lekin bo'lsa qildim qildi qiling qayer qanday qachon nima uchun bilan va yo'q emas mumkin xodim yozing tashkilot ruxsat so'ng parol haqorat",
    "en": "please hello hi help thanks thank why where how who when what my me i we you your our is are was were has have had can could would should do does did not never without someone anyone because after before since the a an of to from with",
    "es": "ayuda jugador cuenta pago contraseña gracias hola necesito dónde cómo puedo estoy favor",
    "pt": "ajuda jogador conta senha pagamento obrigado olá preciso onde como estou favor",
}
_PREFIXES = {
    "ru": ("уточн", "оскорб", "обзыв", "нецензур", "родител", "помог", "застр", "взлом", "пропал", "оплат", "платеж", "платёж", "купил", "покуп", "обман", "мошенн", "переда", "сообщ", "отправ", "провер", "рассмотр", "обращ", "забан", "наказан", "наруш", "использ", "читер", "деньг", "скриншот", "запрещ", "разреш", "персон", "получ", "присыл", "пришл", "обстоят", "аккаунт", "восстанов", "доступ", "потер", "вход", "брон", "фургон", "цитир", "семейн", "мероприят", "склад", "родн"),
    "kk": ("ойын", "балағат", "қорла", "шағым", "көмек", "құпиясөз", "дерек", "төлем", "төле", "тексер", "нақтыла", "жібер", "қызмет", "ереже", "тармақ", "бейне", "жасыл", "отбас", "шайқас", "қатыс", "тіркел", "тыйым", "жазал", "қолдан", "бағдарлам", "келтір"),
    "uz": ("o'yin", "to'lov", "hisob", "haqorat", "shikoyat", "ma'lumot", "tekshir", "qoida", "vaqtni", "serverni", "parol", "xodim", "taqiq", "jazo", "holat", "voqea", "bo'yicha", "yubor", "jo'nat", "aniqla", "firibgar", "tashkilot", "iltimos"),
    "en": ("account", "payment", "refund", "insult", "stolen", "hacked", "screenshot", "evidence", "explain", "review", "specify", "clarif", "circumstance", "support", "banned", "appeal", "password", "recording", "attach", "happen", "threat", "someone", "offens", "player", "forbid", "permit", "allow"),
    "es": ("ayud", "jugador", "contraseña", "cuenta", "pagam", "necesit", "revis", "adjunt", "indique"),
    "pt": ("ajud", "jogador", "senha", "pagamento", "precis", "análise", "informe", "anexe"),
}
_EXACT = {}
for _language, _terms in _LEXICAL.items():
    for _term in _terms.split():
        _EXACT.setdefault(_term, {})[_language] = 1.5


def normalize_retrieval_query(text: str) -> tuple[str, list[str]]:
    """Exact domain typo query view; do not use this as an edited player claim.

    Structured/explicitly labeled nicknames, links, handles and source IDs are
    protected. Unmarked arbitrary proper names cannot be universally identified;
    only a fixed list of ordinary misspelled domain words is eligible.
    """
    if not isinstance(text, str):
        raise TypeError("Query must be text")
    flags = set()

    def correct(segment):
        def replace(match):
            word = match.group()
            target = _correct_word(word.casefold())
            if target == word.casefold() or not (word.islower() or word.isupper() or word.istitle()):
                return word
            flags.add("domain_typo_query_view")
            if word.isupper():
                return target.upper()
            return target.capitalize() if word.istitle() else target
        return _WORDS.sub(replace, segment)

    parts, start = [], 0
    for match in _METADATA.finditer(text):
        parts.extend((correct(text[start:match.start()]), match.group()))
        start = match.end()
    parts.append(correct(text[start:]))
    return "".join(parts), sorted(flags)


def restore_kazakh(text: str, hint: str | None = None, region: str | None = None) -> tuple[str, bool]:
    """Do not globally turn Ukrainian/Belarusian letters into Kazakh.

    Restore only with a KK/KZ context or a distinctive legacy-character signal.
    Original request objects are never modified; the caller records the change.
    """
    if not any(char.casefold() in _LEGACY for char in text):
        return text, False
    context = (hint or "").casefold() == "kk" or (region or "").casefold() == "kz"
    distinctive = bool(re.search(r"[ќѕјў]", text.casefold()))
    mixed_kazakh = bool(re.search(r"[әғқңөұүһ]", text.casefold()))
    if not (context or distinctive or mixed_kazakh):
        return text, False
    result = text.translate(_TABLE)
    return result, result != text


def analyze_language(text: str, hint: str | None = None, region: str | None = None) -> dict:
    """Return bounded language evidence without another model or network call.

    A server hint can break a genuine evidence tie; it cannot turn emoji, IDs or
    unknown short strings into a recognized language. ``normalized_text`` is the
    complete original text with only context-supported legacy restoration. Masks
    used for detection are NEVER applied to that returned view.
    """
    if not isinstance(text, str):
        raise TypeError("Language input must be text")
    restored, changed = restore_kazakh(text, hint, region)
    normalized = unicodedata.normalize("NFKC", restored).casefold().translate(_APOSTROPHES)
    detection = _QUOTED_SOURCE.sub(" ", normalized)
    detection = _METADATA.sub(" ", detection)
    flags = ["legacy_kazakh_restored"] if changed else []
    signals = ["language_heuristic_not_calibrated"]
    if detection != normalized:
        signals.append("detection_ignored_metadata_or_quoted_source")
    scores = dict.fromkeys(_LANGUAGES, 0.)
    strong = dict.fromkeys(_LANGUAGES, 0)
    weak_kazakh_tokens = 0
    for original in _WORDS.findall(detection):
        word = _correct_word(original)
        votes = dict(_EXACT.get(word, {}))
        if len(word) < 2:
            continue
        for language, prefixes in _PREFIXES.items():
            if word.startswith(prefixes):
                votes[language] = max(votes.get(language, 0.), 1.5)
        if _KAZAKH.search(word) and len(word) >= 3:
            votes["kk"] = max(votes.get("kk", 0.), .8)
            weak_kazakh_tokens += 1
        for language, value in votes.items():
            scores[language] += value
            strong[language] += value >= 1.2
        if _LATIN.search(word) and _RUSSIAN.search(word):
            signals.append("mixed_script_token")
    credible = [language for language in _LANGUAGES if scores[language] >= 1.2 and
                (strong[language] or language == "kk" and weak_kazakh_tokens >= 2)]
    candidates = sorted(credible, key=lambda language: (-scores[language], _LANGUAGES.index(language)))
    for language in candidates:
        signals.append(language + "_lexical_evidence" if strong[language] else "kk_distinctive_script_evidence")
    mixed = len(candidates) > 1
    if mixed:
        signals.append("mixed_language_evidence")
    language = "und"
    ambiguous = not candidates
    if candidates:
        top = candidates[0]
        tied = [candidate for candidate in candidates if scores[candidate] >= scores[top] * .85]
        if len(tied) == 1:
            language = top
        elif hint in tied:
            language = hint
            ambiguous = True
            signals.append("server_hint_breaks_evidence_tie")
        else:
            ambiguous = True
            signals.append("primary_language_ambiguous")
    else:
        signals.append("insufficient_language_evidence")
    return {"language": language, "candidates": candidates, "signals": sorted(set(signals)),
            "mixed": mixed, "ambiguous": ambiguous, "normalized_text": restored,
            "normalization_flags": flags}
