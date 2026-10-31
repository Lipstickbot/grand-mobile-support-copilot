"""Conservative, explicit heuristics; these are not a trained classifier or gold labels."""
from __future__ import annotations

from collections import OrderedDict
from datetime import timezone
import hashlib
import re
import threading
import unicodedata
from difflib import SequenceMatcher

from .coretypes import RetrievedRule
from .text import analyze_language
from .intents import critical_intents
from .priority import priority_floor


FALLBACKS = {
    "ru": "Уточните ник, сервер, время и обстоятельства обращения. При наличии приложите запись или скриншот без паролей, кодов и полных платёжных данных. Случай требует проверки сотрудником поддержки.",
    "kk": "Ойыншы атын, серверді, уақытты және оқиғаның мән-жайын нақтылаңыз. Бар болса, құпиясөздер, кодтар және толық төлем деректері жоқ бейне не скриншот қосыңыз. Жағдайды қолдау қызметкері тексеруі қажет.",
    "uz": "O‘yinchi nomi, server, vaqt va voqea tafsilotlarini aniqlashtiring. Bo‘lsa, parol, kod va to‘liq to‘lov ma’lumotlarisiz video yoki skrinshot qo‘shing. Holatni yordam xizmati xodimi tekshirishi kerak.",
    "en": "Please specify the player name, server, time and circumstances. If available, attach a recording or screenshot without passwords, codes or full payment details. A support specialist needs to review this case.",
    "es": "Indique el nombre del jugador, servidor, hora y detalles. Si dispone de pruebas, adjunte una captura o grabación sin contraseñas, códigos ni datos de pago completos. El caso requiere revisión del soporte.",
    "pt": "Informe o nome do jogador, servidor, horário e detalhes. Se possível, anexe uma captura ou gravação sem senhas, códigos ou dados de pagamento completos. O caso precisa de análise do suporte.",
}


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().translate(
        str.maketrans({"’": "'", "‘": "'", "ʻ": "'", "ʼ": "'", "`": "'"})).split())


def detect_language(text: str, hint: str | None = None) -> str:
    return analyze_language(text, hint)["language"]


LANGUAGE_CLARIFICATION = "Пожалуйста, уточните язык ответа (ru/kk/uz/en) и опишите, что произошло."


PATTERNS = {
    "payment": r"донат|плат[её]ж|оплат|пополнил|не приш[её]л.*(?:гк|бонус)|to'lov|bonus kelmadi|төлем|төлед|payment|refund|pago|pagamento",
    "account": r"взлом|аккаунт|уч[её]тн|парол|пропал[аи]? (?:машин|имущество)|құпиясөз|кіре алмай|akkaunt|hisobim|parol|account|cuenta|conta",
    "admin": r"админ|модератор|administrat|мут без|\bбан(?:а|ен|или|ят|нул|ьте)?\b|разбан|ban appeal|unban",
    "cheat": r"\bчит\w*|макрос\w*|автоклик\w*|\b(?:macro|autoclick|makros|avtoklik)\w*|аимбот|aimbot|сквозь стен|через стен|запрещ[её]нн.*программ|devordan|\bchit\w*|cheat|бөгде бағдарлам|тыйым салынған бағдарлам|taqiqlangan dastur",
    "scam": r"мошенн|обман|не отдал|не вернул|вымога|алда|алаяқ|firibgar|aldadi|scam",
    "rmt_ads": r"за реал|реальн.*деньг|прода[её]т аккаунт|реклам|rmt|real pul|haqiqiy pul|нақты ақша",
    "insult": r"оскорб|обзыв|токсич|расизм|унижа\w*.{0,30}(?:меня|родствен|родител)|балағат|қорла|кемсіт|тіл тигіз|haqorat|kamsit|\bso'k|insult|call\w*.{0,35}(?:disgusting|names)|abuse\w*.{0,25}(?:family|parents)",
    "dm": r"\bдм\b|без причин.*уб|убил.*без причин|меня уб|зел[её]н.*зон|\bgz\b|өлтір|o'ldir|жасыл аймақ|yashil hudud",
    "help": r"застр|баг|текстур|телепорт|қате|bug|stuck",
}


def topics_for(text: str) -> set[str]:
    t = normalized(text)
    topics = {key for key, pattern in PATTERNS.items() if re.search(pattern, t)} | critical_intents(text)
    if "rmt_ads" in topics and not re.search(r"взлом|восстанов|кіре алмай|құпиясөз|парол|kira olmay|parol", t):
        topics.discard("account")  # Trading an account is not automatically an access-loss report.
    return topics


def _explicitly_unrelated_topics(text):
    # Narrow denials of what this request is about, not a judgment of guilt.
    denied = set()
    for topic, word in (("cheat", r"cheat\w*|чит\w*"), ("insult", r"insult\w*|оскорб\w*")):
        if re.search(r"(?:not\s+(?:reporting|alleging)|не\s+(?:сообщаю|жалуюсь))\s+(?:about\s+|о\s+)?(?:"+word+r")", text):
            denied.add(topic)
    return denied


def classify(text: str, kind: str) -> tuple[str, str, list[str]]:
    t = normalized(text)
    topics = topics_for(text)
    if kind == "report":
        if "payment" in topics:
            category = "report.payment"
        elif "account" in topics:
            category = "report.account"
        elif topics & {"admin", "cheat", "scam", "rmt_ads", "insult", "dm"}:
            category = "report.complaint"
        elif "help" in topics or re.search(r"помог|көмек|yordam|help", t):
            category = "report.help"
        elif re.search(r"как|где|почему|қалай|қайда|qanday|qayer|how|where|\?", t):
            category = "report.question"
        else:
            category = "report.offtopic"
    else:
        topic = next((key for key in ("admin", "rmt_ads", "cheat", "scam", "insult", "dm") if key in topics), None)
        category = "complaint." + (topic if topic else "invalid" if len(t) < 18 else "other")
    # Mandatory topics are independent of the chosen category and never depend on an admin reply.
    priority = priority_floor(t, topics)
    reasons = ["classification=keyword_heuristic_not_calibrated", "human_review=conservative_baseline"]
    if topics & {"account", "payment", "admin"}:
        reasons.append("mandatory_human_topic=" + ",".join(sorted(topics & {"account", "payment", "admin"})))
    return category, priority, reasons


_EVENT_SCOPE = {
    36: r"семейн\w*\s+(?:мероприят\w*|битв\w*|войн\w*|мп)|семь[яи]\b.{0,40}мероприят\w*|отбас\w*.{0,28}(?:іс[ -]?шара\w*|шайқас\w*|соғыс\w*)|\boila\w*.{0,25}(?:tadbir\w*|jang\w*|urush\w*)|\bfamily\s+(?:event|battle|war)\w*",
    41: r"бизнес\w*|\bbizwar\b|\bbiznes\w*|\bbusiness\s+(?:battle|war)\w*",
    42: r"фургон\w*|\bfurgon\w*|\bvan\b",
}
_REFERENCED_EVENT = re.compile(r"цитир\w*|ссыла\w*|ссылк\w*|ережесін\w*.{0,20}сүйен\w*|тыйым\w*.{0,20}келтір\w*|qoid\w*.{0,20}tayan\w*|taqiq\w*.{0,20}keltir\w*")


def resolve_event_scope(text: str, events=None) -> set[int]:
    """Resolve explicit negations/references; ambiguous comparisons stay unresolved.

    This is a multilingual surface heuristic, not semantic entailment. It never
    infers guilt or borrows permissions from a different event.
    """
    t = normalized(text)
    mentioned = {sid for sid, pattern in _EVENT_SCOPE.items() if re.search(pattern, t)}
    mentioned |= set(events or ()) & set(_EVENT_SCOPE)
    if len(mentioned) <= 1:
        return mentioned
    positive, emphasized = set(), set()
    for clause in re.split(r"[.!?;,\n]+", t):
        if _REFERENCED_EVENT.search(clause):
            continue
        active = set()
        for sid, pattern in _EVENT_SCOPE.items():
            for match in re.finditer(pattern, clause):
                before, after = clause[max(0, match.start()-40):match.start()], clause[match.end():]
                negated = re.search(r"\bне\s+(?:(?:на|в|во|про|о)\s+)?$|\bне\s+(?:битв\w*|захват\w*|войн\w*)(?:\s+(?:за|на))?\s+$|\bnot\s+(?:(?:a|the|in|at|about|for)\s+)*$", before)
                negated = negated or re.match(r"\s+(?:(?:алу|jang\w*|olish)\s+)?(?:емес|emas)\b", after)
                if not negated:
                    active.add(sid)
        positive |= active
        if re.search(r"\b(?:именно|дәл|aynan)\b", clause):
            emphasized |= active
    if len(positive) == 1:
        return positive
    if len(emphasized) == 1:
        return emphasized
    return set()  # No silently chosen top hit for an unresolved multi-event query.


def select_relevant_rules(text: str, candidates: list[RetrievedRule], *, cues=None, events=None) -> list[RetrievedRule]:
    """Conservative topic gate after retrieval; do not fill missing payment/FAQ knowledge with game rules.

    This explicit whitelist sacrifices recall; it does not replace an entailment model or review.
    """
    t = normalized(text)
    topics = topics_for(text) | (set(cues or ()) & {"insult", "cheat", "scam"})
    topics -= _explicitly_unrelated_topics(t)
    if "rmt" in (cues or ()):
        topics.add("rmt_ads")
        if not re.search(r"взлом|восстанов|кіре алмай|құпиясөз|парол|kira olmay|parol", t):
            topics.discard("account")
    if "trade_dispute" in (cues or ()):
        topics.add("scam")  # Potentially applicable rule, never a fraud verdict.
    if topics & {"payment", "account"}:
        return []
    allowed = set()
    for topic, refs in {
        "insult": ("2.1",), "cheat": ("5.2",), "scam": ("4.1",),
    }.items():
        if topic in topics:
            allowed.update("rules:30:" + ref for ref in refs)
    procedural = set()
    if re.search(r"как.{0,35}(?:оформ|подать|жалоб)|оформлени|требован.{0,20}(?:жалоб|доказател)|шағым.{0,25}(?:рәсім|талап)|(?:qanday|qayer).{0,30}shikoyat", t):
        procedural.update("rules:31:" + ref for ref in ("1.3", "1.3.1", "1.3.2", "1.3.3"))
    sections = set()
    if re.search(r"расизм|ксенофоб|религи|нәсіл|ұлтшыл|\birq", t):
        allowed.add("rules:30:2.4")
    if re.search(r"докс|личн.*информ|угрож.*жизн|мекенжай|haqiqiy hayot", t):
        allowed.add("rules:30:2.6")
    if re.search(r"вымога|қорқытып|tovlamach", t):
        allowed.add("rules:30:4.2")
    if re.search(r"реклам|reklam|жарнама", t):
        allowed.add("rules:30:2.3")
    if "rmt" in (cues or ()) or re.search(r"за реал|реальн.*деньг|real pul|haqiqiy pul|нақты ақша|\brmt\b", t):
        allowed.add("rules:30:4.4")
    if re.search(r"зел[её]н\w* зон|\bgz\b|green zone|жасыл (?:аймақ|аумақ)|yashil (?:hudud|zona)", t):
        entry_question = re.search(r"вход|зайти|войти|забеж|вбеж|забега|вбега|\benter\w*|\brun\w*\s+into|кіру|\bkirish", t)
        if not entry_question and re.search(r"огонь|стрел|стрельб|убий|урон|оқ\s*ат|атыс|шабуыл|\b(?:fire|shoot\w*|attack\w*)\b|o'q|otish|hujum", t):
            allowed.add("rules:32:1.1")
        else:
            sections.add(32)
    if "armor" in (cues or ()) or re.search(r"бронежилет|брон(?:я|ю|ь)\b|сауыт|кеудеше|bronejilet|bronijilet|zirh|\b(?:body\s+)?armou?r\b", t):
        matched = resolve_event_scope(text, events)
        if len(matched) == 1:
            allowed.add({36: "rules:36:1.23", 41: "rules:41:2.8", 42: "rules:42:1.7"}[next(iter(matched))])
    substantive = [rule for rule in candidates if rule.canonical_id in allowed or rule.section_id in sections]
    return substantive + [rule for rule in candidates if rule.canonical_id in procedural and rule not in substantive]


_SECRET = re.compile(r"парол\w*|одноразов\w*\s+код\w*|код\w*\s+(?:из\s+)?(?:смс|sms|письма|подтвержден\w*)|password\w*|\botp\b|\bcvv\b|\bcvc\b|\bpin\b|полны\w*.{0,20}(?:номер\w*\s+карт\w*|плат[её]жн\w*\s+данн\w*)|құпиясөз\w*|растау\s+код\w*|бір\s+реттік\s+код\w*|parol\w*|tasdiqlash\s+kod\w*|bir\s+martalik\s+kod\w*|sms\s+kod\w*")
_REQUEST_VERB = re.compile(r"\b(?:укажите|указывайте|сообщите|сообщайте|пришлите|присылайте|отправьте|отправляйте|напишите|пишите|назовите|называйте|передайте|передавайте|вышлите|поделитесь|скиньте|покажите|send|provide|share|tell|give|жібер\w*|айтыңыз|айтпа\w*|беріңіз|берме\w*|көрсетіңіз|жазыңыз|жазба\w*|yubor\w*|jo'nat\w*|ayting|aytmang|bering|bermang|yozing|yozmang)\b")


def _requests_secret(text: str) -> bool:
    # Prefer the closest request verb to each secret. This distinguishes
    # "Укажите сервер, пароль не сообщайте" from asking for the password itself.
    for clause in re.split(r"[.!?;\n]+", text):
        verbs = list(_REQUEST_VERB.finditer(clause))
        credential_bundles = re.finditer(r"данн\w*\s+для\s+(?:вход\w*|авторизац\w*)|(?:login|authentication|access)\s+(?:credentials|details)|кіру\s+деректер\w*|kirish\s+ma'lumot\w*", clause)
        for secret in list(_SECRET.finditer(clause)) + list(credential_bundles):
            nearby = [(min(abs(secret.start()-verb.end()), abs(verb.start()-secret.end())), verb) for verb in verbs]
            if not nearby:
                continue
            distance, verb = min(nearby, key=lambda pair: pair[0])
            if distance > 80:
                continue
            before = clause[max(0, verb.start()-28):verb.start()]
            negative = bool(re.search(r"(?:\bне|\bnever|\bdon't|\bdo not)\s*$", before))
            negative |= bool(re.search(r"^(?:жіберме|айтпа|берме|жазба)|(?:mang|ma)\b", verb.group()))
            if not negative:
                return True
    return False


_GUILT_ASSERTION = re.compile(
    r"\b(?:игрок|пользователь)\s+(?:(?:точно|однозначно)\s+)?(?:виновен|нарушил|использовал\s+читы)\b|"
    r"\b(?:the\s+)?player\s+(?:is\s+(?:definitely\s+)?guilty|(?:definitely\s+)?violated|used\s+cheats)\b|"
    r"ойыншы\s+(?:анық\s+)?кінәлі|кінәсі\s+дәлелденді|"
    r"\bo'yinchi\s+(?:aniq\s+)?(?:aybdor|qoidani\s+buzdi|haqorat\s+qildi)\b")
_ALLEGATION_QUALIFIER = re.compile(
    r"если|возможн|предполож|по\s+(?:ваш\w*\s+)?(?:описан|слов)|вы\s+(?:сообщ|пиш)|"
    r"\b(?:if|may|might|allegedly)\b|according\s+to|you\s+(?:report|say|state)|"
    r"мүмкін|егер|\bболса\b|сөзіңіз|айтуыңыз|хабарла\w*|сіз\s+айт|"
    r"\b(?:ehtimol|bo'lsa)\b|aytishingiz|xabar\s+ber\w*")


def _asserts_guilt(text: str) -> bool:
    for clause in re.split(r"[.!?;\n]+", text):
        if _GUILT_ASSERTION.search(clause) and not _ALLEGATION_QUALIFIER.search(clause):
            return True
    return False


def guard_draft(raw_draft: str, rules: list[RetrievedRule], lang: str, *, player_text: str | None = None) -> tuple[str, list[str]]:
    """Reject observable unsafe output; a passing guard is not proof of correctness."""
    t = normalized(raw_draft)
    issues = []
    if not t or len(raw_draft) > 4000:
        issues.append("draft_empty_or_oversized")
    if any(token in t for token in ("<think>", "</think>", "<|", "system:", "assistant:", "```")):
        issues.append("draft_control_tokens")
    if re.search(r"(?:я|мы)\s+(?:уже\s+)?(?:проверил|просмотрел|изучил|забан|вернул|разбан|восстановил)|"
                 r"(?:скриншот|видео|доказательств).{0,25}(?:подтвержда|проверен|изучен)|"
                 r"будет (?:наказан|забанен|возвращ|разбанен)|(?:гарантиру|обещаю)|"
                 r"(?:возврат|разбан|телепорт).{0,18}(?:выполнен|сделан)|"
                 r"(?:i|we) (?:have |will |already )?(?:banned|refund|unban|reviewed|verified)|"
                 r"(?:ақшаңыз|қаражат).{0,20}қайтарылады|жазаланады|бұғатталады|"
                 r"pul.{0,20}qaytariladi|jazolanadi|bloklanadi|"
                 r"\b(?:mute|ban|demorgan|warn|kick|мут|бан|деморган|варн)\b.{0,28}(?:qilinadi|beriladi|qo'llaniladi|беріледі|салынады)|"
                 r"(?:игрок|пользователь).{0,25}получит.{0,18}(?:мут|бан|варн|наказан)|"
                 r"будет.{0,12}(?:выдан|назначен|вынесен).{0,18}(?:мут|бан|варн|наказан)", t):
        issues.append("draft_action_or_evidence_claim")
    if re.search(r"(?:на (?:скриншоте|видео|записи)|из (?:скриншота|записи)).{0,30}(?:видно|ясно|следует)|"
                 r"(?:видеозапись|доказательства|скриншот).{0,35}(?:позволя|подтвержда|показыва)|"
                 r"(?:ваша )?жалоба.{0,25}(?:отклонена|не может быть рассмотрена)|"
                 r"(?:игрок|пользователь).{0,20}(?:виновен|точно нарушил)|"
                 r"(?:накажем|забаним|верн[её]м (?:деньги|средства)|разбаним)|"
                 r"скриншот\w*.{0,25}(?:доказыва|подтвержда)|"
                 r"\b(?:i|we)\s+(?:(?:have|already)\s+){0,2}(?:reviewed|checked|verified|watched).{0,35}(?:image|screenshot|video)|"
                 r"(?:screenshot|image|video).{0,25}(?:proves|confirms|clearly shows)|"
                 r"(?:скриншот|бейне).{0,35}(?:қарап шықтым|көрдім|тексердім|дәлелдейді)|"
                 r"(?:skrinshot|video|tasvir).{0,35}(?:ko'rib chiqdim|tekshirdim|ko'rdim|isbotlaydi|tasdiqlaydi)", t) or _asserts_guilt(t):
        issues.append("draft_unverified_verdict_or_evidence")
    if _requests_secret(t):
        issues.append("draft_secret_request")
    allowed = {r.canonical_id for r in rules if r.ref is not None and r.source_type == "rule"}
    cited = set(re.findall(r"(?:rules:\d+:[\d.]+|internal:[\w:.-]+)", raw_draft))
    if not cited <= allowed:
        issues.append("draft_unknown_or_internal_citation")
    # Ordinary [player names] are not citations. Authority-like and numeric
    # markers must be full supplied canonical IDs, never fabricated placeholders.
    for marker in re.findall(r"\[([^\]\n]{1,100})\]", raw_draft):
        marker = marker.strip()
        if (re.match(r"^(?:rules?[:_ -]|complaint\.|report\.|internal:|правило[_ :#-]|пункт[_ :#-])", marker, re.I)
                or re.fullmatch(r"\d+(?:[.-]\d+)*", marker)) and marker not in allowed:
            issues.append("draft_noncanonical_citation")
            break
    # A request for clarification need not cite a rule. An actual statement of
    # permission/prohibition must have a supplied source. Citation coverage is
    # measured separately and not manufactured for a safe intake response.
    rule_claim = re.search(r"запрещ|разреш|допустим|рұқсат|тыйым|тақиқ|taqiqlan|ruxsat|forbidden|prohibited|permitted|allowed", t)
    if rule_claim and not cited.intersection(allowed):
        issues.append("draft_missing_rule_citation")
    # Staff decide penalties. Reject even plausible durations rather than risk
    # choosing the wrong sanction branch of the same rule.
    sanction = r"(?:\b(?:mute|muted|ban|banned|suspend\w*|suspension|block\w*|jazo\w*|jazola\w*|blokla\w*)\b|мут\w*|бан\w*|блокиров\w*|жаза\w*|бұғат\w*)"
    duration = r"\d+\s*(?:[-–—]\s*\d+\s*)?(?:мин\w*|час\w*|дн\w*|день|сут\w*|тәулік\w*|күн\w*|сағат\w*|minutes?|hours?|days?|daqiqa\w*|soat\w*|kun\w*)"
    if (re.search(sanction + r".{0,40}" + duration, t)
            or re.search(r"\b(?:should|must|shall|deserves?)\b.{0,30}" + sanction, t)
            or re.search(r"(?:следует|нужно|долж\w*|необходимо).{0,35}(?:наказать|забанить|выдать\s+мут|назначить\s+бан)", t)
            or re.search(sanction + r".{0,25}(?:керек|қажет|лайық|lozim|kerak)", t)):
        issues.append("draft_sanction_recommendation")
    if player_text:
        answer = re.sub(r"\[rules:\d+:[\d.]+\]", "", t)
        source_words = normalized(player_text).split()[:512]
        answer_words = answer.split()[:512]
        if len(source_words) >= 10 and SequenceMatcher(None, source_words, answer_words, autojunk=False).ratio() >= .9:
            issues.append("draft_question_echo")
    # Plain decimal references cannot identify a section in this corpus. Require namespaced IDs.
    if re.search(r"(?:пункт\w*|правил\w*|п\.|rule|тармақ\w*|band)\s*(?:№\s*)?\d+\.\d+", t):
        issues.append("draft_ambiguous_numbered_citation")
    detected = detect_language(raw_draft)
    if lang in {"ru", "kk", "uz", "en"} and detected != lang:
        issues.append("draft_language_unverified")
    if lang not in FALLBACKS:
        issues.append("draft_language_unsupported")
    return (FALLBACKS.get(lang, FALLBACKS["ru"]), issues) if issues else (raw_draft.strip(), [])


class DuplicateTracker:
    """Session-only exact duplicates with caller identity and timestamps, bounded to 4096 keys."""
    def __init__(self):
        self._seen = OrderedDict()
        self._lock = threading.Lock()

    def check(self, reporter: str | None, timestamp, kind: str, text: str) -> bool:
        if not reporter or timestamp is None or timestamp.tzinfo is None:
            return False
        ts = timestamp.astimezone(timezone.utc).timestamp()
        key = (reporter, kind, hashlib.sha256(normalized(text).encode("utf8")).digest())
        with self._lock:
            previous = self._seen.get(key)
            duplicate = previous is not None and 0 <= ts - previous <= 3600
            self._seen[key] = max(ts, previous if previous is not None else ts)
            self._seen.move_to_end(key)
            while len(self._seen) > 4096:
                self._seen.popitem(last=False)
        return duplicate
