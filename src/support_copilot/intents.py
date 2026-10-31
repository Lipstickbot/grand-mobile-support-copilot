"""Bounded high-urgency paraphrase cues, not factual verification or an LLM.

This complements the keyword classifier. A match means a *reported* account
access/payment problem deserves human attention. It never establishes compromise,
successful payment or entitlement to a refund. No data/model/network access.
"""
from __future__ import annotations

import re
import unicodedata

_APOSTROPHES = str.maketrans({char: "'" for char in "‘’ʻʼ`"})
_SOURCE_QUOTE = re.compile(
    r"(?:пример\w*|цитат\w*|правил\w*|пункт\w*|мысал\w*|ереж\w*|"
    r"qoida\w*|misol\w*|example|quote|rule)"
    r"[\s:№\d.=-]{0,30}(?:«[^»]{1,2000}»|\"[^\"]{1,2000}\")|"
    r"(?m:^\s*>[^\n]*)|```[\s\S]*?```",
    re.I,
)
_HYPOTHETICAL = re.compile(
    r"^(?:что\s+(?:будет\s+)?если|что\s+делать[, ]+если|если\b|"
    r"егер\b|agar\b|what\s+if\b|if\b)|"
    r"^(?:како\w*\s+(?:пункт|правил)|which\s+rule|what\s+rule|"
    r"қай\s+ереже|qaysi\s+qoida)|"
    r"^(?:инструкци\w*|памятк\w*|нұсқаулық|yo'riqnoma|support\s+instructions?|instructions?)\s*:|"
    r"^(?:обратитесь|сообщите|contact|please\s+contact|report|хабарласыңыз|yuboring)\b"
    r".{0,150}\b(?:если|if|егер|agar)\b", re.I,
)
_NEGATED_BEFORE = re.compile(
    r"(?:\bне|\bnot|\bnever|\bdidn't|\bhasn't|\bhaven't|\bwasn't|\bweren't|"
    r"\bdoesn't|\bdon't|\bisn't|\baren't|\bcannot|\bcan't)\s+(?:\w+\s+){0,2}$"
)
_NEGATED_AFTER = re.compile(r"^\s+(?:жоқ|емес|emas|yo'q)\b")

# Each account pattern requires an owner-bearing asset and either foreign access
# or a concrete access-loss statement. Mere account/security discussion is not
# sufficient. Languages can be mixed inside a clause.
_ACCOUNT = [
    (
        r"\b(?:чуж\w*|посторонн\w*|неизвестн\w*|кто-то|другой\s+человек)\b",
        r"\bмо(?:й|я|е|ё|его|ему|им|их|ю|[её]м|ей)\b.{0,25}(?:персонаж|профил|уч[её]т|аккаунт)",
        r"получил\w*\s+доступ|вош[её]л|вошли|заш[её]л|заходил|проник\w*|завладел\w*|играет|играют|играл",
    ),
    (
        r"\b(?:бөгде|бөтен|бейтаныс|біреу|басқа\s+адам|өзге\s+адам)\b",
        r"менің.{0,30}(?:кейіпкер|профил|тіркелгі|аккаунт)|кейіпкерім|профилім|тіркелгім|аккаунтым",
        r"қол\s+жеткізіп\s+(?:алды|алған)|қол\s+жеткізді|кіріп\s+(?:алды|кетті)|кірген|кірді|ойнап\s+(?:отыр|жүр)|ойнайды",
    ),
    (
        r"\b(?:begona|notanish|kimdir|boshqa\s+(?:odam|kishi))\b",
        r"\b(?:qahramonim|personajim|profilim|hisobim|akkauntim)\w*",
        r"kirish\s+(?:huquqini|imkonini)\s+oldi|kirib\s+oldi|kirgan|kirdi|egallab\s+oldi|o'ynayapti|o'ynayotgan|foydalanyapti|foydalanmoqda",
    ),
    (
        r"\b(?:someone|somebody|stranger|unknown\s+(?:person|user)|another\s+person)\b",
        r"\b(?:my|our)\s+(?:game\s+)?(?:character|profile|account|login)\b",
        r"(?:got|gained|has|have)\s+access|logged\s+in|signed\s+in|took\s+over|using|playing\s+(?:on|with|as)",
    ),
]
_LOCKOUT = [
    (r"\bмо(?:й|я|е|ё|его|ему|им|их|ю|[её]м|ей)\b.{0,25}(?:персонаж|профил|уч[её]т|аккаунт)",
     r"потерял\w*\s+доступ|пропал\s+доступ|не\s+могу\s+(?:войти|зайти|попасть)"),
    (r"менің.{0,30}(?:кейіпкер|профил|тіркелгі|аккаунт)|кейіпкерім|профилім|тіркелгім|аккаунтым",
     r"кіре\s+алмаймын|кіре\s+алмай\s+қалдым|қолжетімділікті\s+жоғалттым"),
    (r"\b(?:qahramonim|personajim|profilim|hisobim|akkauntim)\w*",
     r"kira\s+olmayapman|kira\s+olmay\s+qoldim|kirish\s+huquqini\s+yo'qotdim"),
    (r"\bmy\s+(?:game\s+)?(?:character|profile|account|login)\b",
     r"locked\s+out|lost\s+access|(?:can't|cannot|can\s+not)\s+(?:log|sign)\s+in"),
]

# A changed first-person sign-in identity is meaningful even when the player
# never says 'account'. Require a past change AND explicit lack of consent.
_CHANGED_ACCESS = [
    (r"данн\w*\s+(?:для\s+)?входа|почт\w*|логин\w*|парол\w*",
     r"изменил\w*|сменил\w*|поменял\w*|измен[её]н\w*",
     r"без\s+моего\s+(?:согласия|разрешения)|я\s+не\s+разрешал\w*|меня\s+не\s+спросил\w*"),
    (r"кіру\s+деректер\w*|логин\w*|пошта\w*|құпиясөз\w*",
     r"ауыстыр(?:ып|ған|ды|ылған)\w*|өзгерт(?:кен|іп|ті|ілген)\w*",
     r"келісімімсіз|рұқсатымсыз|мен\s+рұқсат\s+бермедім"),
    (r"kirish\s+ma'lumot\w*|parol\w*|login\w*|pochta\w*",
     r"o'zgartir(?:ishgan|gan|ildi|ib)\w*|almashtir(?:ishgan|gan|ildi|ib)\w*|o'zgardi",
     r"roziligimsiz|ruxsatimsiz|men\s+ruxsat\s+bermadim"),
    (r"(?:sign[ -]?in|log[ -]?in)\s+(?:details|credentials)|password|email(?:\s+address)?",
     r"\b(?:changed|reset|replaced)\b",
     r"without\s+my\s+(?:permission|consent)|i\s+(?:did\s+not|didn't)\s+(?:authorize|permit)"),
]
_ALLOWED_ACCESS = re.compile(
    r"с\s+моего\s+разрешения|я\s+сам\s+разрешил\w*|"
    r"менің\s+рұқсатыммен|өзім\s+рұқсат\s+бердім|келісіміммен|"
    r"mening\s+ruxsatim\s+bilan|o'zim\s+ruxsat\s+berdim|roziligim\s+bilan|"
    r"with\s+my\s+(?:permission|consent)|i\s+(?:gave|granted)\s+(?:them|him|her)\s+permission"
)

# Payment cues require money/currency + an actual debit/send action + a missing
# credit outcome. Broad words like 'currency' or 'balance' alone never trigger.
_PAYMENT = [
    (r"деньг|средств|рубл|₽|баланс|валют|карт\w*|банк|бонус|покупк",
     r"списал\w*|снял\w*|перев[её]л\w*|вн[её]с\w*|заплатил\w*|отправил\w*|оплатил\w*|"
     r"купил\w*|куплен\w*|(?:совершил|сделал)\w*\s+покупк|покупк\w*.{0,25}(?:успешн\w*|заверш\w*|прошл\w*|совершил\w*)|успешн\w*.{0,20}покупк",
     r"не\s+(?:приш\w*|зачисл\w*|появ\w*|измени\w*|увелич\w*|получил\w*)|"
     r"нет.{0,15}(?:валют|зачислен|пополнен)|(?:валют|зачислен|пополнен)\w*.{0,10}нет|баланс.{0,15}(?:прежн|нол|нул)"),
    (r"ақша|қаражат|теңге|валюта|баланс|карта|төлем|бонус|сатып\s+алу",
     r"ұста(?:ды|лды)|шегер(?:ді|ілді|ген)|алынды|төледім|аудардым|жібердім|шықты|"
     r"сатып\s+(?:алдым|алған|алынған)|сатып\s+алу.{0,20}(?:сәтті\s+өтті|аяқталды|өтті)",
     r"түспеді|түспеген|келмеді|қосылмады|өзгермеді|өзгерген\s+жоқ|өспеді|көрінбейді|алмадым|сол\s+күйі|жоқ"),
    (r"\bpul\b|mablag'|valyuta|balans|karta|\bhaq\b|tanga|hisob|bonus|xarid",
     r"yech(?:di|ildi|ilgan)|ushla(?:di|ndi)|jo'natdim|yubordim|to'ladim|o'tkazdim|ketdi|olindi|"
     r"sotib\s+(?:oldim|olgan|olingan)|xarid.{0,25}(?:muvaffaqiyatli|tugadi)",
     r"kelmadi|tushmadi|tushmagan|qo'shilmadi|o'zgarmadi|oshmadi|ko'rinmayapti|ko'rinmadi|olmadim|yo'q|hali.{0,10}nol"),
    (r"\b(?:card|bank|money|coins|currency|balance|purchase|payment|transaction|charge|paid|bonuses|order)\b",
     r"\b(?:charged|debited|paid|sent|deducted|taken|bought|purchased)\b|left\s+my|"
     r"(?:purchase|payment|transaction|order).{0,25}(?:finished|completed|successful|processed)",
     r"\b(?:not|never)\b.{0,25}(?:credited|arrived|received|updated|appear|increased|changed|delivered)|"
     r"unchanged|still\s+zero|missing|no\s+coins|didn't\s+(?:get|receive)"),
]

_ACCOUNT = [tuple(re.compile(p) for p in row) for row in _ACCOUNT]
_LOCKOUT = [tuple(re.compile(p) for p in row) for row in _LOCKOUT]
_CHANGED_ACCESS = [tuple(re.compile(p) for p in row) for row in _CHANGED_ACCESS]
_PAYMENT = [tuple(re.compile(p) for p in row) for row in _PAYMENT]


def _action_present(text: str, pattern: re.Pattern) -> bool:
    for match in pattern.finditer(text):
        before = text[max(0, match.start()-40):match.start()]
        after = text[match.end():match.end()+20]
        if not _NEGATED_BEFORE.search(before) and not _NEGATED_AFTER.search(after):
            return True
    return False


def critical_intents(text: str) -> set[str]:
    """Return reported high-urgency `account`/`payment` cues only.

    Input should already have the project's scoped legacy-KK restoration. This
    detector does not rewrite arbitrary cyrillic characters or decode bytes.
    Surface rules cannot cover every paraphrase, quotation or negation scope.
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    # Mask code before turning the backtick apostrophe into a straight quote.
    text = _SOURCE_QUOTE.sub(" ", text).translate(_APOSTROPHES)
    found = set()
    clauses = [" ".join(c.split()) for c in re.split(r"[.!?;\n]+", text)]
    for index, clause in enumerate(clauses):
        if not clause or _HYPOTHETICAL.search(clause):
            continue
        # Associate cues in bounded windows; do not silently skip an entire long
        # message. Overlap keeps ordinary short phrases intact at boundaries.
        for start in range(0, len(clause), 400):
            window = clause[start:start+600]
            if not _ALLOWED_ACCESS.search(window) and any(actor.search(window) and _action_present(window, asset) and _action_present(window, action)
                   for actor, asset, action in _ACCOUNT):
                found.add("account")
            if any(_action_present(window, asset) and _action_present(window, action)
                   for asset, action in _LOCKOUT):
                found.add("account")
            if any(credentials.search(window) and _action_present(window, action) and unauthorized.search(window)
                   for credentials, action, unauthorized in _CHANGED_ACCESS):
                found.add("account")
        # A payment/change statement and its outcome often occupy two adjacent
        # sentences. Preserve the period so a denial cannot negate a later verb.
        adjacent = clauses[index+1] if index+1 < len(clauses) else ""
        combined = clause + (". " + adjacent if adjacent and not _HYPOTHETICAL.search(adjacent) else "")
        for start in range(0, len(combined), 400):
            window = combined[start:start+600]
            if any(money.search(window) and _action_present(window, action) and missing.search(window)
                   for money, action, missing in _PAYMENT):
                found.add("payment")
            if any(credentials.search(window) and _action_present(window, action) and unauthorized.search(window)
                   for credentials, action, unauthorized in _CHANGED_ACCESS):
                found.add("account")
    return found
