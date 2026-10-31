"""Weak supervision (training only): derive taxonomy category/priority from admin outcome (train only).

Admin replies, verdict, violation and punishment are NEVER used at inference.
They only produce training targets that a text-only classifier learns to predict.
"""
import re

def norm(t): return " ".join((t or "").casefold().replace("ё", "е").split())

# rule code (section, point) -> category
SECTION_ALIASES = [  # order matters
    (r"\bgz\b|зел[её]н\w* зон|гз\b", "gz"),
    (r"\bжб\b|правил\w* жалоб", "jb"),
    (r"\bоп\b|\bop\b|общ\w* правил|оп\b", "op"),
]
CODE_RE = re.compile(r"(?<![\d.])(\d\.\d{1,2}(?:\.\d{1,2})?)\s*(оп|op|gz|гз|жб)?", re.I)
ABBR = {  # RP abbreviations used by admins
    "dm": "dm", "дм": "dm", "db": "dm", "дб": "dm", "sk": "dm", "ск": "dm", "tk": "dm", "тк": "dm",
    "pg": "dm", "пг": "dm", "mg": "other", "мг": "other", "nonrp": "dm", "нонрп": "dm", "dmg": "dm",
}

def code_to_cat(code, sec):
    if sec in ("gz", "гз"): return "dm"
    if sec == "жб" or code.count(".") == 2 and code.startswith("1."): return "invalid"
    top = code.split(".")[:2]; p = ".".join(top)
    if p in ("2.1", "2.4", "2.6", "3.2"): return "insult"
    if p == "5.2": return "cheat"
    if p in ("5.3", "5.4", "5.1"): return "dm"
    if p in ("4.1", "4.2"): return "scam"
    if p in ("2.3", "4.4"): return "rmt_ads"
    if p in ("3.4",): return "admin"
    return "other"

PLAYER_KW = [
    ("cheat", r"\bчит|soft\b|\bchit|аим|aim|wh\b|вх\b|макрос|автоклик|флай|fly|летает|телепорт\w* по карте|сквозь стен|через стен|5\.2|chit|cheat|софт"),
    ("rmt_ads", r"за реал|реальн\w* деньг|прода[её]т акк|рекла|rmt|real pul|тг канал|telegram|телеграм|продаж\w* вирт"),
    ("scam", r"обман|aldab|aldad|aldov|алда|алаяқ|кинул|скам|scam|мошен|развел|развод|не отдал|не вернул|вымога|aldad|firib|4\.1"),
    ("insult", r"оск|атскарб|обзыва|матер|so.?kin|so.?kya|so.?kdi|sukin|onasi|онасин|haqor|балағат|боқ|оскорб|мат\b|матер|обзыв|родн|мамк|мать|токсич|расизм|haqorat|so'k|сук|пидор|пидар|хуй|еба|бля|2\.1|2\.4"),
    ("dm", r"\bдм\b|\bdm\b|\bdb\b|nrp|\b(?:u|o.?)ld[iy]?r(?:di|ib|gan|yapti|ishdi|adi)\w*|уход\w* от рп|ухо?дит от рп|уход с рп|leaving rp|задерж|повязал|кишан|kishan|верев|веревк|арканом|похищ|вбил|убив|бензопил|non ?rp|нрп|o.?ldir|урди|без причин|просто так|\bdm\b|\bдб\b|\bdb\b|\bск\b|\bsk\b|\bтк\b|\btk\b|убил|убива|без причин|gz\b|зел[её]н|нонрп|non ?rp|урон|o'ldir|сбил|давит|таран"),
]

ADMIN_APPEAL = re.compile(r"наказани\w* (?:было )?выдан\w* (?:верно|неверно|правильно|ошибочно)|будет снят|снято|снимем|"
                          r"с администратором (?:будет )?проведен|администратор\w* (?:получит|будет)|перевыдан|"
                          r"обжалован|амнист|наказание остается|оставлено в силе", re.I)
STRONG_APPEAL = re.compile(r"не соглас|\bопру|опроверж|(?:menga|meni|мне|меня)\b.{0,25}(?:mute|мут|ban\b|бан\b|jail|джайл|qamad)|(?:mute|ban|jail) (?:berdi|beradi|berildi)|сним\w* (?:с меня )?(?:бан|мут|наказ|варн|джайл|блок)|разбан|разблок|пересмотр|обжал|незаслуж|"
    r"(?:мне|меня) (?:дали|выдали|забанил\w*|замутил\w*|посадил\w*)|(?:дали|выдали|выдал) (?:мне )?(?:бан|мут|варн|джайл|предупр|деморган)|"
    r"за что (?:бан|мут|варн|меня)|снизить наказ|перевыдал|ban (?:yech|ochi|olib)|bandan|nohaq|noto.g.ri (?:ban|qaror|jazo)|"
    r"meni (?:ban|qamad)|menga (?:ban|nohaq)|бан(?:ды)? (?:алып|ал)|не үшін бан|бан бер|наказание выдано|unban|wrongly banned", re.I)
PLAYER_ADMIN = re.compile(r"деморган|\bзгс|\bгс\b|ответил\w* на (?:репорт|жб)|\bжб ответ|блат|blokladi|\bадм|adm\b|admin|админ|модер|сним\w* (?:бан|мут|наказ|варн|джайл)|разбан|пересмотр|обжал|"
                          r"выдал\w* (?:мне )?(?:бан|мут|варн|джайл|деморган)|незаслуж|не согласен|ошибочн|"
                          r"дали (?:бан|мут|варн|джайл)|получил (?:бан|мут|варн|джайл)|unban|ban(?:ladi|ni)", re.I)


def admin_codes(text):
    t = norm(text)
    out = []
    for m in CODE_RE.finditer(t):
        code, sec = m.group(1), (m.group(2) or "").lower()
        sec = {"op": "оп", "gz": "gz", "гз": "gz"}.get(sec, sec)
        if not sec and re.search(r"жалоб", t[m.end():m.end()+30]): sec = "жб"
        out.append((code, sec))
    for w in re.findall(r"\b(dm|дм|db|дб|sk|ск|tk|тк|pg|пг|nonrp|нонрп)\b", t):
        out.append((w, "abbr"))
    if re.search(r"зел[её]н\w* зон|\bgz\b", t):
        out.append(("gz", "abbr"))
    return out


def player_topic(text):
    t = norm(text)
    for cat, pat in PLAYER_KW:
        if re.search(pat, t): return cat
    return None


def label_complaint(r):
    player = "\n".join(m["text"] for m in r["messages"] if m["role"] == "player")
    admin = "\n".join(m["text"] for m in r["messages"] if m["role"] == "admin")
    verdict = r.get("verdict") or "other"
    src = []
    # 1. appeals / complaints on administrators
    if STRONG_APPEAL.search(norm(player)):
        return "complaint.admin", "player_appeal"
    if verdict in ("appeal_approved", "appeal_rejected") or ADMIN_APPEAL.search(admin) or (
            PLAYER_ADMIN.search(norm(player + " " + (r.get("rule_refs_raw") or ""))) and verdict not in ("approved",)):
        if verdict in ("appeal_approved", "appeal_rejected") or ADMIN_APPEAL.search(admin):
            return "complaint.admin", "admin_appeal_outcome"
    if verdict == "invalid_format":
        return "complaint.invalid", "admin_invalid_format"
    # 2. rule code the admin applied
    codes = admin_codes((r.get("violation") or "") + "\n" + (r.get("punishment") or "") + "\n" + admin)
    for code, sec in codes:
        if sec == "abbr":
            cat = "dm" if code == "gz" else ABBR.get(code)
        else:
            cat = code_to_cat(code, sec if sec else "оп")
        if cat:
            return "complaint." + cat, "admin_rule_code"
    # 3. invalid / no evidence
    ptopic = player_topic(player + " " + (r.get("rule_refs_raw") or ""))
    if verdict == "invalid_format" or (verdict == "need_evidence" and not r.get("images_count")):
        return "complaint.invalid", "admin_invalid_or_no_evidence"
    if len(norm(player)) < 12 and not r.get("images_count") and not ptopic:
        return "complaint.invalid", "empty_text"
    # 4. player-declared topic (rule_refs_raw + text) — weaker
    if re.search(r"деморган|\bзгс|\bгс\b|ответил\w* на (?:репорт|жб)|\bжб ответ|блат|blokladi|админ\w* (?:выдал|дал|не)|адм\w* (?:выдал|дал)", norm(player)):
        return "complaint.admin", "player_admin_keywords"
    if ptopic:
        return "complaint." + ptopic, "player_keywords"
    if PLAYER_ADMIN.search(norm(player)):
        return "complaint.admin", "player_admin_keywords"
    return "complaint.other", "fallback"


REPORT_KW = [
    ("report.payment", r"донат|задонат|оплат|плат[её]ж|пополн|не приш\w* (?:гк|бонус|донат|деньг)|списал|to'lov|төлем|payment|чек|"
                       r"\bгк\b.{0,20}не (?:приш|пришл|зачисл)|kaspi|каспи|qiwi|карт\w*.{0,15}спис"),
    ("report.account", r"взлом|взломал|пропал\w* (?:машин|авто|дом|имуществ|деньг|бизнес|вещ|скин|акс|все)|украл|"
                       r"акк\w*.{0,20}(?:взлом|увел|забрал|доступ)|парол|не могу (?:войти|зайти)|hisob|akkaunt|аккаунт\w* (?:заш|вош)"),
    ("report.complaint", r"\bоск|оскорб|\bдм\w*|\bdm\b|5\.2|софт|убил|убива|чит|cheat|аим|обман|кинул|скам|мат|сук|пидор|пидар|хуй|еба|"
                         r"бля|гнид|мешает|дмит|дмят|дмшит|токсич|флуд|спам|сбил|давит|haqorat|so'k|o'ldir"),
    ("report.help", r"застр|текстур|баг|bug|глюк|провалил|под карт|вытащ|выташ|не могу выйти|телепорт|тп\b|tp\b|"
                    r"заспавн|спавн|зажал|не (?:работает|грузит|загруж)|вылет|лага|stuck|yordam|qotib|көмек"),
    ("report.question", r"за что|\bкак\b|\bгде\b|\bпочему\b|\bзачем\b|\bсколько\b|\bкогда\b|\bможно\b|\bкакой\b|\bкакая\b|\bкто\b|"
                        r"\bчто (?:делать|будет|такое)|\?|qanday|qayer|qachon|nima|қалай|қайда|how|where|what|why"),
]
HELP_FIRST = r"перевер\w*|перевeр|достан\w*|дастан\w*|вытащ|выташ|застр|текстур|под карт|провалил|не могу выйти|заправ\w*|тачк\w*.{0,15}(?:перев|дост|застр)|stuck|qotib"
EMPTY = re.compile(r"^(?:[+.!?,\s\d]*|сос|sos|памаг\w*(?: пж\w*)?|поможіт\w*|көмектес\w*|yordam(?: bering)?|помогите?|помоги|хелп\w*|help|ау+|а+|привет\w*|здравствуйте|алло|админ\w*|ку+|салам\w*|прив|"
                   r"hello|hi|hey|ало|эй|слыш\w*|ало+|yordam|помогите пожалуйста|помогите плиз|help me|хелп ми плис|пж|пожалуйста)$")
ADMIN_HELP = re.compile(r"(?:сейчас|спешу|попробую|иду|лечу).{0,25}(?:помо|помощ)|телепортир|вытащ|освобод", re.I)
ADMIN_WATCH = re.compile(r"слежу|наблюда|приму меры|разберусь|накажу|проверю игрока|игрок наказан|получит", re.I)
ADMIN_CLARIFY = re.compile(r"уточните|опишите|конкретн|не понял", re.I)


PAYMENT_TEXT = re.compile(r"(?:купил\w*|купив|оплатил\w*|оплатив|задонатил\w*|задонатив|поповнив|пополнил\w*|сатып алдым|to.?ladim|sotib oldim)"
                          r".{0,60}(?:не\s*(?:приш|пришл|появ|зачисл|получ)|не\s*з.?явил|немає|нема|не прийш|түспеді|келмеді|kelmadi|tushmadi)")


def label_report(r):
    player = norm("\n".join(m["text"] for m in r["messages"] if m["role"] == "player"))
    if PAYMENT_TEXT.search(player):
        return "report.payment", "player_payment_phrase"
    if EMPTY.match(re.sub(r"[^\w\s+?!.]", "", player)):
        return "report.offtopic", "empty_text"  # taxonomy: a bare "помогите" is off-topic
    admin = "\n".join(m["text"] for m in r["messages"] if m["role"] == "admin")
    for cat, pat in REPORT_KW[:2]:
        if re.search(pat, player): return cat, "player_keywords"
    if re.search(HELP_FIRST, player):
        return "report.help", "player_keywords"
    if re.search(REPORT_KW[2][1], player) or (ADMIN_WATCH.search(admin) and not EMPTY.match(player)):
        return "report.complaint", "player_or_admin_watch"
    if re.search(REPORT_KW[3][1], player):
        return "report.help", "player_keywords"
    if EMPTY.match(re.sub(r"[^\w\s+?!.]", "", player)) or len(player) < 3:
        return "report.offtopic", "empty_text"
    if re.search(REPORT_KW[4][1], player):
        return "report.question", "player_keywords"
    if ADMIN_HELP.search(admin):
        return "report.help", "admin_help_reply"
    return "report.offtopic", "fallback"


def weak_category(r):
    return label_complaint(r) if r["kind"] == "complaint" else label_report(r)


THREAT_OUTSIDE = re.compile(r"(?:угроз\w*.{0,15}(?:жизн|в реал|адрес|убийств)|угрожа\w*.{0,25}(?:убить|найти|жизн|в реал|адрес)|суицид|покончу с собой|"
                            r"threat\w*.{0,15}(?:life|kill|irl)|amenaz\w*|amea[cç]\w*|өлтіремін|o.?ldiraman|найду|вычислю|приеду|знаю где ты живешь|адрес|сват|по ip|ддос|ddos|доксинг|деанон|убью тебя в реал|в реале)", re.I)
MASS_BUG = re.compile(r"массов\w* (?:баг|вылет|лаг|краш)|у всех (?:не |пропал|лаг|вылет)|весь сервер (?:лаг|упал|вылет)|сервер упал|лежит сервер|никто не может (?:зайти|войти)", re.I)


def weak_priority(r, category):
    """Taxonomy priority rules applied to category + reported evidence (no admin data)."""
    player = norm("\n".join(m["text"] for m in r["messages"] if m["role"] == "player"))
    if category in ("report.account", "report.payment") or THREAT_OUTSIDE.search(player) or MASS_BUG.search(player):
        return "p0"
    evidence = (r.get("images_count") or 0) > 0 or "<url>" in player
    if category in ("complaint.cheat", "complaint.scam", "complaint.rmt_ads"):
        return "p1"
    if category in ("complaint.invalid", "report.offtopic", "report.question"):
        return "p3"
    # every launcher complaint carries >=1 screenshot, so evidence does not separate p1/p2
    if category in ("complaint.insult", "complaint.dm", "complaint.admin", "complaint.other"):
        return "p2"
    if category in ("report.complaint", "report.help"):
        return "p2"
    return "p3"
