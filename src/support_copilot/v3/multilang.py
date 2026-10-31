"""Keyword layer for languages that are rare in the training data (kk, en, es, pt, uk).

The learned model is trained on ~95% Russian/Uzbek text, so for these languages a
precise multilingual keyword match takes precedence over the model's guess.
"""
from __future__ import annotations
import re

COMPLAINT = [  # order = priority
    ("admin", r"unban|banned me|unfair ban|wrong(?:ly)? ban|desban\w*|me banearon|ban injust\w*|me baniram|ban injust\w*|"
              r"бан берді|бан алыңыз|бан алып|неге бан|не үшін бан|әкімші\w* (?:бан|мут|жаза)|"
              r"admin\w* (?:banned|muted|gave)|administrador\w* (?:me )?(?:ban|silenc|castig)|зняти бан|розбан"),
    ("rmt_ads", r"real money|sell(?:s|ing)? (?:an? )?account|dinero real|vende\w* (?:la |una )?cuenta|dinheiro real|vende\w* (?:a |uma )?conta|нақты ақша"),
    ("cheat", r"\bcheat\w*|aimbot|wallhack|\bhack(?:s|er|ing)?\b(?! my)|\bfly(?:ing)?\b|trampa\w*|tramposo|chet\w*|trapa[cç]\w*|"
              r"\bчит\w*|бөгде бағдарлам\w*|софт"),
    ("scam", r"\bscam\w*|stole|didn.?t pay|estaf\w*|rob[oó]\b|no (?:me )?pag[oó]|golpe\w*|roubou|n[aã]o pagou|engan\w*|"
             r"алдады|алдап|алаяқ\w*|ұрлады"),
    ("insult", r"insult\w*|swear\w*|curs(?:e|ed|es|ing)\b|my family|called me|\bmother\b|\bmom\b|\bmadre\b|\bputa\w*|groser\w*|ofend\w*|xing\w*|"
               r"\bm[aã]e\b|palavr[aã]o|қорла\w*|балағат\w*|боқта\w*|анам\w*|тіл тигіз\w*|сөкті|ображ\w*|матюк\w*"),
    ("dm", r"\bkill(?:ed|s|ing)?\b|\brdm\b|no reason|mat[oó]\b|matar\w*|sin (?:raz[oó]n|motivo)|matou|sem (?:motivo|raz[aã]o)|"
           r"өлтір\w*|себепсіз|атып тастады|вбив\w*|убив\w*"),
]
REPORT = [
    ("report.account", r"hacked|lost access|can.?t log ?in|hacke\w*|robaron mi cuenta|no puedo entrar|roubaram (?:a )?minha conta|"
                       r"n[aã]o consigo entrar|аккаунтым\w* (?:бұзды|ұрлады)|кіре алмаймын|зламали"),
    ("report.payment", r"payment|refund|\bpago\b|reembols\w*|pagamento|\bpix\b|төлем\w*|донат"),
    ("report.help", r"\bstuck\b|qotib|tiqilib|chiqa olmay\w*|\bbug\w*|texture|fell|atascad\w*|atrapad\w*|textura|preso|presa|travad\w*|"
                    r"тұрып қалдым|кептеліп|шыға алмай|баг|застряг|перевер\w*|допоможіть|поможіть|памагіт\w*"),
    ("report.complaint", r"insult\w*|kill\w*|cheat\w*|mat[oó]\b|matou|xing\w*|insult\w*|қорла\w*|өлтір\w*|\bчит\w*|\bдм\b|\bdm\b"),
    ("report.question", r"\bhow\b|\bwhere\b|\bwhat\b|\bwhy\b|c[oó]mo|d[oó]nde|qu[eé]\b|por qu[eé]|\bonde\b|o que|por que|"
                        r"қалай|қайда|неге|не істеу|як\b|де\b|чому|\?"),
    ("report.help", r"\bhelp\b|ayuda\w*|ay[uú]dame|ajud\w*|көмек\w*"),
]


def keyword_category(kind: str, text: str) -> str | None:
    t = " ".join((text or "").casefold().split())
    if kind == "complaint":
        for topic, pat in COMPLAINT:
            if re.search(pat, t):
                return "complaint." + topic
        return None
    for cat, pat in REPORT:
        if re.search(pat, t):
            return cat
    return None
