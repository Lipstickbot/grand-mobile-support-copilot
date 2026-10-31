"""Language detection for short, noisy, mixed game messages in the 6 game languages.

Reply languages: ru, kk, uz, en, es, pt (the game client's localisations).
Ukrainian/Belarusian text (common on CIS servers) is answered in Russian.
The upstream ``lang`` field is a dataset heuristic and is only trusted after a sanity
check: it labels Ukrainian as ``kk`` and Uzbek/Russian Latin text as ``pt``.
"""
from __future__ import annotations
import re

SUPPORTED = ("ru", "kk", "uz", "en", "es", "pt")
_KK_ONLY = re.compile(r"[әғқңөұүһ]", re.I)          # letters absent from Russian AND Ukrainian
_KK_WORDS = re.compile(r"\b(?:мен|мені|маған|менің|сен|сені|оны|біз|сіз|емес|деп|үшін|және|бірақ|себепсіз|тастады|атып|жатырмын|жүр|істемедім|сіз|не үшін|жоқ|керек|қалай|бар|бұл|бул|осындай|деген|берді|кірді|"
                       r"ойын\w*|ойыншы\w*|әкімші\w*|рахмет|сәлем\w*|көмек\w*)\b", re.I)
_UK = re.compile(r"[їєґ]|\b(?:що|ви|мене|мені|ти|він|вона|будь ласка|допоможіть|поможіть|немає|тож|ніхто|дякую)\b", re.I)
_CYR = re.compile(r"[а-яёіїєґәғқңөұүһ]", re.I)
_LAT = re.compile(r"[a-zñáéíóúãõçâêô]", re.I)
_UZ = re.compile(r"\b(?:men|meni|menga|man|mani|siz|sizga|iltimos|nima|nega|qil\w*|bilan|uchun|yo.?q|bor\w*|kerak|"
                 r"o.?yin\w*|shikoyat\w*|qoida\w*|ban\w* (?:berdi|yech)|yordam\w*|salom|assalom\w*|rahmat|qanday|qayer\w*|"
                 r"hech|kim|emas|edi|ekan|bo.?l\w*|ber\w*di|ketdi|keldi|so.?k\w*|onasi\w*|haqorat\w*|o.?ldir\w*|nohaq|"
                 r"admin\w*ga|kun|butunlay\w*|ket\w*|vaqt\w*|kuni|soat|pul|mashina|moshin\w*|yashil|hudud|akkaunt\w*|kira|buzish\w*|qanday|qayerda|qachon|politsiya\w*|ishga)\b|o['ʻ’`]|g['ʻ’`]|\w+(?:yapman|yapti|yapsan|ishdi|madim|mayman|maydi|ganman|olmay\w*)\b", re.I)
_EN = re.compile(r"\b(?:the|is|are|you|my|me|he|she|please|help|why|what|how|can|account|banned|player|was|for|and|this|killed|"
                 r"stuck|money|car|admin|report|hello|hi)\b", re.I)
_ES = re.compile(r"[ñ¿¡]|\b(?:el|los|las|que|qué|por|para|con|mi|yo|usted|está|esta|estoy|cuenta|jugador|ayuda|ayúdame|"
                 r"gracias|hola|baneado|insultó|insulto|mató|mato|dinero|coche|carro|pero|muy|también|porque|dónde|donde|"
                 r"cómo|tengo|puedo|hay|pagué|compré|sin|quiero|razón|razon|banearon|ahora|nada|todo|cuando|sí|señor|amigo|ayudar|atascado|la|del|una|un|mí|policía|juego|hago|hacer|todavía)\b|\w+ción\b", re.I)
_PT = re.compile(r"[ãõç]|\b(?:não|nao|você|voce|vc|meu|minha|conta|jogador|ajuda|ajudem|obrigad\w*|olá|ola|banido|xingou|"
                 r"matou|ele|ela|está|tá|ta|eu|isso|aqui|dinheiro|carro|mas|muito|também|porque|onde|como|tenho|posso|tem|"
                 r"paguei|comprei|pra|bom|dia|sem|quero|razão|baniram|agora|nada|tudo|quando|sim|preso|ajudar|travado|obg|faço|fazer|na|no|nos|um|uma|minha|meu|seu|sua|polícia|jogo|ainda)\b|\w+ção\b|\w+ões\b", re.I)
_NOISE = re.compile(r"<(?:url|digits|email|handle|nick)>|\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b|"
                    r"\b(?:op|gz|dm|db|sk|tk|rp|nrp|osk|fraps|soft|warn|jail|dmg|id|mute|ban)\b|https?://\S+", re.I)


def _cyrillic(t: str) -> tuple[str, str]:
    kk_words, uk_words = len(_KK_WORDS.findall(t)), len(_UK.findall(t))
    if _KK_ONLY.search(t) or (kk_words >= 2 and kk_words > uk_words):
        return "kk", "lang=kk_letters_or_words"
    if _UK.search(t):
        return "ru", "lang=ukrainian_answered_in_ru"
    return "ru", "lang=cyrillic"


def _latin(t: str, server_lang: str | None) -> tuple[str, str]:
    scores = {"uz": len(_UZ.findall(t)), "en": len(_EN.findall(t)), "es": len(_ES.findall(t)), "pt": len(_PT.findall(t))}
    if server_lang == "uz":
        scores["uz"] += .5
    best = max(scores, key=scores.get)
    if scores[best] >= 1:
        return best, f"lang=latin_cues_{best}"
    if server_lang in ("uz", "kk", "es", "pt", "en"):
        return server_lang, "lang=latin_server_fallback"
    return "ru", "lang=latin_translit_ru_fallback"


def detect(text: str, hinted: str | None = None, server_lang: str | None = None) -> tuple[str, list[str]]:
    """Return (reply_language, reasons). Never returns 'und': falls back to server language."""
    t = _NOISE.sub(" ", text or "")
    cyr, lat = len(_CYR.findall(t)), len(_LAT.findall(t))
    if cyr == 0 and lat == 0:
        if hinted in SUPPORTED:
            return hinted, ["lang=input_field"]
        fallback = server_lang if server_lang in SUPPORTED else "ru"
        return fallback, ["lang=no_letters_server_fallback"]
    lang, why = _cyrillic(t) if cyr >= lat else _latin(t, server_lang)
    if hinted in SUPPORTED and hinted != lang:
        # trust the upstream field only where the script agrees and no stronger signal contradicts it
        if hinted == "uz" and lang in ("en", "ru") and lat > cyr:
            return "uz", ["lang=input_field"]
        if hinted == "ru" and cyr >= lat and lang == "ru":
            return "ru", ["lang=input_field"]
        return lang, [why, f"lang=overrode_input_field_{hinted}"]
    return lang, [why]
