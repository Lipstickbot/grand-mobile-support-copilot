"""Explicit multilingual intake templates, not generated support procedures.

Translations are provisional until reviewed by staff/native speakers. These
templates collect nonsecret context; they cannot restore access or refund money.
"""
from .policy import FALLBACKS


_TEXT = {
    "ru": {
        "payment": "Укажите ник, сервер, дату, сумму и идентификатор платежа. Не присылайте пароль, коды подтверждения или полные данные карты. Ситуацию должен проверить сотрудник поддержки.",
        "account": "Укажите ник, сервер, время и опишите, как изменился доступ к персонажу. Не присылайте пароль, коды подтверждения или данные для входа. Ситуацию должен проверить сотрудник поддержки.",
        "other": "Уточните ник, сервер, время и обстоятельства обращения. Ситуацию должен проверить сотрудник поддержки.",
        "attached": "Вложения отмечены, но их содержимое здесь недоступно.",
        "absent": "При наличии приложите запись или скриншот без паролей, кодов и полных платёжных данных.",
    },
    "kk": {
        "payment": "Ойыншы атын, серверді, төлем күнін, сомасын және төлем нөмірін жазыңыз. Құпиясөзді, растау кодтарын немесе картаның толық деректерін жібермеңіз. Жағдайды қолдау қызметкері тексеруі қажет.",
        "account": "Ойыншы атын, серверді, уақытты және кейіпкерге кіруде не өзгергенін жазыңыз. Құпиясөзді, растау кодтарын немесе кіру деректерін жібермеңіз. Жағдайды қолдау қызметкері тексеруі қажет.",
        "other": "Ойыншы атын, серверді, уақытты және оқиғаның мән-жайын нақтылаңыз. Жағдайды қолдау қызметкері тексеруі қажет.",
        "attached": "Тіркемелер бар деп көрсетілген, бірақ олардың мазмұны мұнда қолжетімсіз.",
        "absent": "Бар болса, құпиясөздер, кодтар және толық төлем деректері жоқ бейне не скриншот қосыңыз.",
    },
    "uz": {
        "payment": "O‘yinchi nomi, server, to‘lov sanasi, summasi va to‘lov raqamini yozing. Parol, tasdiqlash kodlari yoki kartaning to‘liq ma’lumotlarini yubormang. Holatni yordam xizmati xodimi tekshirishi kerak.",
        "account": "O‘yinchi nomi, server, vaqt va qahramonga kirishda nima o‘zgarganini yozing. Parol, tasdiqlash kodlari yoki kirish ma’lumotlarini yubormang. Holatni yordam xizmati xodimi tekshirishi kerak.",
        "other": "O‘yinchi nomi, server, vaqt va voqea tafsilotlarini aniqlashtiring. Holatni yordam xizmati xodimi tekshirishi kerak.",
        "attached": "Ilovalar borligi ko‘rsatilgan, ammo ularning mazmuni bu yerda mavjud emas.",
        "absent": "Bo‘lsa, parol, kod va to‘liq to‘lov ma’lumotlarisiz video yoki skrinshot qo‘shing.",
    },
    "en": {
        "payment": "Please specify the player name, server, payment date, amount and transaction ID. Do not send passwords, verification codes or full card details. A support specialist needs to review this case.",
        "account": "Please specify the player name, server, time and what changed about access to your character. Do not send passwords, verification codes or login credentials. A support specialist needs to review this case.",
        "other": "Please specify the player name, server, time and circumstances. A support specialist needs to review this case.",
        "attached": "Attachments are reported, but their contents are unavailable here.",
        "absent": "If available, attach a recording or screenshot without passwords, codes or full payment details.",
    },
}


def intake_reply(lang: str, topics=(), *, has_attachments=False) -> str:
    if lang not in _TEXT:
        return FALLBACKS.get(lang, FALLBACKS["ru"])
    topic = next((t for t in ("payment", "account") if t in topics), "other")
    return _TEXT[lang][topic] + " " + _TEXT[lang]["attached" if has_attachments else "absent"]
