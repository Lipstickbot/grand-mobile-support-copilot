"""Reply templates for the 6 game languages (ru/kk/uz/en/es/pt) and rule-mention formatting.

Templates never promise unban, refunds, compensation or punishment of another
player: the decision is always attributed to the administrator. ``{rule}`` is a
nominative rule phrase produced by :func:`rule_phrase`.
"""
from __future__ import annotations

import re

SECTION = {
    "ru": {30: "общих правил", 31: "правил подачи жалоб", 32: "правил Зелёной зоны"},
    "kk": {30: "Жалпы ереженің", 31: "Шағым беру ережесінің", 32: "Жасыл аймақ ережесінің"},
    "uz": {30: "Umumiy qoidalarning", 31: "Shikoyat berish qoidalarining", 32: "Yashil hudud qoidalarining"},
    "en": {30: "General Rules", 31: "Complaint Rules", 32: "Green Zone Rules"},
    "es": {30: "de las Reglas generales", 31: "de las Reglas de quejas", 32: "de las Reglas de la Zona Verde"},
    "pt": {30: "das Regras gerais", 31: "das Regras de denúncias", 32: "das Regras da Zona Verde"},
}

# short topic of frequently cited points: (ru, kk, uz, en)
TOPIC = {
    "rules:30:2.1": ("оскорбление игрока и его родственников", "ойыншыны және оның туыстарын қорлау", "o‘yinchi va uning qarindoshlarini haqorat qilish", "insulting a player or their relatives"),
    "rules:30:2.2": ("пропаганда запрещённых тем", "тыйым салынған тақырыптарды насихаттау", "taqiqlangan mavzularni targ‘ib qilish", "promoting prohibited topics"),
    "rules:30:2.3": ("реклама", "жарнама", "reklama", "advertising"),
    "rules:30:2.4": ("расизм и оскорбление религии", "нәсілшілдік және дінді қорлау", "irqchilik va dinni haqorat qilish", "racism and insulting religion"),
    "rules:30:2.6": ("перенос конфликтов между IC и OOC, угрозы", "IC мен OOC арасындағы қақтығыс, қоқан-лоққы", "IC va OOC o‘rtasidagi nizo, tahdidlar", "IC/OOC conflicts and threats"),
    "rules:30:3.1": ("использование репорта не по назначению", "репортты мақсатсыз пайдалану", "reportdan noto‘g‘ri foydalanish", "misuse of reports"),
    "rules:30:3.2": ("хамство в отношении администрации", "әкімшілікке дөрекілік", "administratsiyaga qo‘pollik", "rudeness towards administrators"),
    "rules:30:3.4": ("жалобы на администрацию и оспаривание её действий", "әкімшілікке шағым және оның әрекеттеріне дау айту", "administratsiyaga shikoyat va uning harakatlariga e’tiroz", "complaints about administrators"),
    "rules:30:1.2": ("выдача себя за администрацию", "өзін әкімші ретінде көрсету", "o‘zini administrator qilib ko‘rsatish", "impersonating the administration"),
    "rules:30:1.4": ("сохранность аккаунта и имущества", "аккаунт пен мүліктің қауіпсіздігі", "akkaunt va mulk xavfsizligi", "account and property safety"),
    "rules:30:1.6": ("запрещённые никнеймы", "тыйым салынған никнеймдер", "taqiqlangan nikneymlar", "prohibited nicknames"),
    "rules:30:4.1": ("мошенничество", "алаяқтық", "firibgarlik", "scamming"),
    "rules:30:4.2": ("вымогательство", "қорқытып алу", "tovlamachilik", "extortion"),
    "rules:30:4.4": ("сделки за реальные деньги", "нақты ақшаға мәміле", "haqiqiy pulga savdo", "real-money trading"),
    "rules:30:5.1": ("злоупотребление RP-механиками", "RP-механикаларды теріс пайдалану", "RP-mexanikalarni suiiste’mol qilish", "abusing RP mechanics"),
    "rules:30:5.2": ("стороннее ПО (читы)", "бөгде бағдарламалар (читтер)", "begona dasturlar (chitlar)", "third-party software (cheats)"),
    "rules:30:5.3": ("уход из RP-ситуации", "RP-жағдайдан кету", "RP-vaziyatdan chiqib ketish", "leaving an RP situation"),
    "rules:30:5.4": ("убийство или урон у дома", "үй жанында өлтіру не зақым келтіру", "uy oldida o‘ldirish yoki zarar yetkazish", "killing or damage near a house"),
    "rules:30:5.6": ("унижающие отыгрыши", "ар-намысты қорлайтын рөлдік әрекеттер", "inson qadr-qimmatini kamsituvchi rol o‘yinlari", "degrading role-play"),
    "rules:32:1.1": ("убийство и урон в Зелёной зоне", "Жасыл аймақта өлтіру және зақым келтіру", "Yashil hududda o‘ldirish va zarar yetkazish", "killing and damage in the Green Zone"),
    "rules:32:1.2": ("криминальные действия в Зелёной зоне", "Жасыл аймақтағы қылмыстық әрекеттер", "Yashil hududdagi jinoiy harakatlar", "criminal actions in the Green Zone"),
    "rules:31:1.2.2": ("отсутствие доказательств", "дәлелдердің болмауы", "dalillarning yo‘qligi", "missing evidence"),
    "rules:31:1.2.3": ("нет точного описания сути", "мәннің нақты сипаттамасы жоқ", "mohiyatning aniq tavsifi yo‘q", "no clear description"),
    "rules:31:1.2.7": ("оскорбления в тексте жалобы", "шағым мәтініндегі қорлау", "shikoyat matnidagi haqorat", "insults in the complaint text"),
    "rules:31:1.2.9": ("дублирование жалобы", "шағымның қайталануы", "shikoyat takrori", "duplicate complaint"),
    "rules:31:1.1.2": ("раздел жалоб не для вопросов и технических проблем", "шағым бөлімі сұрақтар мен техникалық мәселелерге арналмаған", "shikoyatlar bo‘limi savollar va texnik muammolar uchun emas", "the complaints section is not for questions or technical issues"),
    "rules:31:1.3.1": ("обязательные требования к жалобе", "шағымға қойылатын міндетті талаптар", "shikoyatga majburiy talablar", "mandatory complaint requirements"),
}
# es / pt topics, same keys as TOPIC
TOPIC_ES_PT = {
    "rules:31:1.1.2": ("la sección de quejas no es para preguntas ni problemas técnicos", "a seção de denúncias não é para perguntas nem problemas técnicos"),
    "rules:30:2.1": ("insultar a un jugador o a sus familiares", "insultar um jogador ou seus familiares"),
    "rules:30:2.2": ("propaganda de temas prohibidos", "propaganda de temas proibidos"),
    "rules:30:2.3": ("publicidad", "publicidade"),
    "rules:30:2.4": ("racismo e insultos a la religión", "racismo e insultos à religião"),
    "rules:30:2.6": ("conflictos entre IC y OOC, amenazas", "conflitos entre IC e OOC, ameaças"),
    "rules:30:3.1": ("uso indebido del reporte", "uso indevido do report"),
    "rules:30:3.2": ("faltas de respeto a la administración", "desrespeito à administração"),
    "rules:30:3.4": ("quejas sobre la administración y sus acciones", "reclamações sobre a administração e suas ações"),
    "rules:30:1.2": ("hacerse pasar por la administración", "fingir ser da administração"),
    "rules:30:1.4": ("seguridad de la cuenta y los bienes", "segurança da conta e dos bens"),
    "rules:30:1.6": ("apodos prohibidos", "apelidos proibidos"),
    "rules:30:4.1": ("estafa", "golpe"),
    "rules:30:4.2": ("extorsión", "extorsão"),
    "rules:30:4.4": ("comercio con dinero real", "comércio com dinheiro real"),
    "rules:30:5.1": ("abuso de mecánicas de RP", "abuso de mecânicas de RP"),
    "rules:30:5.2": ("software de terceros (cheats)", "software de terceiros (cheats)"),
    "rules:30:5.3": ("abandonar una situación de RP", "abandonar uma situação de RP"),
    "rules:30:5.4": ("matar o dañar cerca de una casa", "matar ou causar dano perto de uma casa"),
    "rules:30:5.6": ("rol degradante", "roleplay degradante"),
    "rules:32:1.1": ("matar o dañar en la Zona Verde", "matar ou causar dano na Zona Verde"),
    "rules:32:1.2": ("acciones criminales en la Zona Verde", "ações criminosas na Zona Verde"),
    "rules:31:1.2.2": ("falta de pruebas", "falta de provas"),
    "rules:31:1.2.3": ("sin una descripción clara de lo ocurrido", "sem uma descrição clara do ocorrido"),
    "rules:31:1.2.7": ("insultos en el texto de la queja", "insultos no texto da denúncia"),
    "rules:31:1.2.9": ("queja duplicada", "denúncia duplicada"),
    "rules:31:1.3.1": ("requisitos obligatorios de la queja", "requisitos obrigatórios da denúncia"),
}
for _cid, (_es, _pt) in TOPIC_ES_PT.items():
    TOPIC[_cid] = TOPIC[_cid] + (_es, _pt)
_LI = {"ru": 0, "kk": 1, "uz": 2, "en": 3, "es": 4, "pt": 5}


def rule_phrase(doc, lang: str) -> str:
    """Nominative mention of a rule point, e.g. 'п. 2.1 общих правил (оскорбление ...)'."""
    sec = SECTION[lang].get(doc.section_id)
    topic = TOPIC.get(doc.canonical_id, (None,) * 6)[_LI[lang]]
    if lang == "ru":
        base = f"п. {doc.ref} {sec}" if sec else f"п. {doc.ref} раздела «{doc.section_title}»"
    elif lang == "kk":
        base = f"{sec} {doc.ref}-тармағы" if sec else f"«{doc.section_title}» бөлімінің {doc.ref}-тармағы"
    elif lang == "uz":
        base = f"{sec} {doc.ref}-bandi" if sec else f"«{doc.section_title}» bo‘limining {doc.ref}-bandi"
    elif lang in ("es", "pt"):
        word = "punto" if lang == "es" else "ponto"
        base = f"{word} {doc.ref} {sec}" if sec else f"{word} {doc.ref} «{doc.section_title}»"
    else:
        base = f"{sec} {doc.ref}" if sec else f"rule {doc.ref} of “{doc.section_title}”"
    return f"{base} ({topic})" if topic else base


T = {
    "ru": {
        "not_complaint": "Уважаемый игрок, благодарим за обращение! Обращение не может быть рассмотрено как жалоба: {rule}. С вопросом по игре напишите в репорт, с технической проблемой обратитесь в техподдержку.",
        "report.help_bare": "Приветствую! Слежу за вами. Опишите, пожалуйста, что случилось — постараюсь помочь.",
        "report.question_where": "Приветствую! Нужное место можно найти через навигатор в игре. Если не получится, уточните, что именно ищете.",
        "sanction": " Если нарушение подтвердится, по этому пункту правилами предусмотрено: {sanction}.",
        "complaint": "Уважаемый игрок, благодарим за обращение! Жалоба{on_nick} принята к рассмотрению. Подходящее правило: {rule}. Администратор проверит приложенные доказательства и сообщит решение.",
        "complaint_norule": "Уважаемый игрок, благодарим за обращение! Жалоба{on_nick} принята к рассмотрению. Администратор проверит приложенные доказательства и сообщит решение.",
        "need_evidence": " Чтобы жалобу можно было рассмотреть, приложите скриншот или видео ({rule2}).",
        "admin": "Здравствуйте! Обращение по поводу действий администратора или выданного наказания передано старшей администрации на проверку. Порядок: {rule}. Решение о снятии или изменении наказания принимает только администрация.",
        "invalid": "Уважаемый игрок, благодарим за обращение! Жалоба не может быть рассмотрена: {rule}. Подайте новую жалобу: укажите ник нарушителя, что именно произошло, и приложите доказательства.",
        "duplicate": "Уважаемый игрок, благодарим за обращение! Эта жалоба повторяет уже поданную ({rule}). Дождитесь ответа по первой жалобе.",
        "report.offtopic": "Приветствую! Уточните, пожалуйста, ваш вопрос или жалобу.",
        "report.help": "Приветствую! Сейчас постараюсь вам помочь. Опишите, пожалуйста, где именно вы находитесь и что случилось.",
        "report.question": "Приветствую! Ваш вопрос передан администратору, ожидайте ответа.",
        "report.complaint": "Приветствую! Слежу за ситуацией. Если нарушение продолжится, подайте жалобу через лаунчер со скриншотом или видео.",
        "report.complaint_rule": "Приветствую! Слежу за ситуацией. Подходящее правило: {rule}. Если нарушение продолжится, подайте жалобу через лаунчер со скриншотом или видео.",
        "report.account": "Приветствую! Обращение о доступе к аккаунту передано специалисту поддержки. Никому не сообщайте пароль и коды из SMS. Учтите: {rule}, за сохранность аккаунта отвечает сам игрок.",
        "report.payment": "Приветствую! Обращение о платеже передано специалисту поддержки. Не отправляйте в чат пароли, коды и полные данные карты.",
        "on_nick": " на игрока {nick}",
    },
    "kk": {
        "not_complaint": "Құрметті ойыншы, өтінішіңіз үшін рахмет! Өтініш шағым ретінде қаралмайды: {rule}. Ойын туралы сұрақты репортқа жазыңыз, техникалық мәселе бойынша техқолдауға жүгініңіз.",
        "report.help_bare": "Сәлеметсіз бе! Сізді бақылап отырмын. Не болғанын жазыңызшы — көмектесуге тырысамын.",
        "report.question_where": "Сәлеметсіз бе! Қажетті орынды ойындағы навигатор арқылы табуға болады. Шықпаса, нақты не іздеп жүргеніңізді жазыңыз.",
        "sanction": " Бұзушылық расталса, бұл тармақ бойынша ережеде көзделген: {sanction}.",
        "complaint": "Құрметті ойыншы, өтінішіңіз үшін рахмет! {nick_kk}шағым қарауға қабылданды. Қатысты ереже: {rule}. Әкімші қоса берілген дәлелдерді тексеріп, шешімін хабарлайды.",
        "complaint_norule": "Құрметті ойыншы, өтінішіңіз үшін рахмет! {nick_kk}шағым қарауға қабылданды. Әкімші қоса берілген дәлелдерді тексеріп, шешімін хабарлайды.",
        "need_evidence": " Шағымды қарау үшін скриншот немесе бейне қосыңыз ({rule2}).",
        "admin": "Сәлеметсіз бе! Әкімшінің әрекеттеріне немесе берілген жазаға қатысты өтінішіңіз аға әкімшілікке тексеруге жіберілді. Тәртіп: {rule}. Жазаны алып тастау немесе өзгерту туралы шешімді тек әкімшілік қабылдайды.",
        "invalid": "Құрметті ойыншы, өтінішіңіз үшін рахмет! Шағымды қарау мүмкін емес: {rule}. Жаңа шағым беріңіз: бұзушының никін, не болғанын көрсетіп, дәлелдерді қосыңыз.",
        "duplicate": "Құрметті ойыншы, өтінішіңіз үшін рахмет! Бұл шағым бұрын берілген шағымды қайталайды ({rule}). Алғашқы шағымға жауапты күтіңіз.",
        "report.offtopic": "Сәлеметсіз бе! Сұрағыңызды немесе шағымыңызды нақтылаңызшы.",
        "report.help": "Сәлеметсіз бе! Қазір көмектесуге тырысамын. Қай жерде тұрғаныңызды және не болғанын жазыңызшы.",
        "report.question": "Сәлеметсіз бе! Сұрағыңыз әкімшіге жіберілді, жауапты күтіңіз.",
        "report.complaint": "Сәлеметсіз бе! Жағдайды бақылап отырмын. Бұзушылық жалғасса, лаунчер арқылы скриншот немесе бейнемен шағым беріңіз.",
        "report.complaint_rule": "Сәлеметсіз бе! Жағдайды бақылап отырмын. Қатысты ереже: {rule}. Бұзушылық жалғасса, лаунчер арқылы скриншот немесе бейнемен шағым беріңіз.",
        "report.account": "Сәлеметсіз бе! Аккаунтқа кіру бойынша өтінішіңіз қолдау маманына жіберілді. Құпиясөзді және SMS-кодтарды ешкімге айтпаңыз. Ескеріңіз: {rule}, аккаунттың қауіпсіздігіне ойыншының өзі жауап береді.",
        "report.payment": "Сәлеметсіз бе! Төлем бойынша өтінішіңіз қолдау маманына жіберілді. Чатқа құпиясөзді, кодтарды және картаның толық деректерін жібермеңіз.",
        "on_nick": "",
    },
    "uz": {
        "not_complaint": "Hurmatli o‘yinchi, murojaatingiz uchun rahmat! Murojaat shikoyat sifatida ko‘rib chiqilmaydi: {rule}. O‘yin bo‘yicha savolni reportga yozing, texnik muammo bo‘yicha texnik yordamga murojaat qiling.",
        "report.help_bare": "Assalomu alaykum! Sizni kuzatyapman. Nima bo‘lganini yozing — yordam berishga harakat qilaman.",
        "report.question_where": "Assalomu alaykum! Kerakli joyni o‘yindagi navigator orqali topishingiz mumkin. Topa olmasangiz, aynan nimani qidirayotganingizni yozing.",
        "sanction": " Qoidabuzarlik tasdiqlansa, ushbu band bo‘yicha qoidalarda nazarda tutilgan: {sanction}.",
        "complaint": "Hurmatli o‘yinchi, murojaatingiz uchun rahmat! {nick_uz}shikoyatingiz ko‘rib chiqishga qabul qilindi. Tegishli qoida: {rule}. Administrator ilova qilingan dalillarni tekshirib, qaror haqida xabar beradi.",
        "complaint_norule": "Hurmatli o‘yinchi, murojaatingiz uchun rahmat! {nick_uz}shikoyatingiz ko‘rib chiqishga qabul qilindi. Administrator ilova qilingan dalillarni tekshirib, qaror haqida xabar beradi.",
        "need_evidence": " Shikoyat ko‘rib chiqilishi uchun skrinshot yoki video ilova qiling ({rule2}).",
        "admin": "Assalomu alaykum! Administrator harakatlari yoki berilgan jazo bo‘yicha murojaatingiz katta administratsiyaga tekshirish uchun yuborildi. Tartib: {rule}. Jazoni olib tashlash yoki o‘zgartirish haqidagi qarorni faqat administratsiya qabul qiladi.",
        "invalid": "Hurmatli o‘yinchi, murojaatingiz uchun rahmat! Shikoyatni ko‘rib chiqib bo‘lmaydi: {rule}. Yangi shikoyat yozing: qoidabuzarning nikini, nima bo‘lganini ko‘rsating va dalillarni ilova qiling.",
        "duplicate": "Hurmatli o‘yinchi, murojaatingiz uchun rahmat! Bu shikoyat avval yuborilgan shikoyatni takrorlaydi ({rule}). Birinchi shikoyatga javobni kuting.",
        "report.offtopic": "Assalomu alaykum! Iltimos, savolingiz yoki shikoyatingizni aniqlashtiring.",
        "report.help": "Assalomu alaykum! Hozir yordam berishga harakat qilaman. Qayerdaligingizni va nima bo‘lganini yozing.",
        "report.question": "Assalomu alaykum! Savolingiz administratorga yuborildi, javobni kuting.",
        "report.complaint": "Assalomu alaykum! Vaziyatni kuzatyapman. Qoidabuzarlik davom etsa, launcher orqali skrinshot yoki video bilan shikoyat yozing.",
        "report.complaint_rule": "Assalomu alaykum! Vaziyatni kuzatyapman. Tegishli qoida: {rule}. Qoidabuzarlik davom etsa, launcher orqali skrinshot yoki video bilan shikoyat yozing.",
        "report.account": "Assalomu alaykum! Akkauntga kirish bo‘yicha murojaatingiz yordam xizmati mutaxassisiga yuborildi. Parol va SMS-kodlarni hech kimga aytmang. E’tibor bering: {rule}, akkaunt xavfsizligi uchun o‘yinchining o‘zi javob beradi.",
        "report.payment": "Assalomu alaykum! To‘lov bo‘yicha murojaatingiz yordam xizmati mutaxassisiga yuborildi. Chatga parol, kodlar va kartaning to‘liq ma’lumotlarini yubormang.",
        "on_nick": "",
    },
    "en": {
        "not_complaint": "Dear player, thank you for reaching out! This cannot be reviewed as a complaint: {rule}. Ask game questions via the in-game report, and contact technical support for technical problems.",
        "report.help_bare": "Hello! I'm watching. Please describe what happened and I'll try to help.",
        "report.question_where": "Hello! You can find the place you need via the in-game navigator. If that doesn't work, tell us exactly what you're looking for.",
        "sanction": " If the violation is confirmed, this rule provides for: {sanction}.",
        "complaint": "Dear player, thank you for reaching out! Your complaint{on_nick} has been accepted for review. Relevant rule: {rule}. An administrator will check the attached evidence and report the decision.",
        "complaint_norule": "Dear player, thank you for reaching out! Your complaint{on_nick} has been accepted for review. An administrator will check the attached evidence and report the decision.",
        "need_evidence": " Please attach a screenshot or video so the complaint can be reviewed ({rule2}).",
        "admin": "Hello! Your request about an administrator's actions or a punishment has been passed to senior administration for review. Procedure: {rule}. Only the administration decides whether a punishment is lifted or changed.",
        "invalid": "Dear player, thank you for reaching out! This complaint cannot be reviewed: {rule}. Please submit a new complaint with the offender's nickname, what happened, and evidence.",
        "duplicate": "Dear player, thank you for reaching out! This complaint repeats one you already submitted ({rule}). Please wait for the answer to the first one.",
        "report.offtopic": "Hello! Please clarify your question or complaint.",
        "report.help": "Hello! I'll try to help you now. Please describe where you are and what happened.",
        "report.question": "Hello! Your question has been passed to an administrator, please wait for an answer.",
        "report.complaint": "Hello! I'm watching the situation. If the violation continues, submit a complaint via the launcher with a screenshot or video.",
        "report.complaint_rule": "Hello! I'm watching the situation. Relevant rule: {rule}. If the violation continues, submit a complaint via the launcher with a screenshot or video.",
        "report.account": "Hello! Your account access request has been passed to a support specialist. Never share your password or SMS codes. Note: {rule}, players are responsible for their account's safety.",
        "report.payment": "Hello! Your payment request has been passed to a support specialist. Do not send passwords, codes or full card details in chat.",
        "on_nick": " against {nick}",
    },
    "es": {
        "not_complaint": "Estimado jugador, ¡gracias por escribirnos! Esto no puede revisarse como una queja: {rule}. Haga sus preguntas sobre el juego mediante el reporte en el juego y contacte al soporte técnico para problemas técnicos.",
        "report.help_bare": "¡Hola! Le estoy observando. Describa qué ocurrió e intentaré ayudarle.",
        "report.question_where": "¡Hola! Puede encontrar el lugar que necesita con el navegador del juego. Si no lo logra, indique qué busca exactamente.",
        "complaint": "Estimado jugador, ¡gracias por escribirnos! Su queja{on_nick} fue aceptada para revisión. Regla aplicable: {rule}. Un administrador revisará las pruebas adjuntas y le comunicará la decisión.",
        "complaint_norule": "Estimado jugador, ¡gracias por escribirnos! Su queja{on_nick} fue aceptada para revisión. Un administrador revisará las pruebas adjuntas y le comunicará la decisión.",
        "need_evidence": " Para que la queja pueda revisarse, adjunte una captura de pantalla o un video ({rule2}).",
        "admin": "¡Hola! Su solicitud sobre las acciones de un administrador o una sanción fue enviada a la administración superior para su revisión. Procedimiento: {rule}. Solo la administración decide si una sanción se retira o se modifica.",
        "invalid": "Estimado jugador, ¡gracias por escribirnos! La queja no puede ser revisada: {rule}. Envíe una nueva queja indicando el apodo del infractor, qué ocurrió exactamente y adjuntando pruebas.",
        "duplicate": "Estimado jugador, ¡gracias por escribirnos! Esta queja repite una que ya envió ({rule}). Espere la respuesta a la primera queja.",
        "report.offtopic": "¡Hola! Por favor, precise su pregunta o queja.",
        "report.help": "¡Hola! Intentaré ayudarle ahora. Por favor, describa dónde se encuentra y qué ocurrió.",
        "report.question": "¡Hola! Su pregunta fue enviada a un administrador, espere la respuesta.",
        "report.complaint": "¡Hola! Estoy vigilando la situación. Si la infracción continúa, envíe una queja desde el launcher con una captura o un video.",
        "report.complaint_rule": "¡Hola! Estoy vigilando la situación. Regla aplicable: {rule}. Si la infracción continúa, envíe una queja desde el launcher con una captura o un video.",
        "report.account": "¡Hola! Su solicitud sobre el acceso a la cuenta fue enviada a un especialista de soporte. No comparta con nadie su contraseña ni los códigos SMS. Tenga en cuenta: {rule}, el jugador es responsable de la seguridad de su cuenta.",
        "report.payment": "¡Hola! Su solicitud sobre el pago fue enviada a un especialista de soporte. No envíe por chat contraseñas, códigos ni los datos completos de su tarjeta.",
        "sanction": " Si se confirma la infracción, este punto de las reglas prevé: {sanction}.",
        "on_nick": " contra {nick}",
    },
    "pt": {
        "not_complaint": "Prezado jogador, obrigado pelo contato! Isto não pode ser analisado como denúncia: {rule}. Envie perguntas sobre o jogo pelo report no jogo e procure o suporte técnico para problemas técnicos.",
        "report.help_bare": "Olá! Estou acompanhando você. Descreva o que aconteceu e vou tentar ajudar.",
        "report.question_where": "Olá! Você pode encontrar o local com o navegador do jogo. Se não conseguir, diga exatamente o que está procurando.",
        "complaint": "Prezado jogador, obrigado pelo contato! Sua denúncia{on_nick} foi aceita para análise. Regra aplicável: {rule}. Um administrador vai verificar as provas anexadas e informar a decisão.",
        "complaint_norule": "Prezado jogador, obrigado pelo contato! Sua denúncia{on_nick} foi aceita para análise. Um administrador vai verificar as provas anexadas e informar a decisão.",
        "need_evidence": " Para que a denúncia possa ser analisada, anexe uma captura de tela ou um vídeo ({rule2}).",
        "admin": "Olá! Sua solicitação sobre as ações de um administrador ou uma punição foi enviada à administração superior para análise. Procedimento: {rule}. Somente a administração decide se uma punição é retirada ou alterada.",
        "invalid": "Prezado jogador, obrigado pelo contato! A denúncia não pode ser analisada: {rule}. Envie uma nova denúncia com o apelido do infrator, o que aconteceu exatamente e as provas.",
        "duplicate": "Prezado jogador, obrigado pelo contato! Esta denúncia repete uma que você já enviou ({rule}). Aguarde a resposta à primeira denúncia.",
        "report.offtopic": "Olá! Por favor, especifique sua pergunta ou denúncia.",
        "report.help": "Olá! Vou tentar ajudar agora. Por favor, descreva onde você está e o que aconteceu.",
        "report.question": "Olá! Sua pergunta foi enviada a um administrador, aguarde a resposta.",
        "report.complaint": "Olá! Estou acompanhando a situação. Se a infração continuar, envie uma denúncia pelo launcher com uma captura de tela ou vídeo.",
        "report.complaint_rule": "Olá! Estou acompanhando a situação. Regra aplicável: {rule}. Se a infração continuar, envie uma denúncia pelo launcher com uma captura de tela ou vídeo.",
        "report.account": "Olá! Sua solicitação sobre o acesso à conta foi enviada a um especialista do suporte. Não compartilhe sua senha nem os códigos SMS com ninguém. Atenção: {rule}, o jogador é responsável pela segurança da própria conta.",
        "report.payment": "Olá! Sua solicitação sobre o pagamento foi enviada a um especialista do suporte. Não envie no chat senhas, códigos nem os dados completos do cartão.",
        "sanction": " Se a infração for confirmada, este ponto das regras prevê: {sanction}.",
        "on_nick": " contra {nick}",
    },
}


def render(key: str, lang: str, *, rule: str = "", rule2: str = "", nick: str | None = None, sanction: str = "") -> str:
    table = T[lang]
    on_nick = table["on_nick"].format(nick=nick) if nick and table["on_nick"] else ""
    nick_kk = f"{nick} ойыншысына қатысты " if nick else ""
    nick_uz = f"{nick} o‘yinchisiga nisbatan " if nick else ""
    text = table[key].format(rule=rule, rule2=rule2, sanction=sanction, on_nick=on_nick, nick=nick or "", nick_kk=nick_kk, nick_uz=nick_uz)
    if not nick:
        text = re.sub(r"! (шағым|shikoyatingiz)", lambda m: "! " + m.group(1)[0].upper() + m.group(1)[1:], text, count=1)
    return text
