"""Authored positive and counterexample behavior checks; no model or network."""
import pytest

from support_copilot.intents import critical_intents


@pytest.mark.parametrize("text", [
    "Чужой человек получил доступ к моему персонажу",
    "Бөгде адам менің кейіпкеріме қол жеткізіп алды",
    "Begona odam mening qahramonimga kirish huquqini oldi",
    "Someone else logged in to my character while I was offline",
    "Неизвестный вошёл в мой профиль вчера, я его не знаю",
    "Бейтаныс менің профиліме кіріп алды, көмек керек",
    "Kimdir profilimga kirib oldi, yordam bering",
    "A stranger took over my game profile yesterday",
    "Я потерял доступ к моему персонажу после смены почты",
    "Кейіпкеріме кіре алмай қалдым",
    "Qahramonimga kira olmayapman",
    "I am locked out of my character",
])
def test_reported_foreign_access_and_access_loss_are_critical(text):
    assert critical_intents(text) == {"account"}


@pytest.mark.parametrize("text", [
    "Деньги списались с карты, но игровая валюта не появилась",
    "Ақша картамнан алынды, бірақ валюта балансы өзгермеді",
    "Valyuta uchun haq yubordim, lekin hisobga qo‘shilmadi",
    "My card was charged for coins, but my balance is unchanged",
    "Перевёл средства на покупку валюты, зачисления нет",
    "Теңгені аудардым, бірақ ойынға ақша түспеді",
    "Kartadan pul yechildi, lekin tangalar kelmadi",
    "I paid for game currency and didn't receive any coins",
    "Pulni joʻnatdim, ammo balans oʻzgarmadi",
])
def test_debit_and_missing_credit_are_critical(text):
    assert critical_intents(text) == {"payment"}


@pytest.mark.parametrize("text", [
    "Никогда не сообщайте пароль от персонажа",
    "Never share your password, keep your profile safe",
    "Құпиясөзді ешкімге бермеңіз",
    "Parolingizni hech kimga bermang",
    "Чужой человек не получил доступ к моему персонажу",
    "Someone has not gained access to my character",
    "Бөгде адам менің кейіпкеріме кірген жоқ",
    "Begona odam qahramonimga kirgan emas",
    "My card was not charged, my balance is unchanged",
    "Деньги не списались, валюта не пришла",
    "I am not locked out of my character",
    "Я не потерял доступ к моему персонажу",
    "Someone logged in to their character, not my profile",
    "Someone logged in to a character",
    "Я перевёл деньги другу, он их получил",
    "Карточка с правилом про игровую валюту",
    "How does the game balance work?",
    "Что делать, если чужой человек получил доступ к моему персонажу?",
    "What if someone logged in to my character?",
    "Пример: «Чужой человек получил доступ к моему персонажу»",
    'Rule: "Someone took over my account"',
    'Which rule applies if someone took over my character?',
    '> My card was charged, but my balance is unchanged',
    "```Деньги списались, валюта не появилась```",
    "",
])
def test_denials_security_advice_and_examples_are_not_active_reports(text):
    assert critical_intents(text) == set()


def test_quoted_example_does_not_hide_separate_actual_report():
    text = 'Пример: «Кто-то вошёл в мой профиль». Но у меня деньги списались с карты, валюта не появилась.'
    assert critical_intents(text) == {"payment"}


def test_multiple_critical_reports_can_coexist():
    text = 'Someone took over my character. My card was charged but my coins never arrived.'
    assert critical_intents(text) == {"account", "payment"}


def test_long_message_does_not_hide_a_critical_statement():
    text = 'дополнительные сведения ' * 80 + 'Чужой человек получил доступ к моему персонажу'
    assert critical_intents(text) == {"account"}


@pytest.mark.parametrize("text", [
    'На моём персонаже играет неизвестный человек',
    'Менің кейіпкеріммен бөтен біреу ойнап жүр',
    'Kimdir personajimdan foydalanyapti',
    'Somebody else has been using my character all morning',
    'Почту сменили без моего разрешения, помогите разобраться',
    'Кіру деректерімді рұқсатымсыз ауыстырған',
    'Loginimni roziligimsiz o‘zgartirishgan',
    'My login credentials were reset without my consent',
    'Пароль поменяли. Я не разрешал этого делать',
    'My password was replaced. I did not authorize that change',
])
def test_reported_intrusion_and_unauthorized_identity_changes(text):
    assert critical_intents(text) == {'account'}


@pytest.mark.parametrize("text", [
    'Я совершил покупку валюты. Купленные бонусы не пришли',
    'Банк списал рубли за валюту, но пополнения нет',
    'Купленные бонусы не появились после покупки',
    'Банк ақшаны ұстады. Бонустар келмеді',
    'Сатып алу сәтті өтті, бірақ бонустар көрінбейді',
    'Bank mablag‘ni ushladi. Tangalar hisobga tushmadi',
    'Xarid tugadi, sotib olingan bonuslar ko‘rinmayapti',
    'My transaction was processed. The coins never arrived',
    'I bought game coins. My balance has not increased',
    'The bank debited my card, yet the bonuses are missing',
])
def test_delivery_failures_and_cross_sentence_outcomes(text):
    assert critical_intents(text) == {'payment'}


@pytest.mark.parametrize("text", [
    'Посторонний играет на моём персонаже с моего разрешения',
    'Менің кейіпкеріммен бөтен адам менің рұқсатыммен ойнап отыр',
    'Begona kishi personajim bilan mening ruxsatim bilan o‘ynayapti',
    'Someone is playing on my character with my permission',
    'Someone is not playing on my character',
    "Someone isn't using my character",
    'Мои данные входа не изменили без моего согласия',
    'Кіру деректерімді рұқсатымсыз ауыстырған жоқ',
    'Loginimni roziligimsiz o‘zgartirishmagan',
    'My password was not reset without my permission',
    'Я сам поменял почту и никому не давал пароль',
    'My sign-in details changed with my consent',
    'Банк деньги не списал, пополнения нет',
    'Банк ақшаны шегермеді, бонус көрінбейді',
    'Bank pulni yechmadi, tangalar kelmadi',
    'The bank did not debit my card, my balance has not increased',
    'Если данные входа изменили без моего согласия, какой пункт действует?',
    'Егер кіру деректерімді рұқсатымсыз ауыстырған болса, не істеу керек?',
    'Agar loginimni roziligimsiz o‘zgartirishgan bo‘lsa, nima qilish kerak?',
    'What if my password was reset without my consent?',
    'Пример: «Банк списал деньги, пополнения нет»',
    'Rule: "The purchase completed, but coins never arrived"',
    'Никогда не пересылайте данные входа посторонним',
    'Never share your sign-in details with anyone',
    'Contact support if your purchase completed but your bonuses never appeared',
    'Обратитесь в поддержку, если банк списал деньги, а пополнения нет',
    'Support instructions: my password was reset without my permission',
    'Нұсқаулық: сатып алу сәтті өтсе, бірақ бонус көрінбесе, хабарласыңыз',
])
def test_new_negations_permission_and_instruction_counterexamples(text):
    assert critical_intents(text) == set()


def test_denied_intrusion_does_not_swallow_later_actual_payment_report():
    text = 'Никто не входил в мой профиль. Банк списал деньги, но пополнения нет.'
    assert critical_intents(text) == {'payment'}


def test_hypothetical_does_not_swallow_later_actual_identity_change():
    text = 'What if someone uses a character? My email was changed without my permission.'
    assert critical_intents(text) == {'account'}
