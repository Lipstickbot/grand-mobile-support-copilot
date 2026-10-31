# Сравнение классификаторов, 31 октября 2026

Все числа ниже — **известные авторские синтетические** примеры, не независимая
человеческая точность. Метки официального valid отсутствуют. Параметры методов
зафиксированы до fit; поиска гиперпараметров, early stopping по тесту и
fine-tuning энкодера не было. Ни один новый метод автоматически не заменяет сервис.

## Данные и модели

Новые методы обучены на одинаковых 164 AI-authored intent-примерах:84 прежних
авторских,64 вопросов/интейков из source-extractive эксперимента и 16 английских.
Только player text и известный kind. Нет admin/verdict/punishment/violation,
идентификаторов заявителя или ника как дополнительных признаков. Нормализованные
совпадения train/test запрещены. Близкие по смыслу шаблоны всё ещё возможны.

- Текущий: E5-small frozen, раздельные по kind ridge-головы категории и отдельная
  приоритетная голова, class balancing, regularization0,05.
- TF-IDF: word1–2 + char_wb3–5, vocabulary/IDF только train; LinearSVC C=1.
- E5 + Logistic Regression: C=1, class balancing, CPU, max_iter1000.
- CatBoost: E5 +12 text-only признаков длины/алфавита/пунктуации/регистра,
  depth4,200 iterations, lr0,05, balanced, CPU4 threads, seed42.

Прежний reference_v1 дополнительно видел 693 keyword weak labels; это справочный
baseline с другим обучением, не честный same-data конкурент новым методам.

## Чистые головы,42 примера

Все 14 категорий представлены по 3 раза, языки RU/KK/UZ — по 14. Ни language gate,
ни keyword fallback, ни safety priority floor здесь не применяются. Их влияние
измеряется отдельно настоящим HTTP.

| Метод | Category accuracy | Category macro-F1 | Priority macro-F1 |
|---|---:|---:|---:|
|Прежний reference ridge|24/42 (57,1%)|0,5499|0,6981|
|Текущий authored ridge|30/42 (71,4%)|0,6755|0,6883|
|TF-IDF + Linear SVM|15/42 (35,7%)|0,3095|0,4858|
|E5 + Logistic Regression|33/42 (78,6%)|0,7753|0,5692|
|E5 + признаки + CatBoost|29/42 (69,0%)|0,6595|0,5780|

## Разрезы category accuracy

| Метод | complaint, n24 | report, n18 | RU, n14 | KK, n14 | UZ, n14 |
|---|---:|---:|---:|---:|---:|
|reference ridge|45,8%|72,2%|64,3%|64,3%|42,9%|
|authored ridge|70,8%|72,2%|78,6%|71,4%|64,3%|
|TF-IDF/SVM|25,0%|50,0%|28,6%|28,6%|50,0%|
|E5/logistic|79,2%|77,8%|78,6%|78,6%|78,6%|
|CatBoost|66,7%|72,2%|64,3%|71,4%|71,4%|

В JSON имеются отдельные accuracy, macro-F1, per-class precision/recall и
confusion matrix для kind и каждого языка. В этом 42-наборе EN/ES/PT/lat/und
не представлены: для них метрики `null`, а не выдуманный ноль. Macro-F1 указан
по классам с поддержкой; дополнительно сохранён вариант по всем taxonomy-классам.

На 16 известных критичных текстах чистая authored ridge и CatBoost дали
p0recall16/16; logistic13/16, SVM12/16. Это маленькая тематическая диагностика,
не надёжность реального потока. После uncertainty/language/keyword маршрутизации
текущий сервис даёт 15/16p0 и 11/16 категория+p0; сравнивать эти числа с чистой
головой без пояснения нельзя.

Лидирующий category-кандидат — logistic, но его приоритет слабее. Продвигать его
всё целиком или выбирать порог по этим 42 нельзя. Пользователь выбрал меньше
пропусков p0: рабочим остаётся authored ridge + независимый safety floor.
Человеческая проверка должна решить, улучшает ли logistic категории реальных
коротких обращений. Отдельная проблема — fallback подавляет часть верных
предсказаний головы; это требует калибровки на **других** проверенных данных.

## Воспроизведение

```powershell
.venv-soup/Scripts/python.exe -m pip install --only-binary=:all: --target .private/eval-deps -r experiments/classifier-comparison/requirements-lock.txt
.venv-soup/Scripts/python.exe -X utf8 scripts/compare_classifiers.py --output .private/classifier-comparison-new
```

Рабочие доказательства: `.private/classifier-comparison-v6-audited/{plan,results}.json`.
Сохранены версии, хеши кода/таксономии/обучающих данных, отдельные головы logistic
в NPZ и собственные CatBoost CBM. Публикации весов или данных нет. Замеры времени
головы исключают построение признаков/HTTP/генерацию и не являются end-to-end latency.

После независимой ручной разметки возможен тот же fit и сравнение всех методов:

```powershell
.venv-soup/Scripts/python.exe -X utf8 scripts/compare_classifiers.py --output .private/classifier-human-new --review-packet .private/audit-v6/review-packet.jsonl
```

Сейчас команда обязана вернуть exit2 и `human_metrics:null`, поскольку пакет
не заполнен. [Протокол двух проверяющих](../../docs/REVIEW_PROTOCOL.md).
Transformer fine-tuning не запускался:164 непроверенных метки не оправдывают
увеличение ёмкости и расход GPU. Больше слабых меток не означает больше истины.

Методы проверены по первичным источникам:
[LinearSVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html),
[LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[CatBoost installation](https://catboost.ai/docs/en/concepts/python-installation),
[E5 model card](https://huggingface.co/intfloat/multilingual-e5-small).
