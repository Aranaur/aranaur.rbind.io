# MATH840 2026/27 — правки до силабусу

Перенести у живий документ (`MATH840-TimeSeries26-27.pdf` / його джерело) до роздачі студентам.
Порядок — за пріоритетом: пункти 1–4 ламають арифметику документа, решта узгоджує механіку з 80-хвилинним слотом.

---

## 1. Контактні години — ПОМИЛКА В ДОКУМЕНТІ

Реальний розклад: лекція 80 хв + практика 80 хв, вівторок, одна за одною.
Лекційні години в силабусі пораховані правильно (9 × 2 акад. год = 18), практичні — **вдвічі завищені**.

| Місце | Було | Стало |
|---|---|---|
| Шапка, `Class hours` | `18 lecture hours and 44 practice hours (62 contact hours, 58 hours of self-study)` | `18 lecture hours and 22 practice hours (40 contact hours, 80 hours of self-study)` |
| Course Structure | `one lecture (2 hours) and one practice session (4 hours) per week` | `one lecture (80 min) and one practice session (80 min) per week, back to back` |
| Course Structure | `Week 10: two practice sessions (4 hours each)` | `two practice sessions (80 min each)` |
| Course Structure | `9 lectures (18 hours), 11 practice sessions (44 hours), 58 hours of self-study` | `9 lectures (18 hours), 11 practice sessions (22 hours), 80 hours of self-study` |

Перевірка 4 ECTS: 40 контактних + 80 самостійних = 120 годин. ✅

## 2. Таблиця самостійної роботи (58 → 80 годин)

| Activity | Hours |
|---|---:|
| Required pre-reading of the assigned chapter before each lecture (Weeks 1–9) | 35 |
| Final project — home improvement phase between Week 10 sessions | 20 |
| Final project — report and presentation slides | 15 |
| Mid-term quiz preparation | 10 |
| **Total** | **80** |

## 3. Сім лабораторних замість восьми

Практичних слотів на тижнях 1–9 рівно дев'ять. Тиждень 1 — онбординг без оцінювання, тиждень 5 — квіз.
Лишається сім: тижні **2, 3, 4, 6, 7, 8, 9**.

| Місце | Було | Стало |
|---|---|---|
| Grading table | `Weekly assignments \| 8 \| 7 \| 56` | `Weekly assignments \| 7 \| 8 \| 56` |
| Weekly in-class assignments | `There are eight weekly assignments, each worth up to 7 points (56 points total)` | `There are seven weekly assignments, each worth up to 8 points (56 points total)` |
| Track A | `Foundational labs (Weeks 1–3)` | `Foundational labs (Weeks 2–3)` |
| Track B | `Labs 4–8 are timed challenges` | `The five challenge labs (Weeks 4, 6, 7, 8, 9) are timed` |

Додати речення в Course Structure:

> **Week 1 is an onboarding session.** No coursework is assessed: the practice slot is used to set up
> environments, issue Track A datasets, and rehearse the submission mechanics end to end.

## 4. Рубрики перераховані на 8 балів

**Track A** (було 1/3/2/1 = 7):

| Criteria | Points |
|---|---:|
| Data preparation | 1 |
| EDA and visualisation | **4** |
| Implementation | 2 |
| Code quality and reproducibility | 1 |

**Track B** (було 3/2/1/1 = 7):

| Criteria | Points |
|---|---:|
| Forecast accuracy vs benchmark | 3 |
| Justification block | **3** |
| EDA and visualisation | 1 |
| Code quality and reproducibility | 1 |

Додатковий бал відданий justification, а не accuracy: це єдина частина роботи, яку читає людина,
і саме там перевіряється розуміння. Якщо хочеш навпаки — міняються два числа.

## 5. Механіка здачі: дві фази замість однієї

80 хвилин недостатньо, щоб специфікувати ARIMA вручну, обґрунтувати її й зібрати сабмішн.

| Місце | Було | Стало |
|---|---|---|
| Overview | `you build, evaluate and submit your solution within the class period` | `you build and submit a valid baseline within the class period, and may refine it until 23:59 the same day` |
| Deliverables | одна здача | **дві**: обов'язковий валідний baseline до кінця слоту + фінальний сабмішн до 23:59 того ж дня |
| Feedback | `The automated accuracy score is published on the same day` | `published the next morning` |

Формулювання для документа:

> Each challenge has two deadlines. A valid baseline submission — correct format, any model — is
> compulsory before the end of the session; this is what makes the work in-class. The final submission
> is due at 23:59 the same day. The automated score is published the following morning, and rubric
> marks are returned before the next practice session.

## 6. Бали за точність — порогові, а не рейтингові

Додати в *How the benchmark works*:

> Accuracy points are awarded against **your own series' benchmark**, not against your classmates:
> beating the benchmark scores 3, coming within 10% of it scores 2, and a worse result scores 1 with a
> required written explanation. Ranking against the class occurs only in the final project, where
> everyone forecasts the same dataset.

## 7. Пул серій: один ряд на весь курс, джерело з коваріатами

| Було | Стало |
|---|---|
| `drawn deterministically from a curated pool by a hash of the student identifier and the lab number` | `by a hash of the student identifier` — **один ряд на весь курс**, видається на тижні 4 |
| `Series are taken from the M3 and M4 forecasting competition archives` | ряди з **екзогенними змінними** (енергетика / ритейл / трафік з погодою і календарем) |

Дві причини, обидві блокуючі:

1. **Тиждень 7 — SARIMAX / dynamic regression.** Ряди M3/M4 анонімні й **не мають екзогенних змінних**.
   На такому ряді SARIMAX вироджується в ARIMA з календарними регресорами, тобто найтехнічніша лекція
   курсу не має практики.
2. **Тиждень 9 — foundation models.** Корпуси претрейну (Chronos та ін.) містять дані M-змагань.
   Рескейл і зсув календаря ховають ідентичність ряду від студента, але не ховають його форму від
   моделі, яка цю форму вже бачила. Zero-shot вимірював би пам'ять, а не узагальнення.

Згадка M3/M4 лишається — у ролі бонусного сюжету тижня 9: додаткова серія з M4 поруч із «чистою»,
щоб студент побачив витік претрейну на власному екрані.

## 8. Тиждень 9 у плані курсу

| Було | Стало |
|---|---|
| `overview of foundation models` (один буліт) | повноцінний блок: zero-shot прогноз на власному ряді, порівняння з моделлю, яку студент будував сім тижнів, і демонстрація витоку претрейну |

Джерело: *Time Series Forecasting Using Foundation Models*. Торішні матеріали цього блоку не містять
взагалі — його треба писати з нуля.

## 9. Пропуски, заміни, тривоги

| Місце | Було | Стало |
|---|---|---|
| Missed weekly assignment | `Capped at 5 of 7 points` | `Capped at 6 of 8 points` |
| Air raid alert during a practice session | `exceeds 20 minutes` | `exceeds 10 minutes` |
| Missed Week 10 challenge (make-up) | `the same 90 minutes of working time` | `the same 80 minutes of working time` |

Поріг 20 хвилин писався під 240-хвилинне заняття. У 80-хвилинному слоті це чверть пари.

## 10. Дрібне

- Посилання на курс у Moodle: `id=3416` → **`id=4432`** (уже виправлено в усіх матеріалах 26autumn).
- `Course language: English/Ukrainian` — уточнити: **матеріали англійською, викладання українською**.
- `Proficiency in R or Python` — уточнити: **курс викладається на Python**; R приймається у здачах,
  але шаблони й підтримка тільки для Python.
- Track A: пул із 50 іменованих серій, 39 студентів → до трьох студентів можуть отримати одну серію.
  Це прийнятно: Track A за задумом не змагальний, а завдання ідентичне для всіх.

---

## Що з цього треба сказати вголос завтра

Пункти 1, 3, 5, 6 змінюють те, що студент прочитає в PDF. Нульова презентація вже містить **виправлені**
цифри й механіку. Якщо документ роздається у старій редакції — розбіжність помітять на першому ж тижні.
