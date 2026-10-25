"""Build the practice notebooks for Google Colab, in English and in Ukrainian.

    python infra/make_notebook.py

writes, in labs/:
  worked_example.ipynb, worked_example_uk.ipynb  - the demonstration everyone follows: one product, nine
                                                   forecasters, a full backtest, conclusions;
  template.ipynb, template_uk.ipynb              - the same toolkit, for one's own series.
The worked examples are stored executed (see infra/run_notebooks.py), so the results show without running.
"""

from pathlib import Path

import nbformat as nbf

LABS = Path(__file__).resolve().parent.parent / "labs"
COLAB = "https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/genesis/sdb/labs/"
DATA_URL = "https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/lectures/genesis/sdb/data/pageviews.csv"

INSTALL = r"""
!pip install -q chronos-forecasting statsforecast timesfm tirex-ts
"""

TOOLKIT = r"""
import os, time, warnings
os.environ["NIXTLA_ID_AS_COL"] = "true"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
warnings.filterwarnings("ignore")
import transformers
transformers.utils.logging.disable_progress_bar()
transformers.utils.logging.set_verbosity_error()
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, MSTL, Naive, SeasonalNaive

plt.rcParams.update({"figure.figsize": (12, 3.6), "axes.grid": True, "grid.alpha": 0.3})
PINK, NAVY, TEAL, ORANGE, SLATE = "#e64173", "#181485", "#20B2AA", "#FFA500", "#314f4f"
CLASSICAL = {"Naive": lambda s: Naive(alias="Naive"),
             "Seasonal naive": lambda s: SeasonalNaive(s, alias="Seasonal naive"),
             "ETS": lambda s: AutoETS(season_length=s, alias="ETS"),
             "ARIMA": lambda s: AutoARIMA(season_length=s, alias="ARIMA"),
             "Theta": lambda s: AutoTheta(season_length=s, alias="Theta"),
             "MSTL": lambda s: MSTL(season_length=[s, 365], alias="MSTL")}
_loaded = {}


def foundation(name):
    # Load each foundation model once; the first call downloads its weights.
    if name not in _loaded:
        if name == "Chronos-2":
            from chronos import BaseChronosPipeline
            _loaded[name] = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        elif name == "TimesFM":
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
            m.compile(timesfm.ForecastConfig(max_context=1024, max_horizon=128, normalize_inputs=True,
                                             use_continuous_quantile_head=True))
            _loaded[name] = m
        elif name == "TiRex":
            import tirex
            _loaded[name] = tirex.load_model("NX-AI/TiRex", device="cpu")
    return _loaded[name]


def forecast_fm(name, history, h):
    # Median and 80% interval (P10, P90) from a foundation model, for one series.
    ctx = torch.tensor(np.asarray(history, dtype=np.float32))
    model = foundation(name)
    if name == "Chronos-2":
        q, _ = model.predict_quantiles([ctx], prediction_length=h, quantile_levels=[0.1, 0.5, 0.9])
        q = np.asarray(q[0]).reshape(h, 3)
        return q[:, 1], q[:, 0], q[:, 2]
    if name == "TimesFM":
        point, q = model.forecast(horizon=h, inputs=[ctx.numpy()])
        q = np.asarray(q)[0]
        return np.asarray(point)[0], q[:, 1], q[:, 9]
    q, mean = model.forecast(context=[ctx], prediction_length=h)
    q = np.asarray(q)[0]
    return np.asarray(mean)[0], q[:, 0], q[:, 8]


def backtest(y, h, windows, season, classical, foundations, freq="D"):
    # Rolling-origin backtest: `windows` cuts of `h` steps at the end of y. One row per date and cut.
    df = pd.DataFrame({"unique_id": "series", "ds": y.index, "y": y.values})
    timing = {}
    start = time.time()
    models = [CLASSICAL[name](season) for name in classical]
    res = StatsForecast(models=models, freq=freq).cross_validation(df=df, h=h, n_windows=windows,
                                                                   step_size=h, level=[80])
    for name in classical:
        timing[name] = (time.time() - start) / len(classical)
    for name in foundations:
        start, parts = time.time(), []
        for cut in sorted(res["cutoff"].unique()):
            mid, lo, hi = forecast_fm(name, y[y.index <= cut].values, h)
            parts.append(pd.DataFrame({"ds": res.loc[res["cutoff"] == cut, "ds"].values, "cutoff": cut,
                                       name: mid, f"{name}-lo-80": lo, f"{name}-hi-80": hi}))
        res = res.merge(pd.concat(parts), on=["ds", "cutoff"])
        timing[name] = time.time() - start
    res["h"] = res.groupby("cutoff").cumcount() + 1
    return res, timing


def score(res, y, season, names, timing=None):
    # MASE (scaled by the seasonal-naive error in the year before the test), MAPE, bias, 80% coverage.
    past = y[y.index <= res["cutoff"].min()].values[-365:]
    scale = np.mean(np.abs(past[season:] - past[:-season]))
    rows = {}
    for name in names:
        e = res["y"] - res[name]
        row = {"MASE": np.mean(np.abs(e)) / scale, "MAPE, %": 100 * np.mean(np.abs(e) / res["y"]),
               "bias, %": 100 * np.mean(-e / res["y"])}
        if f"{name}-lo-80" in res:
            row["inside P10-P90"] = np.mean((res["y"] >= res[f"{name}-lo-80"]) & (res["y"] <= res[f"{name}-hi-80"]))
        if timing and name in timing:
            row["seconds"] = timing[name]
        rows[name] = row
    return pd.DataFrame(rows).T, scale
"""

T = {
    "en": {
        "worked": "worked_example.ipynb", "template": "template.ipynb",
        "w_intro": """
# Worked example: forecasting ChatGPT's attention, end to end

*School of Digital Business 5.0 — Time Series in the AI Era.*

One product, the whole procedure: **look at the data → backtest nine forecasters → read the errors, the
horizons and the intervals → combine → forecast → conclude.** The series is the daily number of views of
the English Wikipedia article on ChatGPT — a public proxy for attention to an AI product.

The notebook is stored **with its results**: you can read it without running anything. To rerun it in
Colab, use `Runtime → Run all`; the first run downloads three models (about 1.5 GB) and takes several
minutes.
""",
        "setup": "## 0. Setup",
        "w_look": """
## 1. Look before forecasting

Three questions before any model: is there a **trend**, a **weekly** rhythm, a **yearly** season?
""",
        "w_look_after": """
**What the charts say**

- **Weekly rhythm:** Tuesday–Wednesday about 15% above a year's average day, Saturday about 20–25%
  below — the same shape every year: the pattern of work and study.
- **Yearly season — but not a stable one:** deep summer dips in 2024 and 2026 (July–August, school and
  university holidays), none in 2025, when a June spike dominated.
- **Level:** waves rather than a trend — a rise through 2025, a slow decline in 2026, and a jump in the
  very last week of the data.

So a good forecaster must reproduce the week, should not lean too hard on a yearly shape that changes,
and must not extrapolate a trend that is not there.
""",
        "w_design": """
## 2. The backtest

**Six windows of 28 days**, May to October 2026, each forecast only from the days before it — all after
the release of every foundation model used here. Nine forecasters:

| Classical (fitted to this series) | Foundation models (no fitting) |
|---|---|
| Naive · Seasonal naive · ETS · ARIMA · Theta · MSTL (week + year) | Chronos-2 · TimesFM 2.5 · TiRex |

This cell takes a few minutes — ARIMA and the three downloads are the slow parts.
""",
        "w_score": """
## 3. The scoreboard

**MASE** below 1 beats "same day last week" (as it did in the year before the test). **Bias** is the
average over- (+) or under-forecast. **Inside P10–P90** should be close to 80%.
""",
        "w_score_after": """
**What the table says**

- **The best model on this series is not the best on average.** Across the lecture's ten products
  Chronos-2 led; here **TiRex and Theta** — a classical method from 2000 — are ahead of it.
- **MSTL** with a yearly season is the worst: four years of history are too few to estimate a yearly
  shape that also changes.
- Look at the last column: the classical intervals are **too wide** (almost every day inside), the
  foundation models' **too narrow** (about 70–77%). Neither is calibrated out of the box.
""",
        "w_horizon": """
## 4. Errors by horizon and by window
""",
        "w_horizon_after": """
- **The first week is the easy one:** its error is about half of the error in weeks 2–4. Re-forecast
  weekly rather than trusting a four-week forecast to the end.
- **Windows differ more than models.** Every model has bad windows (May) and good ones (August). One
  window would have told a different story — which is why there are six.
""",
        "w_combo": """
## 5. A combination, chosen in advance

Week 8 of any forecasting course: an **average** of models is rarely the best and almost never bad. We fix
the pair **before** looking at the scores — one classical model, one foundation model: ETS + Chronos-2.
""",
        "w_combo_after": """
The average beats **both** of its parts. It is not the single best model here — but choosing the single
best by its test score is exactly the mistake a backtest exists to avoid.
""",
        "w_final": """
## 6. The forecast for the next four weeks

From the full history to 24 October 2026: Chronos-2 with its range, and the ETS + Chronos-2 average.
The series jumped in its last week; both forecasts start high and fall back towards the usual weekly
pattern. Whether the jump lasts is exactly what the width of the range is saying.
""",
        "w_conclusions": """
## 7. Conclusions

1. **Backtest on your own series.** The model that wins on average (Chronos-2) is mid-table here; a
   2000-era method (Theta) is near the top. No ranking transfers.
2. **The baselines matter.** Six of eight models beat "same day last week" — but not by much, and MSTL
   did not at all.
3. **Plan with the horizon in mind.** Week 1 is roughly twice as accurate as weeks 2–4: refresh the
   forecast every week.
4. **Check the range before using it.** Classical intervals here are too wide, foundation-model
   intervals too narrow. Widen or narrow them by what the backtest shows.
5. **Combine.** A pre-chosen average of a classical and a foundation model beat both — the cheapest
   insurance in forecasting.
""",
        "t_intro": """
# Template: your own forecasting experiment

*School of Digital Business 5.0 — Time Series in the AI Era.*

The same toolkit as the worked example, for **your** series. Change only the **settings** cell, run
everything (`Runtime → Run all`), then write your conclusions in the last section.

**Public or anonymised data only.** Everything runs inside your Colab session.
""",
        "t_settings": """
## 1. Settings

- `PRODUCT` — one of the lecture's ten products, or `None` to upload your own CSV (two columns: date,
  value; one row per day or per week).
- `SEASON` — 7 for daily data with a weekly rhythm, 52 for weekly data.
- `H` — how far ahead; `WINDOWS` — how many backtest cuts.
- `CLASSICAL`, `FOUNDATION` — which forecasters to compare. ARIMA and MSTL are slow on long daily series.
""",
        "t_look": "## 2. Look at your data",
        "t_run": "## 3. Backtest",
        "t_score": "## 4. Scoreboard",
        "t_plots": "## 5. By horizon and by window",
        "t_final": "## 6. Forecast with the backtest winner",
        "t_conclusions": """
## 7. Your conclusions

Write two or three sentences for each — the numbers above are your evidence:

1. **Which forecaster won, and by how much** over "same period last week"? In how many windows?
2. **Is the range calibrated?** Which model's P10–P90 is closest to 80%?
3. **How does the error grow with the horizon?** How often would you re-forecast?
4. **What would you plan** with this forecast — and **what does it not know** about the coming weeks?
""",
        "lbl": {"actual": "actual", "weekly": "7-day average", "dow": "Average day of week, by year (% of the year's mean)",
                "month": "Average by month, by year (% of the year's mean)", "daily": "Daily views",
                "byh": "MASE by week of the horizon", "byw": "MASE by backtest window",
                "cover": "Share of days inside the 80% interval (target: 80%)", "week": "week",
                "final": "Next 28 days", "combo": "ETS + Chronos-2", "range": "Chronos-2 P10–P90",
                "days": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], "uploaded": "Uploaded"},
    },
    "uk": {
        "worked": "worked_example_uk.ipynb", "template": "template_uk.ipynb",
        "w_intro": """
# Розібраний приклад: прогноз уваги до ChatGPT від початку до кінця

*School of Digital Business 5.0 — Часові ряди в епоху AI.*

Один продукт, уся процедура: **подивитися на дані → забектестити дев'ять прогнозистів → прочитати похибки,
горизонти й інтервали → скомбінувати → спрогнозувати → зробити висновки.** Ряд — щоденна кількість
переглядів статті про ChatGPT в англійській Вікіпедії, публічний індикатор уваги до AI-продукту.

Ноутбук збережено **з результатами**: його можна читати, нічого не запускаючи. Щоб перезапустити в Colab —
`Runtime → Run all`; під час першого запуску завантажуються три моделі (близько 1,5 ГБ), це кілька хвилин.
""",
        "setup": "## 0. Налаштування",
        "w_look": """
## 1. Спершу подивитися на дані

Три питання до будь-якої моделі: чи є **тренд**, **тижневий** ритм, **річна** сезонність?
""",
        "w_look_after": """
**Що кажуть графіки**

- **Тижневий ритм:** вівторок–середа приблизно на 15% вищі за середній день року, субота — на 20–25% нижча;
  форма та сама щороку — ритм роботи й навчання.
- **Річна сезонність — але нестабільна:** глибокі літні провали у 2024 і 2026 роках (липень–серпень,
  канікули), а у 2025-му — жодного, бо домінував червневий сплеск.
- **Рівень:** хвилі, а не тренд — зростання протягом 2025 року, повільний спад у 2026-му і стрибок в
  останній тиждень даних.

Отже, хороший прогнозист має відтворити тиждень, не надто покладатися на річну форму, яка змінюється, і не
продовжувати тренд, якого немає.
""",
        "w_design": """
## 2. Бектест

**Шість вікон по 28 днів**, травень–жовтень 2026, кожне прогнозується лише з днів до нього — і всі вони
після релізу кожної foundation model з тесту. Дев'ять прогнозистів:

| Класичні (підганяються під цей ряд) | Foundation models (без підгонки) |
|---|---|
| Наївний · Сезонний наївний · ETS · ARIMA · Theta · MSTL (тиждень + рік) | Chronos-2 · TimesFM 2.5 · TiRex |

Ця клітинка працює кілька хвилин — найдовше рахуються ARIMA і завантаження трьох моделей.
""",
        "w_score": """
## 3. Таблиця результатів

**MASE** менше 1 — краще за «як тиждень тому» (як воно було в рік до тесту). **Зміщення** — середнє
завищення (+) чи заниження (−). **Inside P10–P90** має бути близько 80%.
""",
        "w_score_after": """
**Що каже таблиця**

- **Найкраща модель на цьому ряді — не найкраща в середньому.** На десяти продуктах лекції лідирувала
  Chronos-2; тут попереду **TiRex і Theta** — класичний метод 2000 року.
- **MSTL** з річною сезонністю — найгірший: чотирьох років історії замало, щоб оцінити річну форму, яка ще й
  змінюється.
- Остання колонка: класичні інтервали **занадто широкі** (майже всі дні всередині), інтервали foundation
  models — **занадто вузькі** (близько 70–77%). З коробки не відкалібровані ні ті, ні інші.
""",
        "w_horizon": "## 4. Похибки за горизонтом і за вікнами",
        "w_horizon_after": """
- **Перший тиждень — найлегший:** його похибка приблизно вдвічі менша, ніж у тижнях 2–4. Перепрогнозуйте
  щотижня, а не довіряйте чотиритижневому прогнозу до кінця.
- **Вікна відрізняються сильніше, ніж моделі.** У кожної моделі є погані вікна (травень) і хороші
  (серпень). Одне вікно розповіло б іншу історію — тому їх шість.
""",
        "w_combo": """
## 5. Комбінація, обрана заздалегідь

Тиждень 8 будь-якого курсу прогнозування: **середнє** моделей рідко найкраще і майже ніколи не погане.
Фіксуємо пару **до** того, як дивимося на результати — одна класична модель, одна foundation model: ETS + Chronos-2.
""",
        "w_combo_after": """
Середнє обіграє **обидві** свої складові. Це не найкраща окрема модель — але обирати найкращу окрему модель за
її результатом на тесті і є тією помилкою, від якої бектест захищає.
""",
        "w_final": """
## 6. Прогноз на наступні чотири тижні

З усієї історії до 24 жовтня 2026: Chronos-2 з діапазоном і середнє ETS + Chronos-2. В останній тиждень
ряд стрибнув угору; обидва прогнози стартують високо й повертаються до звичного тижневого патерну. Чи
стрибок збережеться — саме про це говорить ширина діапазону.
""",
        "w_conclusions": """
## 7. Висновки

1. **Бектестуйте на власному ряді.** Модель, що виграє в середньому (Chronos-2), тут у середині таблиці;
   метод 2000 року (Theta) — серед лідерів. Рейтинги не переносяться.
2. **Бенчмарки важливі.** Шість із восьми моделей обіграли «як тиждень тому» — але ненабагато, а MSTL не
   обіграв зовсім.
3. **Плануйте з урахуванням горизонту.** Перший тиждень приблизно вдвічі точніший за тижні 2–4:
   оновлюйте прогноз щотижня.
4. **Перевірте діапазон, перш ніж ним користуватися.** Класичні інтервали тут занадто широкі, інтервали
   foundation models — занадто вузькі. Розширюйте чи звужуйте їх за тим, що показав бектест.
5. **Комбінуйте.** Заздалегідь обране середнє класичної та foundation model обіграло обидві — найдешевша
   страховка в прогнозуванні.
""",
        "t_intro": """
# Шаблон: ваш власний експеримент з прогнозування

*School of Digital Business 5.0 — Часові ряди в епоху AI.*

Той самий інструментарій, що й у розібраному прикладі, для **вашого** ряду. Змініть лише клітинку
**налаштувань**, запустіть усе (`Runtime → Run all`), а потім запишіть висновки в останньому розділі.

**Лише публічні або анонімізовані дані.** Усе працює всередині вашої сесії Colab.
""",
        "t_settings": """
## 1. Налаштування

- `PRODUCT` — один із десяти продуктів лекції або `None`, щоб завантажити власний CSV (дві колонки: дата,
  значення; один рядок на день або на тиждень).
- `SEASON` — 7 для щоденних даних з тижневим ритмом, 52 для щотижневих.
- `H` — на скільки вперед; `WINDOWS` — скільки вікон бектесту.
- `CLASSICAL`, `FOUNDATION` — які прогнозисти порівнювати. ARIMA і MSTL повільні на довгих щоденних рядах.
""",
        "t_look": "## 2. Подивіться на свої дані",
        "t_run": "## 3. Бектест",
        "t_score": "## 4. Таблиця результатів",
        "t_plots": "## 5. За горизонтом і за вікнами",
        "t_final": "## 6. Прогноз переможцем бектесту",
        "t_conclusions": """
## 7. Ваші висновки

Дві-три фрази на кожне питання — цифри вище і є вашими доказами:

1. **Який прогнозист переміг і наскільки** обіграв «як тиждень тому»? У скількох вікнах?
2. **Чи відкалібрований діапазон?** Чий P10–P90 найближчий до 80%?
3. **Як похибка росте з горизонтом?** Як часто ви б перепрогнозовували?
4. **Що б ви спланували** за цим прогнозом — і **чого він не знає** про наступні тижні?
""",
        "lbl": {"actual": "факт", "weekly": "середнє за 7 днів", "dow": "Середній день тижня за роками (% від середнього року)",
                "month": "Середнє за місяцями за роками (% від середнього року)", "daily": "Щоденні перегляди",
                "byh": "MASE за тижнями горизонту", "byw": "MASE за вікнами бектесту",
                "cover": "Частка днів усередині 80% інтервалу (ціль: 80%)", "week": "тиждень",
                "final": "Наступні 28 днів", "combo": "ETS + Chronos-2", "range": "Chronos-2 P10–P90",
                "days": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Нд"], "uploaded": "Завантажений ряд"},
    },
}

LOOK = r"""
fig, ax = plt.subplots()
ax.plot(y.index, y.values, color="lightgray", lw=0.8, label="{daily}")
ax.plot(y.index, y.rolling(7).mean(), color=SLATE, lw=1.5, label="{weekly}")
ax.legend(loc="upper left"); ax.set_title(TITLE); plt.show()

recent = y[y.index >= y.index.max() - pd.Timedelta(days=3 * 365)]
rel = recent / recent.groupby(recent.index.year).transform("mean") - 1
fig, axes = plt.subplots(1, 2, figsize=(12, 3.4))
for year, g in rel.groupby(rel.index.year):
    axes[0].plot(range(7), g.groupby(g.index.dayofweek).mean().reindex(range(7)).values, "o-", label=str(year))
    axes[1].plot(range(1, 13), g.groupby(g.index.month).mean().reindex(range(1, 13)).values, "o-", label=str(year))
axes[0].set_xticks(range(7), {days}); axes[0].set_title("{dow}", fontsize=10)
axes[1].set_xticks(range(1, 13)); axes[1].set_title("{month}", fontsize=10)
for ax in axes:
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.legend(fontsize=8)
plt.show()
"""

PLOTS = r"""
by_h = res.groupby((res["h"] - 1) // 7).apply(
    lambda g: pd.Series({{m: np.mean(np.abs(g["y"] - g[m])) / scale for m in SHOW}}), include_groups=False)
by_w = res.groupby("cutoff").apply(
    lambda g: pd.Series({{m: np.mean(np.abs(g["y"] - g[m])) / scale for m in NAMES}}), include_groups=False)
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), width_ratios=[1, 1.4])
for m in SHOW:
    axes[0].plot(by_h.index + 1, by_h[m], "o-", label=m)
axes[0].set_xticks(by_h.index + 1); axes[0].set_xlabel("{week}"); axes[0].set_title("{byh}", fontsize=10)
axes[0].legend(fontsize=8)
im = axes[1].imshow(by_w.T.values, cmap="RdYlGn_r", vmin=0.2, vmax=1.6, aspect="auto")
axes[1].set_yticks(range(len(NAMES)), NAMES); axes[1].set_xticks(range(len(by_w)), [f"{{c:%d.%m}}" for c in by_w.index])
for i in range(len(NAMES)):
    for j in range(len(by_w)):
        axes[1].text(j, i, f"{{by_w.iloc[j, i]:.2f}}", ha="center", va="center", fontsize=7)
axes[1].set_title("{byw}", fontsize=10)
plt.show()

covered = board["inside P10-P90"].dropna()
fig, ax = plt.subplots(figsize=(12, 2.8))
ax.bar(covered.index, covered.values, color=[PINK if m in FOUNDATION else SLATE for m in covered.index])
ax.axhline(0.8, color="black", lw=1, ls="--")
ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0)); ax.set_ylim(0, 1.05)
ax.set_title("{cover}", fontsize=10); plt.show()
"""


def cells_for(kind, lang):
    t, lbl = T[lang], T[lang]["lbl"]
    cells = []
    md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
    code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))
    look = LOOK.format(**{k: (repr(v) if k == "days" else v) for k, v in lbl.items()})
    plots = PLOTS.format(**lbl)

    if kind == "worked":
        md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{t['worked']})\n" + t["w_intro"])
        md(t["setup"])
        code(INSTALL)
        code(TOOLKIT)
        md(t["w_look"])
        code(f"""
data = pd.read_csv("{DATA_URL}", parse_dates=["ds"])
y = data[data["unique_id"] == "ChatGPT"].set_index("ds")["y"].asfreq("D")
print(len(y), y.index.min().date(), y.index.max().date())
""" + look.replace("TITLE", "'ChatGPT'"))
        md(t["w_look_after"])
        md(t["w_design"])
        code("""
H, WINDOWS, SEASON = 28, 6, 7
CLASSICAL_MODELS = ["Naive", "Seasonal naive", "ETS", "ARIMA", "Theta", "MSTL"]
FOUNDATION = ["Chronos-2", "TimesFM", "TiRex"]
res, timing = backtest(y, H, WINDOWS, SEASON, CLASSICAL_MODELS, FOUNDATION)
NAMES = CLASSICAL_MODELS + FOUNDATION
print("cut-offs:", [f"{c:%Y-%m-%d}" for c in sorted(res["cutoff"].unique())])
""")
        md(t["w_score"])
        code("""
board, scale = score(res, y, SEASON, NAMES, timing)
board.round(3).style.background_gradient(subset=["MASE"], cmap="RdYlGn_r").format(precision=3)
""")
        md(t["w_score_after"])
        md(t["w_horizon"])
        code("SHOW = ['Seasonal naive', 'ETS', 'Theta', 'Chronos-2', 'TiRex']\n" + plots)
        md(t["w_horizon_after"])
        md(t["w_combo"])
        code(f"""
res["{lbl['combo']}"] = res[["ETS", "Chronos-2"]].mean(axis=1)
combo_board, _ = score(res, y, SEASON, ["ETS", "Chronos-2", "{lbl['combo']}"])
combo_board[["MASE", "MAPE, %"]].round(3)
""")
        md(t["w_combo_after"])
        md(t["w_final"])
        code(f"""
mid, lo, hi = forecast_fm("Chronos-2", y.values, H)
ets = StatsForecast(models=[AutoETS(season_length=SEASON)], freq="D").forecast(
    df=pd.DataFrame({{"unique_id": "s", "ds": y.index, "y": y.values}}), h=H)["AutoETS"].values
future = pd.date_range(y.index[-1] + pd.Timedelta(days=1), periods=H)
fig, ax = plt.subplots()
ax.plot(y.index[-90:], y.values[-90:], color=SLATE, lw=1, label="{lbl['actual']}")
ax.fill_between(future, lo, hi, color=PINK, alpha=0.2, label="{lbl['range']}")
ax.plot(future, mid, color=PINK, lw=2, label="Chronos-2")
ax.plot(future, (mid + ets) / 2, "--", color=NAVY, lw=1.5, label="{lbl['combo']}")
ax.legend(loc="upper left"); ax.set_title("{lbl['final']}"); plt.show()
pd.DataFrame({{"P10": lo, "P50": mid, "P90": hi, "{lbl['combo']}": (mid + ets) / 2}}, index=future).round(0).head(14)
""")
        md(t["w_conclusions"])
    else:
        md(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({COLAB}{t['template']})\n" + t["t_intro"])
        md(t["setup"])
        code(INSTALL)
        code(TOOLKIT)
        md(t["t_settings"])
        code("""
PRODUCT = "Duolingo"   # ChatGPT, Duolingo, Spotify, Netflix, Tinder, Strava, TikTok, Notion, Threads, Black Friday, or None
SEASON = 7             # 7 for daily data, 52 for weekly data
H = 28                 # forecast horizon, in steps
WINDOWS = 6            # backtest windows
CLASSICAL_MODELS = ["Seasonal naive", "ETS", "Theta"]   # also: "Naive", "ARIMA", "MSTL" (slow)
FOUNDATION = ["Chronos-2", "TiRex"]                     # also: "TimesFM"
""")
        md(t["t_look"])
        code(f"""
if PRODUCT:
    data = pd.read_csv("{DATA_URL}", parse_dates=["ds"])
    y = data[data["unique_id"] == PRODUCT].set_index("ds")["y"].dropna()
    TITLE = PRODUCT
else:
    from google.colab import files
    uploaded = files.upload()
    raw = pd.read_csv(next(iter(uploaded)))
    raw.columns = ["ds", "y"] + list(raw.columns[2:])
    y = raw.assign(ds=pd.to_datetime(raw["ds"])).set_index("ds")["y"].sort_index().dropna()
    TITLE = "{lbl['uploaded']}"
FREQ = pd.infer_freq(y.index[-30:]) or "D"
y = y.asfreq(FREQ).interpolate()
print(len(y), y.index.min().date(), y.index.max().date(), "frequency:", FREQ)
""" + look.replace("TITLE", "TITLE"))
        md(t["t_run"])
        code("""
res, timing = backtest(y, H, WINDOWS, SEASON, CLASSICAL_MODELS, FOUNDATION, freq=FREQ)
NAMES = CLASSICAL_MODELS + FOUNDATION
print("cut-offs:", [f"{c:%Y-%m-%d}" for c in sorted(res["cutoff"].unique())])
""")
        md(t["t_score"])
        code("""
board, scale = score(res, y, SEASON, NAMES, timing)
board.round(3).style.background_gradient(subset=["MASE"], cmap="RdYlGn_r").format(precision=3)
""")
        md(t["t_plots"])
        code("SHOW = NAMES\n" + plots)
        md(t["t_final"])
        code(f"""
best = board["MASE"].idxmin()
print("backtest winner:", best)
future = pd.date_range(y.index[-1], periods=H + 1, freq=FREQ)[1:]
if best in FOUNDATION:
    mid, lo, hi = forecast_fm(best, y.values, H)
else:
    out = StatsForecast(models=[CLASSICAL[best](SEASON)], freq=FREQ).forecast(
        df=pd.DataFrame({{"unique_id": "s", "ds": y.index, "y": y.values}}), h=H, level=[80])
    mid, lo, hi = out[best].values, out[f"{{best}}-lo-80"].values, out[f"{{best}}-hi-80"].values
fig, ax = plt.subplots()
ax.plot(y.index[-90:], y.values[-90:], color=SLATE, lw=1, label="{lbl['actual']}")
ax.fill_between(future, lo, hi, color=PINK, alpha=0.2, label="P10–P90")
ax.plot(future, mid, color=PINK, lw=2, label=best)
ax.legend(loc="upper left"); ax.set_title("{lbl['final']}"); plt.show()
pd.DataFrame({{"P10": lo, "P50": mid, "P90": hi}}, index=future).round(0).head(14)
""")
        md(t["t_conclusions"])
    return cells


def write(kind, lang):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells_for(kind, lang)
    nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                      "language_info": {"name": "python"}}
    path = LABS / T[lang][kind]
    nbf.write(nb, path)
    print("written", path)


if __name__ == "__main__":
    for lang in T:
        for kind in ("worked", "template"):
            write(kind, lang)
