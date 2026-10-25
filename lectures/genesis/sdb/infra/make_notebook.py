"""Build labs/forecast_your_metric.ipynb - the hands-on part of the lecture, for Google Colab.

    python infra/make_notebook.py
"""

from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parent.parent / "labs" / "forecast_your_metric.ipynb"
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Aranaur/aranaur.rbind.io/blob/main/lectures/genesis/sdb/labs/forecast_your_metric.ipynb)

# Forecast your own metric with a foundation model

*School of Digital Business 5.0 — Time Series in the AI Era.*

Four steps: **pick a series → backtest → read the result → decide.** Run the cells from top to bottom
(`Shift + Enter`). The first run downloads the model, about 500 MB; it takes a minute or two.

**Use public or anonymised data only.** Everything runs inside your Colab session; nothing is sent to a
forecasting service.
""")

md("## 0. Install")
code(r"""
!pip install -q chronos-forecasting
import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from chronos import BaseChronosPipeline
model = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
print("ready")
""")

md(r"""
## 1. Pick a series

**Option A** — one of the lecture's ten products: set `PRODUCT` below.

**Option B** — your own CSV with two columns, a date and a value, one row per day (or per week). Set
`PRODUCT = None`, run the cell, and upload the file when asked.
""")
code(r"""
PRODUCT = "ChatGPT"   # ChatGPT, Duolingo, Spotify, Netflix, Tinder, Strava, TikTok, Notion, Threads, Black Friday

if PRODUCT:
    url = ("https://raw.githubusercontent.com/Aranaur/aranaur.rbind.io/main/"
           "lectures/genesis/sdb/data/pageviews.csv")
    data = pd.read_csv(url, parse_dates=["ds"])
    y = data[data["unique_id"] == PRODUCT].set_index("ds")["y"].dropna()
else:
    from google.colab import files
    uploaded = files.upload()
    raw = pd.read_csv(next(iter(uploaded)))
    raw.columns = ["ds", "y"] + list(raw.columns[2:])
    y = raw.assign(ds=pd.to_datetime(raw["ds"])).set_index("ds")["y"].sort_index().dropna()

y = y[y.index >= y.index.max() - pd.Timedelta(days=3 * 365)]       # the last three years are plenty
print(f"{len(y)} points, {y.index.min():%Y-%m-%d} to {y.index.max():%Y-%m-%d}")
y.plot(figsize=(11, 3), title="Your series"); plt.show()
""")

md(r"""
## 2. Backtest: the model against "same period last week"

Six windows of 28 steps at the end of the history. In each window both forecasters see only what came
before it — exactly what you would have had on that day.

`SEASON` is the length of the repeating pattern: **7** for daily data with a weekly rhythm, **52** for
weekly data with a yearly one.
""")
code(r"""
H, WINDOWS, SEASON = 28, 6, 7

rows = []
for k in range(WINDOWS):
    end = len(y) - (WINDOWS - k) * H
    history, actual = y.iloc[:end], y.iloc[end:end + H].to_numpy()
    q, _ = model.predict_quantiles([torch.tensor(history.to_numpy(), dtype=torch.float32)],
                                   prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    q = np.asarray(q[0]).reshape(H, 3)
    rule = history.to_numpy()[-SEASON:][np.arange(H) % SEASON]
    rows.append(pd.DataFrame({"ds": y.index[end:end + H], "actual": actual, "rule": rule,
                              "P10": q[:, 0], "P50": q[:, 1], "P90": q[:, 2], "window": k + 1}))
bt = pd.concat(rows)

past = y.iloc[:len(y) - WINDOWS * H].to_numpy()
scale = np.mean(np.abs(past[SEASON:] - past[:-SEASON]))            # typical miss of the rule, before the test
mase = lambda f: np.mean(np.abs(bt["actual"] - bt[f])) / scale
print(f"MASE  same period last week: {mase('rule'):.2f}   Chronos-2: {mase('P50'):.2f}   (lower is better)")
print(f"Chronos-2 better in {sum(np.mean(np.abs(g.actual - g.P50)) < np.mean(np.abs(g.actual - g.rule)) for _, g in bt.groupby('window'))} of {WINDOWS} windows")
print(f"Share of actual values inside P10-P90: {np.mean((bt.actual >= bt.P10) & (bt.actual <= bt.P90)):.0%}  (about 80% if calibrated)")

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(y.index[-(WINDOWS * H + 60):], y.iloc[-(WINDOWS * H + 60):], color="black", lw=1, label="actual")
for _, g in bt.groupby("window"):
    ax.fill_between(g.ds, g.P10, g.P90, color="#e64173", alpha=0.2)
    ax.plot(g.ds, g.P50, color="#e64173", lw=1.5)
    ax.plot(g.ds, g.rule, "--", color="#181485", lw=1)
ax.set_title("Backtest: Chronos-2 (pink, P10-P90 band) and same period last week (dashed)")
plt.show()
""")

md(r"""
## 3. The forecast you would use

If the backtest says the model is worth it, this is the next 28 steps with the range.
""")
code(r"""
q, _ = model.predict_quantiles([torch.tensor(y.to_numpy(), dtype=torch.float32)], prediction_length=H,
                               quantile_levels=[0.1, 0.5, 0.9])
q = np.asarray(q[0]).reshape(H, 3)
step = y.index[-1] - y.index[-2]
future = pd.date_range(y.index[-1] + step, periods=H, freq=step)
forecast = pd.DataFrame({"P10": q[:, 0], "P50": q[:, 1], "P90": q[:, 2]}, index=future)

fig, ax = plt.subplots(figsize=(12, 3.5))
ax.plot(y.index[-120:], y.iloc[-120:], color="black", lw=1)
ax.fill_between(future, forecast.P10, forecast.P90, color="#e64173", alpha=0.2, label="P10-P90")
ax.plot(future, forecast.P50, color="#e64173", lw=2, label="P50")
ax.legend(); plt.show()
forecast.round(0).head(10)
""")

md(r"""
## 4. Decide — write it down

Answer in two or three sentences, for your team:

1. **Does the model beat the rule?** By how much (MASE), and in how many of the six windows?
2. **Can you trust the range?** Is the share inside P10–P90 near 80%, or far from it?
3. **What would you plan with it** — and what would make you change the plan?
4. **What does the model not know** about the next 28 days — a campaign, a launch, a holiday?
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"colab": {"provenance": []}, "kernelspec": {"display_name": "Python 3", "name": "python3"},
                  "language_info": {"name": "python"}}
nbf.write(nb, OUT)
print("written", OUT)
