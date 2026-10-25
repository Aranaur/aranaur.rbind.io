"""Does a pretrained model forecast a window it saw in pretraining better than one it did not?

Builds data/foundation_leakage.csv for the Week 9 lecture.

M4 Monthly is in the pretraining corpus of Chronos (Ansari et al. 2024, Table 3, "in-domain") and of
Chronos-2 (Ansari et al. 2025, Table 6), with the last H = 18 observations of every series held out.
So for an M4 series the 18 months *before* the official test were seen in pretraining; the official
test was not. M3 Monthly is in neither corpus, so neither window was seen.

For 1000 series from each, both windows are forecast by AutoETS, seasonal naive, Chronos-Bolt (small)
and Chronos-2, and every forecast is scored by MASE (m = 12, scaled on its own context). Memory would
show as Chronos doing better, relative to ETS, on the M4 pretraining window than on the M4 test - by
more than the same comparison moves on M3.

    python foundation_leakage.py              # ~30 minutes on a laptop CPU; downloads M3 and M4 once

The M-competition files go to a cache outside the repository (FORECAST_DATA_CACHE, default
~/.cache/m-competitions): M4 Monthly alone is 90 MB.
"""

import os
import warnings
from pathlib import Path

os.environ["NIXTLA_ID_AS_COL"] = "true"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
from chronos import BaseChronosPipeline
from datasetsforecast.m3 import M3
from datasetsforecast.m4 import M4
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data" / "foundation_leakage.csv"
CACHE = Path(os.environ.get("FORECAST_DATA_CACHE", Path.home() / ".cache" / "m-competitions"))
H, N, SEED = 18, 1000, 840
MODELS = {"Chronos-Bolt": "amazon/chronos-bolt-small", "Chronos-2": "amazon/chronos-2"}


def sample(df: pd.DataFrame, n: int, rng: np.random.Generator) -> pd.DataFrame:
    """n series long enough for two 18-month windows and six years of context, on a monthly calendar."""
    df = df.sort_values(["unique_id", "ds"])
    lengths = df.groupby("unique_id").size()
    keep = rng.choice(lengths[lengths >= 2 * H + 72].index.to_numpy(), n, replace=False)
    out = df[df["unique_id"].isin(keep)].copy()
    out["ds"] = pd.Timestamp("1900-01-01") + out.groupby("unique_id").cumcount().apply(
        lambda k: pd.DateOffset(months=int(k)))
    return out


def mase(history: np.ndarray, actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)) / np.mean(np.abs(history[12:] - history[:-12])))


def main() -> None:
    rng = np.random.default_rng(SEED)
    m3, *_ = M3.load(CACHE, group="Monthly")
    m4, *_ = M4.load(CACHE, group="Monthly")
    datasets = {"M3": sample(m3, N, rng), "M4": sample(m4, N, rng)}
    pipes = {alias: BaseChronosPipeline.from_pretrained(name, device_map="cpu") for alias, name in MODELS.items()}

    rows = []
    for dataset, df in datasets.items():
        for window, cut in [("pretraining window", 2 * H), ("held-out test", H)]:
            groups = [g for _, g in df.groupby("unique_id", sort=True)]
            history = pd.concat([g.iloc[:-cut] for g in groups])
            future = {g["unique_id"].iloc[0]: g["y"].to_numpy()[len(g) - cut:len(g) - cut + H] for g in groups}
            reference = StatsForecast(models=[AutoETS(season_length=12), SeasonalNaive(season_length=12)],
                                      freq="MS").forecast(df=history, h=H)
            context = [torch.tensor(g["y"].to_numpy()[:-cut], dtype=torch.float32) for g in groups]
            pretrained = {}
            for alias, pipe in pipes.items():
                _, mean = pipe.predict_quantiles(context, prediction_length=H, quantile_levels=[0.5])
                pretrained[alias] = [np.asarray(m).ravel()[:H] for m in mean]
            for k, g in enumerate(groups):
                uid = g["unique_id"].iloc[0]
                hist, actual = g["y"].to_numpy()[:-cut], future[uid]
                ref = reference[reference["unique_id"] == uid]
                row = {"dataset": dataset, "window": window, "unique_id": uid,
                       "ETS": mase(hist, actual, ref["AutoETS"].to_numpy()),
                       "snaive": mase(hist, actual, ref["SeasonalNaive"].to_numpy())}
                row.update({alias: mase(hist, actual, pretrained[alias][k]) for alias in pipes})
                rows.append(row)
            print(dataset, window, "done", flush=True)

    pd.DataFrame(rows).round(5).to_csv(OUT, index=False)
    print("written", OUT)


if __name__ == "__main__":
    main()
