"""Moirai on the Walmart stores of the foundation-models lecture: four windows of eight weeks.

Builds data/fm_moirai_walmart.csv and data/fm_moirai_timing.csv for the Week 10 deck.

uni2ts (Moirai) pins numpy < 2, pandas 2.1 and torch < 2.5, so it cannot live in the course .venv.
Run this in its own environment:

    uv venv .venv-moirai --python 3.11
    VIRTUAL_ENV=.venv-moirai uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=.venv-moirai uv pip install uni2ts
    .venv-moirai/Scripts/python infra/moirai_walmart.py         # about a minute

The protocol is the one of Peixeiro (2026), chapter 3 onwards: stores 1-4, the last 32 weeks cut into
four windows of 8; for each window every model sees only the weeks before it. The one exogenous
variable is the holiday flag - the only one whose future is truly known.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
STORES, H, WINDOWS = [1, 2, 3, 4], 8, 4


def load() -> pd.DataFrame:
    w = pd.read_csv(DATA / "walmart.csv", parse_dates=["ds"])
    w = w[w["unique_id"].isin(STORES)].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    w["unique_id"] = w["unique_id"].astype(str)
    return w


def median(forecasts) -> np.ndarray:
    return np.stack([np.median(f.samples, axis=0) if hasattr(f, "samples") else f.quantile(0.5)
                     for f in forecasts])


def quantile(forecasts, q) -> np.ndarray:
    return np.stack([np.quantile(f.samples, q, axis=0) if hasattr(f, "samples") else f.quantile(q)
                     for f in forecasts])


def main() -> None:
    w = load()
    dates = np.sort(w["ds"].unique())
    rows, timing = [], {}
    for k in range(WINDOWS):
        cutoff = dates[len(dates) - (WINDOWS - k) * H - 1]
        history = w[w["ds"] <= cutoff]
        window = w[(w["ds"] > cutoff) & (w["ds"] <= dates[np.searchsorted(dates, cutoff) + H])]
        target_only = PandasDataset.from_long_dataframe(history, item_id="unique_id", timestamp="ds",
                                                        target="y", freq="W-FRI")
        # history plus the future holiday flag; the target over the window is never read
        with_flag = pd.concat([history, window.assign(y=np.nan)])
        flagged = PandasDataset.from_long_dataframe(with_flag, item_id="unique_id", timestamp="ds",
                                                    target="y", freq="W-FRI", feat_dynamic_real=["holiday"])
        flagged_inputs = [{**entry, "target": entry["target"][:-H]} for entry in flagged]

        runs = {
            "Moirai": (MoiraiForecast(module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
                                      prediction_length=H, context_length=128, patch_size="auto",
                                      num_samples=200, target_dim=1, feat_dynamic_real_dim=0,
                                      past_feat_dynamic_real_dim=0), target_only),
            "Moirai + holiday": (MoiraiForecast(module=MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-small"),
                                                prediction_length=H, context_length=128, patch_size="auto",
                                                num_samples=200, target_dim=1, feat_dynamic_real_dim=1,
                                                past_feat_dynamic_real_dim=0), flagged_inputs),
            "Moirai-2.0": (Moirai2Forecast(module=Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small"),
                                           prediction_length=H, context_length=128, target_dim=1,
                                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0), target_only),
        }
        out = window[["unique_id", "ds", "y"]].copy()
        out["cutoff"] = pd.Timestamp(cutoff)
        for name, (model, data) in runs.items():
            start = time.time()
            forecasts = list(model.create_predictor(batch_size=32).predict(data))
            timing[name] = timing.get(name, 0.0) + time.time() - start
            out[name] = median(forecasts).ravel()
            if name == "Moirai":
                out["Moirai-lo-80"] = quantile(forecasts, 0.1).ravel()
                out["Moirai-hi-80"] = quantile(forecasts, 0.9).ravel()
        rows.append(out)

    result = pd.concat(rows)
    result["unique_id"] = result["unique_id"].astype(int)
    result.round(2).to_csv(DATA / "fm_moirai_walmart.csv", index=False)
    pd.Series(timing, name="seconds").round(2).to_csv(DATA / "fm_moirai_timing.csv", index_label="model")
    smape = lambda f: np.mean(200 * np.abs(result["y"] - result[f]) / (result["y"].abs() + result[f].abs()))
    print({name: round(smape(name), 3) for name in runs})


if __name__ == "__main__":
    main()
