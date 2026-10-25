"""TimeGPT on the Walmart stores of the foundation-models lecture.

Builds data/fm_timegpt_walmart.csv, data/fm_timegpt_shap.csv, data/fm_timegpt_anomalies.csv and
data/fm_timegpt_timing.csv for the Week 10 deck.

TimeGPT runs on Nixtla's servers, so it needs an API key (see labs/.timegpt-setup.qmd). The key is read
from the environment variable NIXTLA_API_KEY, or from a line NIXTLA_API_KEY=... in the .env file at the
repository root, which is gitignored. It is never printed and never written anywhere else.

    python infra/timegpt_walmart.py        # six API calls, under a minute

The protocol matches infra/moirai_walmart.py and the deck: stores 1-4, the last 32 weeks cut into four
windows of 8, the holiday flag as the one exogenous variable. Fine-tuning settings are the book's
(Peixeiro 2026, ch. 3): 10 steps, MAE loss, depth 1 and 2.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from nixtla import NixtlaClient

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
ROOT = HERE.parents[4]
STORES, H, WINDOWS = [1, 2, 3, 4], 8, 4


def api_key() -> str:
    if os.environ.get("NIXTLA_API_KEY"):
        return os.environ["NIXTLA_API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("NIXTLA_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("No NIXTLA_API_KEY in the environment or in .env at the repository root.")


def main() -> None:
    client = NixtlaClient(api_key=api_key())
    if not client.validate_api_key():
        raise SystemExit("The API key was rejected by Nixtla.")

    w = pd.read_csv(DATA / "walmart.csv", parse_dates=["ds"])
    w = w[w["unique_id"].isin(STORES)].sort_values(["unique_id", "ds"]).reset_index(drop=True)
    target = w[["unique_id", "ds", "y"]]
    flagged = w[["unique_id", "ds", "y", "holiday"]]
    cv = dict(h=H, n_windows=WINDOWS, step_size=H, freq="W-FRI", time_col="ds", target_col="y", id_col="unique_id")

    runs = {
        "TimeGPT": dict(df=target, level=[80]),
        "TimeGPT fine-tuned": dict(df=target, finetune_steps=10, finetune_loss="mae"),
        "TimeGPT fine-tuned, depth 2": dict(df=target, finetune_steps=10, finetune_loss="mae", finetune_depth=2),
        "TimeGPT + holiday": dict(df=flagged),
    }
    result, timing = None, {}
    for name, kwargs in runs.items():
        start = time.time()
        out = client.cross_validation(**kwargs, **cv)
        timing[name] = time.time() - start
        keep = out[["unique_id", "ds", "cutoff", "y", "TimeGPT"]].rename(columns={"TimeGPT": name})
        if name == "TimeGPT":
            keep["TimeGPT-lo-80"] = out["TimeGPT-lo-80"].to_numpy()
            keep["TimeGPT-hi-80"] = out["TimeGPT-hi-80"].to_numpy()
        result = keep if result is None else result.merge(keep.drop(columns="y"), on=["unique_id", "ds", "cutoff"])

    # Shapley values of the holiday flag on the last window
    cutoff = np.sort(w["ds"].unique())[-H - 1]
    history = flagged[flagged["ds"] <= cutoff]
    future = flagged[flagged["ds"] > cutoff].drop(columns="y")
    client.forecast(df=history, X_df=future, h=H, freq="W-FRI", time_col="ds", target_col="y", id_col="unique_id",
                    feature_contributions=True)
    shap = client.feature_contributions

    # Anomalies: in-sample forecasts with a 99% interval; outside means anomalous. The client's default
    # model (timegpt-2.1, nixtla 0.9.0) returned a constant in-sample forecast here - every week flagged -
    # so this uses timegpt-1, the model of the book. Forecasts and cross-validation use the default.
    anomalies = client.detect_anomalies(df=target, freq="W-FRI", time_col="ds", target_col="y", id_col="unique_id",
                                        level=99, model="timegpt-1")

    result.round(2).to_csv(DATA / "fm_timegpt_walmart.csv", index=False)
    shap.round(4).to_csv(DATA / "fm_timegpt_shap.csv", index=False)
    anomalies.round(2).to_csv(DATA / "fm_timegpt_anomalies.csv", index=False)
    pd.Series(timing, name="seconds").round(2).to_csv(DATA / "fm_timegpt_timing.csv", index_label="model")
    smape = lambda f: np.mean(200 * np.abs(result["y"] - result[f]) / (result["y"].abs() + result[f].abs()))
    print({name: round(smape(name), 3) for name in runs})
    print("anomalies flagged:", int(anomalies["anomaly"].sum()), "of", len(anomalies))


if __name__ == "__main__":
    main()
