"""Prophet vs NeuralProphet on Messi's Wikipedia page views: what does the AR-Net add?

Builds data/neuralprophet_messi.csv and data/neuralprophet_ar_weights.csv for the Week 9 lecture.

NeuralProphet 0.9.0 (June 2024, the last release) needs numpy < 2 and pandas < 3, so it cannot live in
the course .venv. Run this in its own environment:

    uv venv .venv-neuralprophet --python 3.12
    VIRTUAL_ENV=.venv-neuralprophet uv pip install torch --index-url https://download.pytorch.org/whl/cpu
    VIRTUAL_ENV=.venv-neuralprophet uv pip install neuralprophet==0.9.0 prophet "pandas==2.2.3"
    .venv-neuralprophet/Scripts/python infra/neuralprophet_messi.py      # about two minutes

Every model is fitted once, on the history to 30 September 2025, and scored on the following year.
All three get the same 0/1 tournament flag as a known future regressor. The models with lags use the
actual views up to the forecast origin at prediction time - that is what an autoregression does - but
none of them is refitted.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from neuralprophet import NeuralProphet, set_log_level
from prophet import Prophet

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
set_log_level("ERROR")

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CUTOFF = pd.Timestamp("2025-10-01")
LAGS, STEPS = 14, 7
TOURNAMENTS = [("2016-06-03", 23), ("2018-06-14", 31), ("2019-06-14", 23), ("2021-06-13", 27),
               ("2022-11-20", 28), ("2024-06-20", 24), ("2026-06-11", 38)]


def load() -> pd.DataFrame:
    messi = pd.read_csv(DATA / "messi_wikipedia.csv", parse_dates=["ds"])
    messi["y"] = np.log(messi["y"])
    messi["tournament"] = 0.0
    for start, days in TOURNAMENTS:
        inside = messi["ds"].between(pd.Timestamp(start), pd.Timestamp(start) + pd.Timedelta(days=days))
        messi.loc[inside, "tournament"] = 1.0
    return messi


def main() -> None:
    torch.manual_seed(1)
    np.random.seed(1)
    messi = load()
    train = messi[messi["ds"] < CUTOFF]
    out = messi[messi["ds"] >= CUTOFF][["ds", "y", "tournament"]].reset_index(drop=True)

    prophet = Prophet()
    prophet.add_regressor("tournament")
    prophet.fit(train)
    out["Prophet"] = prophet.predict(out[["ds", "tournament"]])["yhat"].to_numpy()

    plain = NeuralProphet(epochs=60, learning_rate=0.01)
    plain = plain.add_future_regressor("tournament")
    plain.fit(train[["ds", "y", "tournament"]], freq="D", progress=None)
    out["NeuralProphet"] = plain.predict(out[["ds", "y", "tournament"]])["yhat1"].to_numpy()

    torch.manual_seed(1)
    ar = NeuralProphet(n_lags=LAGS, n_forecasts=STEPS, epochs=60, learning_rate=0.01)
    ar = ar.add_future_regressor("tournament")
    ar.fit(train[["ds", "y", "tournament"]], freq="D", progress=None)
    # Predict over the test year with the last LAGS days of training in front, so day one has its lags.
    context = messi[messi["ds"] >= CUTOFF - pd.Timedelta(days=LAGS)][["ds", "y", "tournament"]]
    # Row ds, column yhat{k}: the forecast for ds made k days earlier.
    pred = ar.predict(context)
    pred = pred[["ds"] + [f"yhat{k}" for k in range(1, STEPS + 1)]]
    out = out.merge(pred.rename(columns={f"yhat{k}": f"AR-Net h={k}" for k in range(1, STEPS + 1)}),
                    on="ds", how="left")

    # The AR-Net here is a linear layer (no hidden layers): weights[step, lag].
    layer = [m for m in ar.model.ar_net.modules() if isinstance(m, torch.nn.Linear)][-1]
    weights = pd.DataFrame(layer.weight.detach().numpy(), index=[f"h={k}" for k in range(1, STEPS + 1)],
                           columns=[f"lag {j}" for j in range(1, LAGS + 1)])

    out.round(5).to_csv(DATA / "neuralprophet_messi.csv", index=False)
    weights.round(5).to_csv(DATA / "neuralprophet_ar_weights.csv")
    print(out.drop(columns="ds").sub(out["y"], axis=0).abs().mean().round(3))


if __name__ == "__main__":
    main()
