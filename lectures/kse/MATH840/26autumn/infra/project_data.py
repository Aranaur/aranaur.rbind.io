"""Build the final-project dataset: disguise the source, cut the hidden horizon, fake a weather forecast.

The source is the hourly generation-and-weather series kept on the site's datasets page. It is public,
so the copy students get is disguised the way the Track B pool is: the target is rescaled, the calendar
moves forward by exactly seven years (2557 days, so the hours stay continuous), and the columns get
English names.

The horizon is the last 48 hours. Students get the actual weather for the history and a *forecast* of
it for the horizon: the actual values plus noise that grows from the first day to the second, as a
real day-ahead forecast would. Without the noise the weather columns would be an oracle.

Outputs, all under grading/project/ (gitignored - this is the answer key until the leaderboard closes):
    moodle/train.csv              history, with actual weather          -> Moodle
    moodle/weather_forecast.csv   48 hours of forecast weather          -> Moodle
    truth.csv                     the hidden 48 hours, target and actual weather
    source.csv                    an untouched copy of the source

Usage:
    python project_data.py path/to/energy_weather_ua_2016_2019.csv
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "grading" / "project"

H = 48
SCALE = 1.37
SHIFT = pd.Timedelta(days=2557)        # 2016-01-01 -> 2023-01-01, two leap days in between
SEED = 840

NAMES = {
    "Data": "ds",
    "Heneratsia": "y",
    "Hmarnist": "cloud_cover",
    "Temperatura": "temperature",
    "Shvydkist_vity": "wind_speed",
    "Atm_tysk": "pressure",
    "Vologist_povitria": "humidity",
    "Soniachni_godunu": "daylight",    # 1 at 08-14 every day of the year: sun above the horizon
    "Schilnist_povitria": "air_density",
}

# Forecast error as a share of each column's own spread: 1.5 and 2.5 degrees for temperature, the
# same shares for the other continuous columns.
DAY_SHARE = (0.14, 0.23)
# Cloud cover is in notches of 0.25 and is one notch off with this probability instead. Daylight is
# astronomy, which a forecast gets exactly right, so it is left alone.
DAY_FLIP = (0.15, 0.25)


def weather_forecast(actual: pd.DataFrame, spread: pd.Series, rng: np.random.Generator) -> pd.DataFrame:
    fc = actual.copy()
    day = np.repeat([0, 1], H // 2)
    share = np.array(DAY_SHARE)[day]
    flip = rng.random(H) < np.array(DAY_FLIP)[day]

    for col, digits in (("temperature", 0), ("wind_speed", 0), ("pressure", 0), ("humidity", 0),
                        ("air_density", 4)):
        noisy = fc[col] + rng.normal(0, 1, H) * share * spread[col]
        fc[col] = noisy.round(digits)
    fc["wind_speed"] = fc["wind_speed"].clip(lower=0)
    fc["humidity"] = fc["humidity"].clip(0, 100)
    for col in ("temperature", "wind_speed", "pressure", "humidity"):
        fc[col] = fc[col].astype(int)

    step = rng.choice([-0.25, 0.25], H)
    fc["cloud_cover"] = np.where(flip, (fc["cloud_cover"] + step).clip(0, 1), fc["cloud_cover"])
    return fc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    args = ap.parse_args()

    (OUT / "moodle").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.source, OUT / "source.csv")

    d = pd.read_csv(args.source, parse_dates=["Data"]).rename(columns=NAMES)
    d["ds"] = d["ds"] + SHIFT
    d["y"] = (d["y"] * SCALE).round(1)
    assert d["ds"].diff().dropna().eq(pd.Timedelta(hours=1)).all(), "the source has gaps"

    train, test = d.iloc[:-H].copy(), d.iloc[-H:].copy()
    weather = [c for c in d.columns if c not in ("ds", "y")]
    spread = train[weather].std()

    fc = weather_forecast(test[["ds", *weather]].reset_index(drop=True), spread,
                          np.random.default_rng(SEED))

    fmt = "%Y-%m-%d %H:%M:%S"
    train.to_csv(OUT / "moodle" / "train.csv", index=False, date_format=fmt)
    fc.to_csv(OUT / "moodle" / "weather_forecast.csv", index=False, date_format=fmt)
    test.to_csv(OUT / "truth.csv", index=False, date_format=fmt)

    print(f"train: {len(train)} hours, {train['ds'].min()} to {train['ds'].max()}")
    print(f"horizon: {test['ds'].min()} to {test['ds'].max()}")
    print("forecast error on the horizon (mean absolute):")
    print((fc[weather] - test[weather].reset_index(drop=True)).abs().mean().round(3).to_string())


if __name__ == "__main__":
    main()
