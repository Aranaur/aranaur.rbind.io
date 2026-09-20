"""Build the Track B series pool: one series per student, issued anonymised, known to be beatable.

What the students were promised in Week 1, and what this script therefore has to deliver:

  * one series per student, drawn by a hash of their own identifier, theirs for the whole course;
  * anonymised on issue - no name, values rescaled, calendar shifted, history trimmed;
  * a benchmark that is the *best of four simple methods*, chosen on a validation window;
  * a pool pre-screened so that a reference AutoETS/AutoARIMA lands between 0.70 and 0.90 of
    that benchmark. Below 0.70 the series is trivial, above 0.90 effectively unbeatable.

Each source series yields several **variants**, trimmed to end at different dates. A variant is a
different forecasting problem - its own hidden holdout, its own scale, its own dates - so two
students who happen to draw the same source cannot trade answers, and the pool is large enough
that drawing the same *variant* is rare. Screening is what decides which variants may be issued,
and it runs in two stages because AutoARIMA costs twenty times what AutoETS does: ETS on every
variant, ARIMA only where ETS left the outcome open.

The holdout never leaves this machine: `data/trackb/` gets the served history, and the future the
students are forecasting stays in `grading/` with the identities and the reference numbers.

    python trackb_pool.py --limit 12     # smoke test on a few sources
    python trackb_pool.py                # the real build, ~25 minutes

Rebuilds are deterministic: every random choice is seeded from the source key and the variant
number, so a second run issues the same series with the same disguise, and a student's data never
changes under them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
DATA = COURSE / "data"
PUBLIC = DATA / "trackb"          # served to students over raw.githubusercontent
PRIVATE = COURSE / "grading"      # gitignored: identities, holdout, reference scores

# Horizon, shortest acceptable served history, and how many variants to cut, by seasonal period.
HORIZON = {12: 12, 4: 8}
MIN_SERVED = {12: 180, 4: 56}
MAX_SERVED = {12: 264, 4: 72}     # 22 years of months, 18 of quarters: enough to model, quick to fit
VARIANTS = {12: 6, 4: 4}
BAND = (0.70, 0.90)               # the promise on the Week 1 slide
ETS_GATE = (0.70, 1.30)           # where it is still worth paying for AutoARIMA
METHODS = ("mean", "naive", "snaive", "drift")


# --------------------------------------------------------------------------- candidate sources
@dataclass
class Source:
    key: str                  # stable identity of the source series, never published
    source: str
    label: str
    freq: str
    m: int
    frame: pd.DataFrame = field(repr=False)   # columns ds, y - the full real history


def sources(limit: int | None = None) -> list[Source]:
    out: list[Source] = []

    retail = pd.read_csv(DATA / "aus_retail.csv", parse_dates=["Month"])
    for (state, industry), g in retail.groupby(["State", "Industry"], sort=True):
        s = (g[["Month", "Turnover"]].rename(columns={"Month": "ds", "Turnover": "y"})
             .dropna().sort_values("ds").reset_index(drop=True))
        if len(s) >= 300 and (s["y"] > 0).all():
            out.append(Source(f"retail|{state}|{industry}", "aus_retail",
                              f"{industry}, {state}", "MS", 12, s))

    emp = pd.read_csv(DATA / "us_employment_titles.csv", parse_dates=["ds"])
    last = emp["ds"].max()
    for uid, g in emp.groupby("unique_id", sort=True):
        s = g[["ds", "y"]].dropna().sort_values("ds").reset_index(drop=True)
        if len(s) >= 300 and (s["y"] > 0).all() and s["ds"].max() == last:
            # Stamped at month end in this file; the pool speaks month starts.
            s["ds"] = s["ds"].values.astype("datetime64[M]")
            out.append(Source(f"employment|{uid}", "us_employment", g["Title"].iloc[0], "MS", 12, s))

    tourism = pd.read_csv(DATA / "tourism.csv", parse_dates=["ds"])
    for (region, purpose), g in tourism.groupby(["Region", "Purpose"], sort=True):
        s = g.groupby("ds", as_index=False)["y"].sum().sort_values("ds").reset_index(drop=True)
        if len(s) == 80 and (s["y"] > 0).all() and s["y"].mean() >= 20:
            out.append(Source(f"tourism|{region}|{purpose}", "tourism",
                              f"{purpose} trips to {region}", "QS", 4, s))

    out.sort(key=lambda c: c.key)
    return out[:limit] if limit else out


# --------------------------------------------------------------------------- disguise
def rng_for(key: str, variant: int) -> np.random.Generator:
    """One stream per variant, so the disguise survives a rebuild."""
    seed = int(hashlib.sha256(f"MATH840|trackb|{key}|{variant}".encode()).hexdigest()[:16], 16)
    return np.random.default_rng(seed)


def code_for(key: str, variant: int) -> str:
    return "TS-" + hashlib.sha256(f"MATH840|code|{key}|{variant}".encode()).hexdigest()[:5].upper()


def disguise(src: Source, variant: int) -> tuple[pd.DataFrame, dict] | None:
    """Trim the history, rescale the values, shift the calendar by whole years.

    Whole years, not months: a within-year shift would break the link between the season and the
    calendar, and with it every legitimate calendar regressor the later weeks need. Shifting by
    years hides which years these are while January stays January.

    The variant number fixes how much is cut off the end, which is what gives each variant its own
    hidden holdout.
    """
    rng = rng_for(src.key, variant)
    s = src.frame.copy()
    m, h = src.m, HORIZON[src.m]

    drop_back = variant * (m if m == 12 else 2)
    if len(s) - drop_back < MIN_SERVED[m] + h:
        return None
    if drop_back:
        s = s.iloc[:-drop_back].reset_index(drop=True)

    # Then a whole number of cycles off the front, so the series starts at the same phase, and a
    # hard cap on what is left: a student does not need 70 years of history, and AutoARIMA on a
    # thousand observations costs twenty times what it costs on three hundred.
    room = len(s) - (MIN_SERVED[m] + h)
    cycles = int(rng.integers(0, min(room // m, 4) + 1))
    s = s.iloc[cycles * m:].reset_index(drop=True)
    if len(s) > MAX_SERVED[m] + h:
        s = s.iloc[-(MAX_SERVED[m] + h):].reset_index(drop=True)

    factor = float(np.round(10 ** rng.uniform(-1.0, 1.0) * rng.uniform(0.5, 2.0), 4))
    s["y"] = np.round(s["y"].to_numpy(float) * factor, 3)

    shift_years = int(rng.integers(4, 16)) * int(rng.choice([-1, 1]))
    s["ds"] = s["ds"] + pd.DateOffset(years=shift_years)

    return s, {"variant": variant, "drop_back": drop_back, "drop_front": cycles * m,
               "factor": factor, "shift_years": shift_years}


# --------------------------------------------------------------------------- benchmark and scoring
def benchmark_forecasts(y: np.ndarray, h: int, m: int) -> dict[str, np.ndarray]:
    """The four Week 1 methods, exactly as the students implemented them in Lab 2."""
    T = len(y)
    return {
        "mean": np.repeat(y.mean(), h),
        "naive": np.repeat(y[-1], h),
        "snaive": np.array([y[-m + (i % m)] for i in range(h)]),
        "drift": y[-1] + np.arange(1, h + 1) * (y[-1] - y[0]) / (T - 1),
    }


def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def mase(actual: np.ndarray, forecast: np.ndarray, served: np.ndarray, m: int) -> float:
    """MASE scaled by the in-sample seasonal naive error of the *served* history.

    The same scale for every model on this series, and the same one the scorer uses, so a student
    can reproduce their own number on their own validation window before submitting.
    """
    scale = float(np.mean(np.abs(served[m:] - served[:-m])))
    return float(np.mean(np.abs(actual - forecast)) / scale)


def choose_benchmark(served: np.ndarray, m: int, h: int) -> tuple[str, dict[str, float]]:
    """Pick the best of the four on a validation window - the h observations before the holdout."""
    train, valid = served[:-h], served[-h:]
    errors = {name: rmse(valid, fc) for name, fc in benchmark_forecasts(train, h, m).items()}
    return min(errors, key=errors.get), errors


def reference(served: np.ndarray, holdout: np.ndarray, m: int, h: int, which: str) -> float:
    """MASE of AutoETS or AutoARIMA fitted on the served history, scored on the hidden holdout."""
    from statsforecast.models import AutoARIMA, AutoETS

    model = AutoETS(season_length=m) if which == "ets" else AutoARIMA(season_length=m)
    try:
        forecast = np.asarray(model.fit(y=served).predict(h=h)["mean"], dtype=float)
    except Exception:
        return float("nan")
    if not np.isfinite(forecast).all():
        return float("nan")
    return mase(holdout, forecast, served, m)


def screen(src: Source, variant: int) -> dict | None:
    """Disguise, split, benchmark, reference - and decide whether this variant may be issued."""
    m, h = src.m, HORIZON[src.m]
    made = disguise(src, variant)
    if made is None:
        return None
    frame, info = made

    y = frame["y"].to_numpy(float)
    served, holdout = y[:-h], y[-h:]
    if len(served) < MIN_SERVED[m] or not np.isfinite(y).all() or (y <= 0).any():
        return None

    name, valid_errors = choose_benchmark(served, m, h)
    bench_fc = benchmark_forecasts(served, h, m)[name]
    bench_mase = mase(holdout, bench_fc, served, m)
    if not np.isfinite(bench_mase) or bench_mase <= 0:
        return None

    ets = reference(served, holdout, m, h, "ets")
    if not np.isfinite(ets):
        return None
    ets_skill = ets / bench_mase

    # AutoARIMA can only improve the reference, so it cannot rescue a series that is already
    # trivial - and it cannot save one where ETS is hopeless. Pay for it only in between.
    arima = float("nan")
    if ETS_GATE[0] <= ets_skill <= ETS_GATE[1]:
        arima = reference(served, holdout, m, h, "arima")

    best = min([v for v in (ets, arima) if np.isfinite(v)])
    skill = best / bench_mase

    return {
        "code": code_for(src.key, variant),
        "key": src.key,
        "source": src.source,
        "label": src.label,
        "freq": src.freq,
        "m": m,
        "h": h,
        "n_served": len(served),
        "served_start": str(frame["ds"].iloc[0].date()),
        "served_end": str(frame["ds"].iloc[len(served) - 1].date()),
        "benchmark": name,
        "benchmark_mase": round(bench_mase, 4),
        "benchmark_rmse": round(rmse(holdout, bench_fc), 4),
        "ref_mase_ets": round(ets, 4),
        "ref_mase_arima": round(arima, 4) if np.isfinite(arima) else None,
        "ref_best": "AutoARIMA" if np.isfinite(arima) and arima < ets else "AutoETS",
        "skill": round(skill, 4),
        "issued": bool(BAND[0] <= skill <= BAND[1]),
        **info,
        "holdout": [round(float(v), 3) for v in holdout],
        "valid_rmse": {k: round(v, 4) for k, v in valid_errors.items()},
        "_frame": frame,
        "_served_len": len(served),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="screen only the first N sources")
    args = ap.parse_args()

    pool = sources(args.limit)
    print(f"{len(pool)} source series, up to {max(VARIANTS.values())} variants each")

    rows, started = [], time.time()
    for i, src in enumerate(pool, 1):
        for variant in range(VARIANTS[src.m]):
            row = screen(src, variant)
            if row is not None:
                rows.append(row)
        if i % 20 == 0 or args.limit:
            issued = sum(r["issued"] for r in rows)
            print(f"  [{i}/{len(pool)}] screened {len(rows)} variants, issued {issued}, "
                  f"{time.time() - started:.0f}s")

    issued = [r for r in rows if r["issued"]]
    print(f"\nscreened {len(rows)} variants | issued {len(issued)} | band {BAND[0]}-{BAND[1]} "
          f"| {time.time() - started:.0f}s")
    if not issued:
        return

    PUBLIC.mkdir(parents=True, exist_ok=True)
    for old in PUBLIC.glob("*.csv"):
        old.unlink()
    for r in issued:
        served = r["_frame"].iloc[: r["_served_len"]][["ds", "y"]].copy()
        served["ds"] = served["ds"].dt.strftime("%Y-%m-%d")
        served.to_csv(PUBLIC / f"{r['code']}.csv", index=False)

    index = pd.DataFrame([{"code": r["code"], "freq": r["freq"], "m": r["m"], "h": r["h"],
                           "n": r["n_served"], "start": r["served_start"], "end": r["served_end"]}
                          for r in issued]).sort_values("code")
    index.to_csv(PUBLIC / "index.csv", index=False)

    PRIVATE.mkdir(parents=True, exist_ok=True)
    secret = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (PRIVATE / "trackb_pool.json").write_text(
        json.dumps(secret, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    skills = np.array([r["skill"] for r in issued])
    print("issued by source:\n" + pd.Series([r["source"] for r in issued]).value_counts().to_string())
    print("issued by seasonal period:\n" + pd.Series([r["m"] for r in issued]).value_counts().to_string())
    print(f"benchmark chosen: {pd.Series([r['benchmark'] for r in issued]).value_counts().to_dict()}")
    print(f"reference model:  {pd.Series([r['ref_best'] for r in issued]).value_counts().to_dict()}")
    print(f"skill: min {skills.min():.2f} median {np.median(skills):.2f} max {skills.max():.2f}")
    print(f"sources represented: {len({r['key'] for r in issued})} of {len(pool)}")
    print(f"\npublic:  {PUBLIC} ({len(issued)} series + index.csv)")
    print(f"private: {PRIVATE / 'trackb_pool.json'}")
    print(f"expected duplicate draws for 39 students: {39 * 38 / 2 / len(issued):.1f} pairs")


if __name__ == "__main__":
    main()
