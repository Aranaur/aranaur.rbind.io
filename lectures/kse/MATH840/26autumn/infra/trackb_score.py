"""Score Track B submissions against the hidden holdout.

    python trackb_score.py ../grading/submissions/ch1
    python trackb_score.py ../grading/submissions/ch1 --out ../grading/ch1_scores.csv

One row per submission: the MASE of the forecast, the MASE of the student's own benchmark, the skill
ratio between them, and the accuracy points that follow. Everything it reads and everything it writes
lives under `grading/`, which is gitignored - the holdout is the whole point of the exercise.

The benchmark is recomputed here from the *public* series, by the same rule the template uses, so a
student can reproduce their benchmark exactly; only the holdout is ours. Format errors are reported,
never repaired: a file that cannot be scored earns 0 for accuracy and says why.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trackb_pool import benchmark_forecasts, choose_benchmark, mase

HERE = Path(__file__).resolve().parent
COURSE = HERE.parent
PUBLIC = COURSE / "data" / "trackb"
POOL = COURSE / "grading" / "trackb_pool.json"

COLUMNS = ["student_id", "code", "ds", "yhat", "yhat_lo_80", "yhat_hi_80"]


def points_for(skill: float) -> int:
    """The thresholds promised in Week 1: beat it, come within 10%, or explain yourself."""
    if skill <= 1.0:
        return 3
    if skill <= 1.1:
        return 2
    return 1


def score_one(path: Path, pool: dict[str, dict]) -> dict:
    row: dict = {"file": path.name, "student_id": None, "code": None, "points": 0, "note": ""}
    try:
        sub = pd.read_csv(path)
    except Exception as err:
        row["note"] = f"unreadable: {type(err).__name__}"
        return row

    missing = [c for c in COLUMNS if c not in sub.columns]
    if missing:
        row["note"] = f"missing columns: {', '.join(missing)}"
        return row

    row["student_id"] = str(sub["student_id"].iloc[0])
    code = str(sub["code"].iloc[0])
    row["code"] = code
    if sub["code"].nunique() != 1:
        row["note"] = "more than one series code in the file"
        return row
    if code not in pool:
        row["note"] = "series code is not in the pool"
        return row

    entry = pool[code]
    h, m = entry["h"], entry["m"]
    served = pd.read_csv(PUBLIC / f"{code}.csv", parse_dates=["ds"])
    expected = pd.date_range(served["ds"].iloc[-1], periods=h + 1,
                             freq="MS" if m == 12 else "QS")[1:]

    if len(sub) != h:
        row["note"] = f"expected {h} rows, found {len(sub)}"
        return row
    got = pd.to_datetime(sub["ds"], errors="coerce")
    if got.isna().any() or not (got.to_numpy() == expected.to_numpy()).all():
        row["note"] = f"dates must be {expected[0].date()} to {expected[-1].date()}, monthly/quarterly in order"
        return row
    if sub[["yhat", "yhat_lo_80", "yhat_hi_80"]].isna().to_numpy().any():
        row["note"] = "NaN in the forecast columns"
        return row

    y = served["y"].to_numpy(float)
    holdout = np.asarray(entry["holdout"], dtype=float)
    yhat = sub["yhat"].to_numpy(float)

    # The student's benchmark, recomputed from public data exactly as the template does.
    name, _ = choose_benchmark(y, m, h)
    bench = benchmark_forecasts(y, h, m)[name]

    student_mase = mase(holdout, yhat, y, m)
    bench_mase = mase(holdout, bench, y, m)
    skill = student_mase / bench_mase
    inside = ((holdout >= sub["yhat_lo_80"].to_numpy(float))
              & (holdout <= sub["yhat_hi_80"].to_numpy(float))).mean()

    row.update({
        "benchmark": name,
        "mase": round(student_mase, 4),
        "benchmark_mase": round(bench_mase, 4),
        "skill": round(skill, 4),
        "points": points_for(skill),
        "coverage_80": round(float(inside), 2),
        "reference_skill": entry["skill"],
        "note": "beat the benchmark" if skill <= 1 else
                ("within 10%" if skill <= 1.1 else "worse than the benchmark - explanation required"),
    })
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("submissions", help="directory of SURNAME_ch1_forecast.csv files")
    ap.add_argument("--out", help="write the scores to this CSV as well")
    args = ap.parse_args()

    if not POOL.exists():
        raise SystemExit(f"{POOL} not found - run trackb_pool.py first")
    pool = {entry["code"]: entry for entry in json.loads(POOL.read_text(encoding="utf-8"))
            if entry["issued"]}

    files = sorted(Path(args.submissions).glob("*.csv"))
    if not files:
        raise SystemExit(f"no CSV files in {args.submissions}")

    scores = pd.DataFrame([score_one(f, pool) for f in files])
    ordered = [c for c in ["file", "student_id", "code", "benchmark", "mase", "benchmark_mase",
                           "skill", "points", "coverage_80", "reference_skill", "note"]
               if c in scores.columns]
    scores = scores[ordered]
    print(scores.to_string(index=False))

    scored = scores[scores["points"] > 0]
    print(f"\n{len(scored)} of {len(scores)} scored | "
          f"points {dict(scores['points'].value_counts().sort_index())}")
    if len(scored):
        print(f"median skill {scored['skill'].median():.2f} | "
              f"beat the benchmark: {(scored['skill'] <= 1).sum()} | "
              f"mean 80% coverage {scored['coverage_80'].mean():.2f}")

    # Two students on the same series is possible - the draw is a hash, not an allocation.
    shared = scores[scores["code"].notna()]["code"].value_counts()
    shared = shared[shared > 1]
    if len(shared):
        print("\nsame series drawn by more than one student (check the forecasts differ):")
        for code, n in shared.items():
            who = ", ".join(scores.loc[scores["code"] == code, "student_id"].astype(str))
            print(f"  {code}: {n} submissions - {who}")

    if args.out:
        scores.to_csv(args.out, index=False)
        print(f"\nwritten: {args.out}")


if __name__ == "__main__":
    main()
