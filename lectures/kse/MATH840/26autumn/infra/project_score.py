"""Score the final-project submissions and build the leaderboards.

Each stage is one download from Moodle ("Download all submissions"):

    session1   the Session 1 activity, after 11:30 on 3 November
    update1    the Final activity, 4 November 09:00 - public leaderboard
    update2    the Final activity, 5 November 09:00 - public leaderboard
    final      the Final activity, after 23:59 on 5 November (deadline moved from 21:00 for power outages)

    python project_score.py session1 path/to/download.zip
    python project_score.py update1  path/to/download.zip
    python project_score.py final    path/to/download.zip

The zip is unpacked into grading/project/submissions/<stage>/ and every CSV in it is scored on the
hidden 48 hours, the public first 24 and the private last 24. Everything lands in grading/project/board/:

    <stage>.csv        one row per file: who, validity, every metric on every part
    <stage>_forecasts.csv, snaive.csv   the forecasts themselves, for the Session 2 dashboard
    public.csv         the public leaderboard: the latest valid file per student, first 24 hours only,
                       with the move since the previous update and the RMSE change against Session 1
    public_<stage>.csv the same, as it stood after each stage
    private.csv        (final) the private leaderboard: the better of Session 1 and final, last 24 hours
    points.csv         (final) the accuracy points, 18 of the 32
    repos.csv          (final) the repository links from the online text

public.csv is the only file the public dashboard reads. Nothing else may leave grading/ before Session 2.

The checks are looser than the template's check_submission: a file is scored whenever all 48 hours can
be read from it. Semicolons, decimal commas, another column order, a lower bound above the forecast - all
scored, and listed as notes so the instructor sees them.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
# MATH840_PROJECT_DIR points the scorer at a copy, for a dry run on made-up submissions.
PROJECT = Path(os.environ.get("MATH840_PROJECT_DIR", HERE.parent / "grading" / "project"))

H, M, ALPHA = 48, 24, 0.05
STAGES = ("session1", "update1", "update2", "final")
COLUMNS = ["ds", "forecast", "lower_95", "upper_95"]
BENCHMARK = "----SNAIVE----"
WEIGHTS = {"RMSE": 0.3, "MAPE": 0.3, "Winkler": 0.4}
PARTS = {"all": slice(0, H), "public": slice(0, H // 2), "private": slice(H // 2, H)}


# --- metrics -----------------------------------------------------------------------------------------

def metrics(actual, forecast, lower, upper) -> dict:
    err = actual - forecast
    miss = np.clip(lower - actual, 0, None) + np.clip(actual - upper, 0, None)
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(np.mean(np.abs(err / actual)) * 100),
        "Winkler": float(np.mean(upper - lower + 2 / ALPHA * miss)),
        "coverage": float(np.mean((actual >= lower) & (actual <= upper))),
    }


def snaive(history: np.ndarray) -> pd.DataFrame:
    # The template's benchmark, to the digit: the last day twice, with the seasonal-naive interval.
    point = np.array([history[-M + (i % M)] for i in range(H)])
    sigma = np.std(history[M:] - history[:-M], ddof=1)
    half = 1.96 * sigma * np.sqrt(np.arange(H) // M + 1)
    return pd.DataFrame({"forecast": point, "lower_95": np.clip(point - half, 0, None),
                         "upper_95": point + half})


def score_frame(fc: pd.DataFrame, actual: np.ndarray) -> dict:
    row = {}
    for part, cut in PARTS.items():
        m = metrics(actual[cut], *(fc[c].to_numpy(float)[cut] for c in ("forecast", "lower_95", "upper_95")))
        row.update({f"{part}_{k}": v for k, v in m.items()})
    return row


def composite(board: pd.DataFrame, part: str) -> pd.Series:
    # 0.3 RMSE rank + 0.3 MAPE rank + 0.4 Winkler rank; ties share the better rank.
    return sum(w * board[f"{part}_{k}"].rank(method="min") for k, w in WEIGHTS.items())


# --- reading what Moodle gives -------------------------------------------------------------------------

def owner(path: Path, root: Path) -> str:
    # "Ivan Petrenko_123456_assignsubmission_file/x.csv" or "..._assignsubmission_file_x.csv" -> "Ivan Petrenko"
    part = path.relative_to(root).parts[0]
    found = re.match(r"(.+?)_\d+_assignsubmission_", part)
    return found.group(1).strip() if found else path.stem


FORECAST_NAMES = {"forecast", "mean", "yhat", "y_hat", "pred", "prediction", "predicted", "y_pred",
                  "point", "point_forecast", "fc", "fcst", "value", "y"}
TIME_NAMES = {"ds", "date", "datetime", "time", "timestamp", "hour", "data"}


def resolve_columns(raw: pd.DataFrame) -> dict | None:
    """Map the student's column names onto COLUMNS: {} when they already match, None when hopeless.

    The names are the student's choice of library, not of forecast: StatsForecast writes `mean, lo-95,
    hi-95` or `AutoARIMA, AutoARIMA-lo-95, AutoARIMA-hi-95`, Prophet `yhat, yhat_lower, yhat_upper`. So a
    wrong name is fixed, not failed, and the note records what was renamed.
    """
    cols = [c for c in raw.columns if c not in ("unique_id", "unnamed:_0", "")]
    if all(c in cols for c in COLUMNS):
        return {}
    flat = {c: re.sub(r"[^a-z0-9]", "", c) for c in cols}
    found = {}

    time = [c for c in cols if c in TIME_NAMES or flat[c] in TIME_NAMES]
    found["ds"] = "ds" if "ds" in cols else (time[0] if time else cols[0])

    def bound(words):
        hits = [c for c in cols if re.search(rf"(^|[^a-z])({words})", c) and c != found["ds"]]
        with95 = [c for c in hits if "95" in c]
        return (with95 or hits or [None])[0]
    found["lower_95"] = "lower_95" if "lower_95" in cols else bound("lo|low|lower|min")
    found["upper_95"] = "upper_95" if "upper_95" in cols else bound("hi|high|up|upper|max")

    point = [c for c in cols if c == "forecast" or flat[c] in FORECAST_NAMES]
    if not point and found["lower_95"]:
        # StatsForecast names the point column after the model and the bounds after it: "x" and "x-lo-95".
        stem = re.split(r"[-_]?(lo|low|lower)", found["lower_95"])[0]
        point = [c for c in cols if c == stem]
    rest = [c for c in cols if c not in (found["ds"], found["lower_95"], found["upper_95"])]
    found["forecast"] = point[0] if point else (rest[0] if len(rest) == 1 else None)

    if None in found.values() or len(set(found.values())) < 4:
        # Last resort: four columns in the template's order, whatever they are called.
        if len(cols) != 4:
            return None
        found = dict(zip(COLUMNS, cols))
    return {src: dst for dst, src in found.items() if src != dst}


def read_submission(path: Path, hours: pd.Series) -> tuple[pd.DataFrame | None, list[str]]:
    notes = []
    try:
        raw = pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception as exc:                                   # noqa: BLE001 - any unreadable file
        return None, [f"unreadable: {exc.__class__.__name__}"]

    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]
    renamed = resolve_columns(raw)
    if renamed is None:
        return None, [f"missing columns {[c for c in COLUMNS if c not in raw.columns]}; has {list(raw.columns)}"]
    if renamed:
        raw = raw.rename(columns=renamed)
        notes.append("columns renamed: " + ", ".join(f"{a}->{b}" for a, b in renamed.items()))
    elif list(raw.columns[:4]) != COLUMNS:
        notes.append("column order")

    fc = raw[COLUMNS].copy()
    fc["ds"] = pd.to_datetime(fc["ds"].str.strip(), errors="coerce")
    for col in COLUMNS[1:]:
        text = fc[col].str.strip()
        if text.str.contains(",", na=False).any():
            text = text.str.replace(",", ".", regex=False)
            if "decimal commas" not in notes:
                notes.append("decimal commas")
        fc[col] = pd.to_numeric(text, errors="coerce")

    if fc["ds"].isna().any():
        return None, notes + ["unreadable hours in ds"]
    if fc["ds"].duplicated().any():
        return None, notes + ["repeated hours"]
    fc = fc.set_index("ds").reindex(hours)
    if fc.isna().any().any():
        absent = int(fc.isna().any(axis=1).sum())
        return None, notes + [f"{absent} of {H} hours missing or empty"]

    fc = fc.reset_index(drop=True)
    if len(raw) != H:
        notes.append(f"{len(raw)} rows, scored on the {H} expected")
    if (fc["lower_95"] < 0).any():
        notes.append("negative lower bound")
    if ((fc["lower_95"] > fc["forecast"]) | (fc["forecast"] > fc["upper_95"])).any():
        notes.append("forecast outside its interval")
    return fc, notes


def repo_links(folder: Path) -> pd.DataFrame:
    # The link belongs in the online text (onlinetext.html), but some students upload it as a file instead:
    # repo.txt, README.md. Every text file is searched; the online text wins when both have a link.
    found = {}
    texts = sorted((t for t in folder.rglob("*") if t.is_file() and t.suffix.lower() in (".html", ".txt", ".md")),
                   key=lambda t: "onlinetext" not in t.name)
    for t in texts:
        link = re.search(r"https?://github\.com/[^\s\"'<>]+", html.unescape(t.read_text(errors="ignore")))
        who = owner(t, folder)
        if link and who not in found:
            found[who] = link.group(0).rstrip(".,)")
        found.setdefault(who, "")
    return pd.DataFrame(sorted(found.items()), columns=["student", "repo"])


# --- stages --------------------------------------------------------------------------------------------

def unpack(stage: str, source: Path) -> Path:
    dest = PROJECT / "submissions" / stage
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as z:
            for info in z.infolist():
                path = Path(z.extract(info, dest))
                # Keep the upload time from the archive: it decides between two CSVs of one student.
                stamp = datetime(*info.date_time).timestamp()
                os.utime(path, (stamp, stamp))
    else:
        shutil.copytree(source, dest, dirs_exist_ok=True)
    apply_exceptions(stage, dest)
    return dest


def apply_exceptions(stage: str, dest: Path) -> None:
    """Swap in files the instructor accepted by hand, kept in grading/project/exceptions/<stage>/.

    Each entry is a folder named like Moodle's, "<Name>_<id>_assignsubmission_file", holding the file that
    replaces whatever that student uploaded at this stage. Kept apart from the archives so that the
    decision stays visible and every re-run of the scorer repeats it.
    """
    extra = PROJECT / "exceptions" / stage
    if not extra.exists():
        return
    for folder in sorted(p for p in extra.iterdir() if p.is_dir()):
        who = owner(folder / "x", extra)
        for old in [p for p in dest.iterdir() if owner(p / "x" if p.is_dir() else p, dest) == who]:
            shutil.rmtree(old) if old.is_dir() else old.unlink()
        shutil.copytree(folder, dest / folder.name)
        print(f"  exception: {who}'s {stage} file replaced by {', '.join(f.name for f in folder.iterdir())}")


def score_stage(stage: str, folder: Path, truth: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actual = truth["y"].to_numpy(float)
    rows, forecasts = [], []
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".csv", ".txt"):
            continue
        fc, notes = read_submission(path, truth["ds"])
        row = {"stage": stage, "student": owner(path, folder), "file": path.name,
               "modified": datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
               "valid": fc is not None, "notes": "; ".join(notes)}
        if fc is not None:
            row.update(score_frame(fc, actual))
            forecasts.append(fc.assign(stage=stage, student=row["student"], file=path.name,
                                       ds=truth["ds"].to_numpy()))
        rows.append(row)

    board = pd.DataFrame(rows)
    if board.empty:
        raise SystemExit(f"no CSV files in {folder}")
    # A student who uploaded a new CSV next to the old one instead of replacing it has two: the newest
    # valid one counts.
    # Only valid files count here: a repo link saved as repo.txt next to the CSV is not a second forecast.
    ok = board[board["valid"]]
    many = ok[ok.duplicated("student", keep=False)]
    board = (board.sort_values(["student", "valid", "modified"], ascending=[True, False, False])
             .drop_duplicates("student", keep="first").reset_index(drop=True))
    for who in many["student"].unique():
        kept = board.set_index("student").loc[who]
        print(f"  ! {who}: {int((many['student'] == who).sum())} valid forecast files, scoring {kept['file']} "
              f"({kept['modified']})")
    fcs = pd.concat(forecasts, ignore_index=True) if forecasts else pd.DataFrame()
    if not fcs.empty:
        fcs = fcs.merge(board[["student", "file"]], on=["student", "file"]).drop(columns="file")
    return board, fcs


def benchmark_row(train: pd.DataFrame, truth: pd.DataFrame) -> dict:
    row = {"student": BENCHMARK, "file": "", "valid": True, "notes": ""}
    row.update(score_frame(snaive(train["y"].to_numpy(float)), truth["y"].to_numpy(float)))
    return row


def leaderboard(entries: pd.DataFrame, part: str) -> pd.DataFrame:
    cols = [f"{part}_{k}" for k in ("RMSE", "MAPE", "Winkler", "coverage")]
    board = entries[["student", *cols]].copy()
    board["score"] = composite(board, part)
    board = board.sort_values(["score", f"{part}_RMSE"]).reset_index(drop=True)
    board.insert(0, "place", board["score"].rank(method="min").astype(int))
    return board.rename(columns=lambda c: c.replace(f"{part}_", ""))


def with_changes(public: pd.DataFrame, stage: str, out: Path) -> pd.DataFrame:
    # Two columns for reference, both on the public 24 hours only: the move since the previous snapshot,
    # and the RMSE against the student's own Session 1 file. Neither affects any points.
    earlier = [s for s in STAGES[:STAGES.index(stage)] if (out / f"public_{s}.csv").exists()]
    if earlier:
        before = pd.read_csv(out / f"public_{earlier[-1]}.csv").set_index("student")["place"]
        move = public["student"].map(before) - public["place"]
        public["since_last"] = [
            "" if who == BENCHMARK else "new" if pd.isna(m) else "=" if m == 0 else f"↑{m:.0f}" if m > 0
            else f"↓{-m:.0f}" for who, m in zip(public["student"], move)]
    else:
        public["since_last"] = ""
    s1 = out / "session1.csv"
    if s1.exists():
        first = pd.read_csv(s1).query("valid").set_index("student")["public_RMSE"]
        public["vs_session1"] = (public["RMSE"] / public["student"].map(first) - 1) * 100
        # Blank, not +0%, for whoever is still on the Session 1 file: there is nothing to compare yet.
        later = [out / f"{s}.csv" for s in STAGES[1:STAGES.index(stage) + 1] if (out / f"{s}.csv").exists()]
        updated = set().union(*(pd.read_csv(f).query("valid")["student"] for f in later)) if later else set()
        public.loc[~public["student"].isin(updated), "vs_session1"] = float("nan")
    else:
        public["vs_session1"] = float("nan")
    return public


def latest_valid(stages: list[pd.DataFrame]) -> pd.DataFrame:
    # Each student's most recent valid file across the stages given, oldest first.
    pool = pd.concat([s[s["valid"]] for s in stages], ignore_index=True)
    return pool.drop_duplicates("student", keep="last").reset_index(drop=True)


def points(s1: pd.DataFrame, final: pd.DataFrame, bench: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    students = sorted(set(s1["student"]) | set(final["student"]))
    pts = pd.DataFrame({"student": students}).set_index("student")

    ok1 = s1[s1["valid"]].set_index("student")
    pts["participation"] = [5 if s in ok1.index else 0 for s in students]
    skill = ok1["all_RMSE"] / bench["all_RMSE"]
    pts["session1_skill"] = skill.reindex(students)
    # A copy of the benchmark comes back as 1 give or take the rounding of a CSV. The template submitted
    # unchanged is not a near miss: it gets the points for not beating the benchmark, so 5 is for 1 < k <= 1.1.
    pts["session1_skill"] = pts["session1_skill"].round(6)
    pts["benchmark"] = [0 if np.isnan(k) else 8 if k < 1 else 5 if 1 < k <= 1.1 else 2
                        for k in pts["session1_skill"]]

    # A file beats the benchmark on the private 24 hours if it has the lower RMSE OR the better combined rank
    # (decided 5 Nov: the handout named neither, and each reading alone shut out someone who beat it on the
    # other). The combined rank is compared in two pools, all candidate files and the private board as shown;
    # beating the benchmark in either counts.
    def beats(pool: pd.DataFrame) -> pd.Series:
        pool = pd.concat([pool, pd.DataFrame([bench])], ignore_index=True)
        score = composite(pool, "private")
        won = (pool["private_RMSE"] < bench["private_RMSE"]) | (score < score[pool["student"] == BENCHMARK].iloc[0])
        return won[pool["student"] != BENCHMARK].set_axis(pool.index[pool["student"] != BENCHMARK])

    # The better of the two files: one that beats the benchmark first, then the lower composite among
    # everyone's candidates.
    cands = pd.concat([s1[s1["valid"]], final[final["valid"]]], ignore_index=True)
    cands["score"] = composite(cands, "private")
    cands["beats"] = beats(cands)
    best = (cands.sort_values(["beats", "score", "private_RMSE"], ascending=[False, True, True])
                 .drop_duplicates("student").set_index("student"))
    best["beats"] = best["beats"] | beats(best.reset_index()).set_axis(best.index)
    pts["ranked_file"] = best["stage"].reindex(students)
    pts["private_RMSE"] = best["private_RMSE"].reindex(students)
    pts["beats_benchmark"] = best["beats"].reindex(students)

    beat = best[best["beats"]].copy()
    beat["score"] = composite(beat, "private")
    order = beat.sort_values(["score", "private_RMSE"]).index.to_list()
    thirds = np.array_split(np.array(order, dtype=object), 3) if order else []
    tier = {s: p for chunk, p in zip(thirds, (5, 4, 3)) for s in chunk}
    # Equal scores split by a third's edge share the better tier, as ties share the better rank in composite().
    level = beat["score"].round(6)
    tier = {s: max(tier[o] for o in order if level[o] == level[s]) for s in order}
    pts["ranking"] = [tier.get(s, 1 if s in best.index else 0) for s in students]

    pts["accuracy_total"] = pts[["participation", "benchmark", "ranking"]].sum(axis=1)
    return pts.reset_index(), best.reset_index()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("source", type=Path, help="the zip from Moodle, or a folder")
    args = ap.parse_args()

    truth = pd.read_csv(PROJECT / "truth.csv", parse_dates=["ds"])
    train = pd.read_csv(PROJECT / "moodle" / "train.csv", parse_dates=["ds"])
    out = PROJECT / "board"
    out.mkdir(exist_ok=True)

    folder = unpack(args.stage, args.source)
    board, fcs = score_stage(args.stage, folder, truth)
    board.to_csv(out / f"{args.stage}.csv", index=False)
    fcs.to_csv(out / f"{args.stage}_forecasts.csv", index=False)
    bench = benchmark_row(train, truth)
    snaive(train["y"].to_numpy(float)).assign(ds=truth["ds"].to_numpy()).to_csv(out / "snaive.csv", index=False)

    print(f"{args.stage}: {len(board)} students, {board['valid'].sum()} valid files")
    for _, r in board[(~board["valid"]) | (board["notes"] != "")].iterrows():
        print(f"  {'INVALID' if not r['valid'] else 'note   '} {r['student']}: {r['notes']}")

    # The public leaderboard: each student's latest valid file so far, scored on the first 24 hours.
    done = [pd.read_csv(out / f"{s}.csv") for s in STAGES[:STAGES.index(args.stage) + 1]
            if (out / f"{s}.csv").exists()]
    public = leaderboard(pd.concat([latest_valid(done), pd.DataFrame([bench])], ignore_index=True), "public")
    public = with_changes(public, args.stage, out)
    public.to_csv(out / "public.csv", index=False)
    public.to_csv(out / f"public_{args.stage}.csv", index=False)
    (out / "public.json").write_text(json.dumps({
        "stage": args.stage, "updated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "students": int(len(public) - 1),
    }), encoding="utf-8")
    print(f"public leaderboard: {len(public) - 1} students + benchmark -> board/public.csv")

    if args.stage == "final":
        s1 = pd.read_csv(out / "session1.csv")
        pts, best = points(s1, board, bench)
        pts.to_csv(out / "points.csv", index=False)
        private = leaderboard(pd.concat([best, pd.DataFrame([bench])], ignore_index=True), "private")
        private = private.merge(best[["student", "stage"]], on="student", how="left")
        private.to_csv(out / "private.csv", index=False)
        repo_links(folder).to_csv(out / "repos.csv", index=False)
        print(pts.to_string(index=False))


if __name__ == "__main__":
    main()
