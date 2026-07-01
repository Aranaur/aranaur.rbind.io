"""
grade_mu.py — INSTRUCTOR ONLY. Reveal the hidden world AND the realized data.

Lab 8 (nonparametric tests / delivery-time A/B) companion to ``grade.py`` /
``grade_mc.py`` / ``grade_gof.py`` / ``grade_chi.py``. For each student email
it prints:

  * the **intended** hidden world from :mod:`greenbowl_mu` and the decision the
    lab expects, plus the decisive evidence (from ``_reveal``), and
  * the **realized** diagnostics actually computed from *that student's* data
    (design paired?, headline Welch-t p, Mann-Whitney p, P(test<control), the
    Hodges-Lehmann shift, Brunner-Munzel p, the median difference and the
    P90 ratio; for the paired world also the sign-test and signed-rank p and
    the median per-courier saving), and
  * a ``⚠`` flag whenever the realized data **contradicts** the intended
    signature — because by design ~2-8% of seeds in some worlds land off their
    archetype, and a student who correctly reported what *their* (unlucky)
    sample showed must NOT be failed against the intended verdict.

Usage
-----
    # one email
    uv run --with numpy --with scipy --with pandas python grade_mu.py student7@kse.org.ua

    # a whole roster (any CSV; the email column is auto-detected)
    uv run --with numpy --with scipy --with pandas python grade_mu.py roster.csv
    uv run --with numpy --with scipy --with pandas python grade_mu.py roster.csv --out grades.csv

On Windows add ``PYTHONUTF8=1`` in front so Cyrillic prints cleanly.

The roster CSV may have a header. The email column is taken as the first column
whose name contains "email"/"mail"/"пошта"/"id", or — if there is no such
header — the first column. Blank lines are skipped.
"""

import argparse
import collections
import csv
import sys

import numpy as np
from scipy.stats import (mannwhitneyu, ttest_ind, wilcoxon, binomtest,
                         brunnermunzel)

import greenbowl_mu as g

_HL_CAP = 2000   # HL on 150k x 150k pairs is infeasible -> fixed-seed subsample


def _hl_shift(t, c, cap=_HL_CAP):
    """Hodges-Lehmann shift (test - control), min. Negative = test faster.
    Subsamples with a fixed seed when an arm is too large for the full
    pairwise matrix (the lab tells students to do the same)."""
    rng = np.random.default_rng(0)
    if len(t) > cap:
        t = rng.choice(t, cap, replace=False)
    if len(c) > cap:
        c = rng.choice(c, cap, replace=False)
    return float(np.median(np.subtract.outer(t, c)))


def realized(email):
    """Compute the diagnostics a correct student would get from THIS data."""
    exp = g.load_my_experiment(email)
    c, t = exp.control, exp.test

    df = exp.data
    paired = bool(df["courier_id"].duplicated().any())

    mw = mannwhitneyu(t, c)
    out = {
        "n_c": exp.n_control, "n_t": exp.n_test, "paired": paired,
        "welch_p": ttest_ind(c, t, equal_var=False).pvalue,
        "mw_p": mw.pvalue,
        # scipy's U for the first argument counts pairs test > control
        "P_t_lt_c": 1 - mw.statistic / (len(t) * len(c)),
        "hl": _hl_shift(t, c),
        "bm_p": brunnermunzel(c, t).pvalue,
        "med_diff": float(np.median(t) - np.median(c)),
        "p90_ratio": float(np.quantile(t, 0.9) / np.quantile(c, 0.9)),
        "sign_p": np.nan, "sr_p": np.nan, "med_saving": np.nan,
    }
    if paired:
        piv = df.pivot(index="courier_id", columns="arm", values="minutes")
        d = piv["control"].to_numpy() - piv["test"].to_numpy()  # >0 = faster
        nz = d[d != 0]
        out["sign_p"] = binomtest(int((nz > 0).sum()), len(nz), 0.5,
                                  alternative="greater").pvalue
        out["sr_p"] = wilcoxon(d, alternative="greater").pvalue
        out["med_saving"] = float(np.median(d))
    return out


def discrepancies(world, d):
    """List the ways the realized data contradicts the intended signature."""
    out = []
    a = g.ALPHA
    if world == "win":
        if d["mw_p"] >= a:
            out.append("Mann-Whitney does NOT reject")
        if d["hl"] > -g.BUSINESS_MDE:
            out.append(f"HL={d['hl']:.1f} min short of the business threshold")
    elif world == "null":
        if d["mw_p"] < a:
            out.append("Mann-Whitney rejects (unlucky A/A)")
        if d["welch_p"] < a:
            out.append("Welch t rejects (unlucky A/A)")
    elif world == "outlier":
        if d["welch_p"] >= a:
            out.append("Welch t does NOT reject (the fake win is missing)")
        if d["mw_p"] < a:
            out.append("Mann-Whitney rejects too (ranks see an effect)")
    elif world == "trivial":
        if d["mw_p"] >= a:
            out.append("Mann-Whitney does NOT reject (effect washed out)")
        if abs(d["hl"]) >= 1:
            out.append(f"HL={d['hl']:.2f} min not 'trivial'")
    elif world == "shape":
        if d["welch_p"] >= a:
            out.append("Welch t does NOT reject (mean gap missing)")
        if d["mw_p"] < a:
            out.append("Mann-Whitney rejects (P(t<c) drifted off 0.5)")
        if d["p90_ratio"] <= 1.5:
            out.append(f"P90 ratio={d['p90_ratio']:.2f} tail did not explode")
    elif world == "paired":
        if d["mw_p"] < a or d["welch_p"] < a:
            out.append("pooled tests reject (the pooled path sees the effect)")
        if d["sr_p"] >= a:
            out.append("signed-rank does NOT reject on the differences")
        elif d["sign_p"] >= a:
            out.append("sign test does NOT reject (signed-rank does)")
    return out


def reference(email):
    """Return a dict: email -> intended world + realized diagnostics + flags."""
    r = g._reveal(email)
    d = realized(email)
    flags = discrepancies(r["world"], d)
    return {"email": email, **r, "realized": d, "flags": flags}


def _fmt_realized(d):
    s = (f"n={d['n_c']}/{d['n_t']}  paired={'YES' if d['paired'] else 'no'}  "
         f"Welch p={d['welch_p']:.2g}  MW p={d['mw_p']:.2g}  "
         f"P(t<c)={d['P_t_lt_c']:.3f}  HL={d['hl']:+.1f} min  "
         f"BM p={d['bm_p']:.2g}  med diff={d['med_diff']:+.1f}  "
         f"P90 ratio={d['p90_ratio']:.2f}")
    if d["paired"]:
        s += (f"  |  sign p={d['sign_p']:.2g}  SR p={d['sr_p']:.2g}  "
              f"med saving={d['med_saving']:.1f} min")
    return s


def _read_emails(path):
    """Pull the email column out of a roster CSV (header optional)."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    header = rows[0]
    keys = ("email", "mail", "пошта", "id")
    col = 0
    if any(any(k in cell.lower() for k in keys) for cell in header):
        col = next(i for i, cell in enumerate(header)
                   if any(k in cell.lower() for k in keys))
        rows = rows[1:]  # drop the header
    return [row[col].strip() for row in rows if row and row[col].strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Reveal the hidden world + realized diagnostics per student (Lab 8).")
    ap.add_argument("source", help="a student email, or a path to a roster CSV")
    ap.add_argument("--out", help="write the full table to this CSV", default=None)
    args = ap.parse_args(argv)

    emails = (_read_emails(args.source) if args.source.lower().endswith(".csv")
              else [args.source])
    results = [reference(e) for e in emails]

    w = max((len(r["email"]) for r in results), default=5)
    for r in results:
        print(f"{r['email']:{w}s}  {r['world']:8s} [{r['difficulty']:6s}]  -> {r['decision']}")
        print(f"{'':{w}s}  evidence: {r['evidence']}")
        print(f"{'':{w}s}  realized: {_fmt_realized(r['realized'])}")
        if r["flags"]:
            print(f"{'':{w}s}  ⚠ realized data contradicts the intended world: "
                  + "; ".join(r["flags"]))

    if len(results) > 1:
        tally = collections.Counter(r["difficulty"] for r in results)
        worlds = collections.Counter(r["world"] for r in results)
        print(f"\nroster difficulty mix: "
              + ", ".join(f"{k}={tally.get(k, 0)}"
                          for k in ("easy", "medium", "hard")))
        print("roster world mix:      "
              + ", ".join(f"{wd}={worlds.get(wd, 0)}" for wd in g._WORLDS))
        print("(grade difficulty-aware: a clean 'win' is less work than a "
              "'shape'/'paired'.)")

    if args.out:
        cols = ["email", "world", "difficulty", "decision", "evidence",
                "n_control", "n_test", "paired", "welch_p", "mw_p", "P_t_lt_c",
                "hl_shift", "bm_p", "med_diff", "p90_ratio",
                "sign_p", "sr_p", "med_saving", "flags"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in results:
                d = r["realized"]
                wr.writerow({
                    "email": r["email"], "world": r["world"],
                    "difficulty": r["difficulty"],
                    "decision": r["decision"], "evidence": r["evidence"],
                    "n_control": d["n_c"], "n_test": d["n_t"],
                    "paired": d["paired"],
                    "welch_p": d["welch_p"], "mw_p": d["mw_p"],
                    "P_t_lt_c": d["P_t_lt_c"], "hl_shift": d["hl"],
                    "bm_p": d["bm_p"], "med_diff": d["med_diff"],
                    "p90_ratio": d["p90_ratio"],
                    "sign_p": d["sign_p"], "sr_p": d["sr_p"],
                    "med_saving": d["med_saving"],
                    "flags": "; ".join(r["flags"]),
                })
        print(f"\nSaved {len(results)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
