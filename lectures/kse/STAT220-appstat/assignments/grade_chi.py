"""
grade_chi.py — INSTRUCTOR ONLY. Reveal the hidden world AND the realized data.

Lab 7 (chi-square / categorical A/B) companion to ``grade.py`` /
``grade_mc.py`` / ``grade_gof.py``. For each student email it prints:

  * the **intended** hidden world from :mod:`greenbowl_chi` and the decision the
    lab expects, plus the decisive evidence (from ``_reveal``), and
  * the **realized** diagnostics actually computed from *that student's* table
    (SRM p, raw/merged homogeneity p + Cramer's V + min_expected, the
    Premium-or-not 2x2 Fisher p, the arm x city p, and the within-city reject
    count), and
  * a ``⚠`` flag whenever the realized data **contradicts** the intended
    signature — because by design ~2-8% of seeds in some worlds land off their
    archetype, and a student who correctly reported what *their* (unlucky)
    sample showed must NOT be failed against the intended verdict.

Usage
-----
    # one email
    uv run --with numpy --with scipy --with pandas python grade_chi.py student7@kse.org.ua

    # a whole roster (any CSV; the email column is auto-detected)
    uv run --with numpy --with scipy --with pandas python grade_chi.py roster.csv
    uv run --with numpy --with scipy --with pandas python grade_chi.py roster.csv --out grades.csv

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
from scipy.stats import chisquare, chi2_contingency, fisher_exact

import greenbowl_chi as g


def _homog_full(table):
    """Homogeneity on a table as-is. Tolerates an all-zero column/row (the
    chi-square is then simply inapplicable -> ran=False)."""
    table = np.asarray(table, float)
    try:
        chi2, p, dof, expd = chi2_contingency(table)
    except ValueError:                       # zero-marginal column: too sparse
        return {"ran": False, "p": np.nan, "V": np.nan, "minE": np.nan}
    n = table.sum()
    V = np.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"ran": True, "p": p, "V": V, "minE": expd.min()}


def realized(email):
    """Compute the diagnostics a correct student would get from THIS table."""
    exp = g.load_my_experiment(email)
    tbl = exp.table().astype(float)               # 2 x 5 arm x basket
    n_c, n_t = exp.n_control, exp.n_test

    # SRM: goodness-of-fit of arm sizes vs the intended split
    obs = np.array([n_c, n_t], float)
    srm_p = chisquare(obs, f_exp=obs.sum() * np.array(g.INTENDED_SPLIT)).pvalue

    raw = _homog_full(tbl)                        # full 5-tier
    merged = _homog_full(np.column_stack([tbl[:, :3], tbl[:, 3] + tbl[:, 4]]))

    # Premium-or-not 2x2 + Fisher (the pilot's honest collapse)
    prem = tbl[:, 3] + tbl[:, 4]
    rest = tbl[:, :3].sum(axis=1)
    fisher_p = fisher_exact([[prem[0], rest[0]], [prem[1], rest[1]]])[1]

    # arm x city independence + within-city reject count (the confound check)
    armcity = _homog_full(exp.table(by="arm_city").astype(float))
    within_reject = 0
    for ct in exp.table(by="city").values():
        r = _homog_full(np.asarray(ct, float))
        if r["ran"] and r["p"] < g.ALPHA:
            within_reject += 1

    return {"n_c": n_c, "n_t": n_t, "srm_p": srm_p, "raw": raw,
            "merged": merged, "fisher_p": fisher_p, "armcity": armcity,
            "within_reject": within_reject}


def discrepancies(world, d):
    """List the ways the realized data contradicts the intended signature."""
    out = []
    raw, merged = d["raw"], d["merged"]
    if world == "win":
        if d["srm_p"] < g.ALPHA:
            out.append("SRM unexpectedly FAILS")
        if not (raw["ran"] and raw["p"] < g.ALPHA):
            out.append("homogeneity does NOT reject")
        elif raw["V"] < 0.10:
            out.append(f"V={raw['V']:.3f} below the 'win' range")
    elif world == "trivial":
        if not (merged["ran"] and merged["p"] < g.ALPHA):
            out.append("homogeneity does NOT reject (effect washed out)")
        if merged["ran"] and merged["V"] >= 0.02:
            out.append(f"V={merged['V']:.3f} not 'trivial'")
    elif world == "srm":
        if d["srm_p"] >= g.ALPHA:
            out.append("SRM does NOT fail")
    elif world == "pilot":
        if d["fisher_p"] < g.ALPHA:
            out.append("Fisher IS significant (lucky pilot)")
    elif world == "rare":
        if raw["ran"] and not np.isnan(raw["minE"]) and raw["minE"] >= 5:
            out.append(f"min_expected={raw['minE']:.1f} >= 5 (E>=5 rule not triggered)")
        if merged["ran"] and merged["p"] < g.ALPHA:
            out.append("merged test rejects (spurious)")
    elif world == "confound":
        if not (d["armcity"]["ran"] and d["armcity"]["p"] < g.ALPHA):
            out.append("arm x city NOT dependent")
        if not (raw["ran"] and raw["p"] < g.ALPHA):
            out.append("pooled homogeneity does NOT reject")
    return out


def reference(email):
    """Return a dict: email -> intended world + realized diagnostics + flags."""
    r = g._reveal(email)
    d = realized(email)
    flags = discrepancies(r["world"], d)
    return {"email": email, **r, "realized": d, "flags": flags}


def _fmt_realized(d):
    raw, merged = d["raw"], d["merged"]
    raw_s = (f"p={raw['p']:.2g} V={raw['V']:.3f} minE={raw['minE']:.1f}"
             if raw["ran"] else "raw 5-tier: too sparse to run")
    return (f"n={d['n_c']}/{d['n_t']}  SRM p={d['srm_p']:.3g}  "
            f"raw[{raw_s}]  merged p={merged['p']:.2g}  "
            f"Fisher(2x2) p={d['fisher_p']:.3g}  "
            f"armxcity p={d['armcity']['p']:.2g}  "
            f"within-city rejects={d['within_reject']}/4")


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
        description="Reveal the hidden world + realized diagnostics per student (Lab 7).")
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
              "'pilot'/'confound'.)")

    if args.out:
        cols = ["email", "world", "difficulty", "decision", "evidence",
                "n_control", "n_test", "srm_p", "raw_p", "raw_V", "raw_minE",
                "merged_p", "merged_V", "fisher_p", "armcity_p",
                "within_reject", "flags"]
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
                    "srm_p": d["srm_p"],
                    "raw_p": d["raw"]["p"], "raw_V": d["raw"]["V"],
                    "raw_minE": d["raw"]["minE"],
                    "merged_p": d["merged"]["p"], "merged_V": d["merged"]["V"],
                    "fisher_p": d["fisher_p"], "armcity_p": d["armcity"]["p"],
                    "within_reject": d["within_reject"],
                    "flags": "; ".join(r["flags"]),
                })
        print(f"\nSaved {len(results)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
