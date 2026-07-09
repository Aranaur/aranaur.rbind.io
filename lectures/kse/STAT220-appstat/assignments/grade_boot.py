"""
grade_boot.py — INSTRUCTOR ONLY. Reveal the hidden world AND the realized data.

Lab 9 (final: bootstrap / revenue-uplift A/B) companion to ``grade.py`` /
``grade_mc.py`` / ``grade_gof.py`` / ``grade_chi.py`` / ``grade_mu.py``. For
each student email it prints:

  * the **intended** hidden world from :mod:`greenbowl_boot` and the decision
    the lab expects, plus the decisive evidence (from ``_reveal``), and
  * the **realized** diagnostics actually computed from *that student's* data
    (arm sizes, ARPU per arm, zero share, the naive headline %-CI, the honest
    Poisson-bootstrap uplift CI and its three-zone read against BREAKEVEN,
    P(uplift >= breakeven), EV per year, the median-uplift CI, the top-1/
    top-5 user share of each arm's revenue, the winsorised-at-P99 uplift,
    and — for small-n worlds — the simulated a-priori power at DESIGN_MDE),
    and
  * a ``⚠`` flag whenever the realized data **contradicts** the intended
    signature — the builders redraw until the archetype holds, but a few
    seeds can still land off it, and a student who correctly reported what
    *their* (unlucky) sample showed must NOT be failed against the intended
    verdict.

Usage
-----
    # one email
    uv run --with numpy --with pandas --with statsmodels --with scipy python grade_boot.py student7@kse.org.ua

    # a whole roster (any CSV; the email column is auto-detected)
    uv run --with numpy --with pandas --with statsmodels --with scipy python grade_boot.py roster.csv
    uv run --with numpy --with pandas --with statsmodels --with scipy python grade_boot.py roster.csv --out grades.csv

    # also verify each student's NATIONAL streaming readout (the Lab 9
    # `uplift_ci_stream` task) against the hidden country truth (~15 s/student)
    uv run --with numpy --with pandas --with statsmodels --with scipy python grade_boot.py student7@kse.org.ua --country

On Windows add ``PYTHONUTF8=1`` in front so Cyrillic prints cleanly.

The roster CSV may have a header. The email column is taken as the first
column whose name contains "email"/"mail"/"пошта"/"id", or — if there is no
such header — the first column. Blank lines are skipped.
"""

import argparse
import collections
import csv
import sys

import numpy as np
from scipy.stats import ttest_ind

import greenbowl_boot as g

_MED_CAP = 20_000    # median-uplift bootstrap on huge arms -> fixed subsample
_POWER_N_MAX = 2500  # simulate power at DESIGN_MDE only for small-n worlds


def _pois_uplift_ci(c, t, B=2000, alpha=0.05, seed=0):
    """Central Poisson-bootstrap CI for mean(t)/mean(c) - 1, chunked over B
    so the weight matrices stay small even at 150k/arm."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(c, float), np.asarray(t, float)
    up_hat = t.mean() / c.mean() - 1
    star = np.empty(B)
    chunk = max(1, int(1e7 // max(c.size, t.size)))
    done = 0
    while done < B:
        b = min(chunk, B - done)
        wc = rng.poisson(1.0, size=(b, c.size))
        wt = rng.poisson(1.0, size=(b, t.size))
        star[done:done + b] = ((wt @ t / wt.sum(axis=1))
                               / (wc @ c / wc.sum(axis=1)) - 1)
        done += b
    q_lo, q_hi = np.quantile(star, [alpha / 2, 1 - alpha / 2])
    return up_hat, 2 * up_hat - q_hi, 2 * up_hat - q_lo, star


def _median_uplift_ci(c, t, B=1000, alpha=0.05, seed=0, cap=_MED_CAP):
    """Central classic-bootstrap CI for median(t)/median(c) - 1 (subsampled
    with a fixed seed when an arm is too large for the B x N matrix)."""
    rng = np.random.default_rng(seed)
    c, t = np.asarray(c, float), np.asarray(t, float)
    if c.size > cap:
        c = rng.choice(c, cap, replace=False)
    if t.size > cap:
        t = rng.choice(t, cap, replace=False)
    up_hat = np.median(t) / np.median(c) - 1
    c_star = np.median(rng.choice(c, size=(B, c.size), replace=True), axis=1)
    t_star = np.median(rng.choice(t, size=(B, t.size), replace=True), axis=1)
    star = t_star / c_star - 1
    q_lo, q_hi = np.quantile(star, [alpha / 2, 1 - alpha / 2])
    return up_hat, 2 * up_hat - q_hi, 2 * up_hat - q_lo


def _naive_pct_ci(c, t, z=1.96):
    """The platform headline: UAH-difference CI divided by the frozen
    control mean (the Practice 9 trap)."""
    d_hat = t.mean() - c.mean()
    se = np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size)
    return (d_hat - z * se) / c.mean(), (d_hat + z * se) / c.mean()


def _top_share(x, k):
    """Share of an arm's total revenue held by its top-k users."""
    x = np.sort(np.asarray(x, float))
    return float(x[-k:].sum() / x.sum())


def _winsorised_uplift(c, t, q=99):
    """Uplift of means after capping both arms at their own P99."""
    cw = np.minimum(c, np.percentile(c, q))
    tw = np.minimum(t, np.percentile(t, q))
    return float(tw.mean() / cw.mean() - 1)


def _power_at_mde(control, n, uplift=None, n_exps=100, B=300, seed=1):
    """A-priori power at DESIGN_MDE by simulation: resample the control as
    the 'world', inject the uplift, count Poisson-CI rejections vs zero."""
    if uplift is None:
        uplift = g.DESIGN_MDE
    rng, hits = np.random.default_rng(seed), 0
    pilot = np.asarray(control, float)
    for i in range(n_exps):
        c = rng.choice(pilot, n, replace=True)
        t = rng.choice(pilot, n, replace=True) * (1 + uplift)
        _, lo, _hi, _ = _pois_uplift_ci(c, t, B=B, seed=seed + i)
        hits += lo > 0
    return hits / n_exps


def _country_ci(email, B=600, alpha=0.05, seed=0):
    """One-pass streaming Poisson bootstrap over the national warehouse —
    the reference implementation of what the lab's `uplift_ci_stream` should
    produce (slow-ish: ~15 s per student; opt-in via --country)."""
    stream = g.load_country_stream(email)
    S_c, W_c = np.zeros(B), np.zeros(B)
    S_t, W_t = np.zeros(B), np.zeros(B)
    sum_c = sum_t = 0.0
    n_c = n_t = 0
    for i in range(stream.n_shards):
        c, t = stream.shard(i)
        rng = np.random.default_rng((seed, i))
        wc = rng.poisson(1.0, size=(B, c.size))
        wt = rng.poisson(1.0, size=(B, t.size))
        S_c += wc @ c
        W_c += wc.sum(axis=1)
        S_t += wt @ t
        W_t += wt.sum(axis=1)
        sum_c += c.sum(); n_c += c.size
        sum_t += t.sum(); n_t += t.size
    up_hat = (sum_t / n_t) / (sum_c / n_c) - 1
    star = (S_t / W_t) / (S_c / W_c) - 1
    q_lo, q_hi = np.quantile(star, [alpha / 2, 1 - alpha / 2])
    return float(up_hat), float(2 * up_hat - q_hi), float(2 * up_hat - q_lo)


def _zone(lo, hi):
    if lo > g.BREAKEVEN:
        return "SHIP"
    if hi < g.BREAKEVEN:
        return "KILL"
    return "GREY"


def realized(email):
    """Compute the diagnostics a correct student would get from THIS data."""
    exp = g.load_my_experiment(email)
    c, t = exp.control, exp.test

    n_big = max(c.size, t.size) > 50_000
    up, lo, hi, star = _pois_uplift_ci(c, t, B=1000 if n_big else 2000)
    med_up, med_lo, med_hi = _median_uplift_ci(c, t)
    naive_lo, naive_hi = _naive_pct_ci(c, t)

    out = {
        "city": exp.city, "n_c": exp.n_control, "n_t": exp.n_test,
        "user_base": exp.user_base,
        "arpu_c": float(c.mean()), "arpu_t": float(t.mean()),
        "zero_share": float((c == 0).mean()),
        "welch_p": float(ttest_ind(t, c, equal_var=False).pvalue),
        "naive_lo": float(naive_lo), "naive_hi": float(naive_hi),
        "uplift": float(up), "lo": float(lo), "hi": float(hi),
        "zone": _zone(lo, hi),
        "p_breakeven": float((star >= g.BREAKEVEN).mean()),
        "ev_myr": float(exp.annual_revenue
                        * (star.mean() - g.BREAKEVEN) / 1e6),
        "med_uplift": float(med_up),
        "med_lo": float(med_lo), "med_hi": float(med_hi),
        "top1_t": _top_share(t, 1), "top5_t": _top_share(t, 5),
        "top5_c": _top_share(c, 5),
        "wins_uplift": _winsorised_uplift(c, t),
        "power_mde": np.nan,
    }
    if exp.n_control <= _POWER_N_MAX:
        out["power_mde"] = _power_at_mde(c, min(exp.n_control, exp.n_test))
    return out


def discrepancies(world, d):
    """List the ways the realized data contradicts the intended signature."""
    out = []
    if world == "ship":
        if d["zone"] != "SHIP":
            out.append(f"zone={d['zone']}, CI does not clear breakeven")
    elif world == "harm":
        if d["hi"] >= 0:
            out.append("CI does not sit entirely below zero")
        if d["zone"] != "KILL":
            out.append(f"zone={d['zone']} instead of KILL")
    elif world == "grey":
        if d["lo"] <= 0:
            out.append("not significant against zero")
        if not (d["lo"] < g.BREAKEVEN < d["hi"]):
            out.append("breakeven is not inside the CI")
    elif world == "trivial":
        if d["lo"] <= 0:
            out.append("not significant against zero")
        if d["hi"] >= g.BREAKEVEN:
            out.append("CI reaches breakeven (not 'trivial')")
    elif world == "whale":
        if d["top5_t"] < 0.04:
            out.append(f"top-5 share={d['top5_t']:.1%}: whale signature weak")
        if abs(d["wins_uplift"]) > 0.025:
            out.append(f"winsorised uplift={d['wins_uplift']:+.1%} not ~0")
        if d["welch_p"] >= g.ALPHA and d["naive_lo"] <= 0:
            out.append("the naive headline does not even 'win'")
    elif world == "under":
        if not (d["lo"] < 0 < d["hi"]):
            out.append("CI does not cover zero")
        if d["hi"] - d["lo"] < 0.12:
            out.append(f"CI width={d['hi'] - d['lo']:.1%} not 'hopelessly wide'")
        if not np.isnan(d["power_mde"]) and d["power_mde"] > 0.35:
            out.append(f"power at MDE={d['power_mde']:.0%} not tiny")
    return out


def reference(email, country=False):
    """Return a dict: email -> intended world + realized diagnostics + flags."""
    r = g._reveal(email)
    d = realized(email)
    flags = discrepancies(r["world"], d)
    out = {"email": email, **r, "realized": d, "flags": flags,
           "country": None}
    if country:
        up, lo, hi = _country_ci(email)
        out["country"] = {"uplift": up, "lo": lo, "hi": hi}
        if not (lo < r["country_uplift"] < hi):
            flags.append(f"country CI ({lo:+.2%}, {hi:+.2%}) misses the "
                         f"true {r['country_uplift']:+.2%}")
    return out


def _fmt_realized(d):
    s = (f"{d['city']}, n={d['n_c']}/{d['n_t']}  ARPU={d['arpu_c']:.0f}->"
         f"{d['arpu_t']:.0f} UAH  zeros={d['zero_share']:.0%}\n"
         f"  uplift={d['uplift']:+.2%}  CI=({d['lo']:+.2%}, {d['hi']:+.2%})"
         f"  zone={d['zone']}  P(>=BE)={d['p_breakeven']:.0%}"
         f"  EV={d['ev_myr']:+.1f}M UAH/yr\n"
         f"  naive=({d['naive_lo']:+.2%}, {d['naive_hi']:+.2%})"
         f"  Welch p={d['welch_p']:.2g}"
         f"  median uplift={d['med_uplift']:+.2%}"
         f" ({d['med_lo']:+.2%}, {d['med_hi']:+.2%})\n"
         f"  top1/top5 test share={d['top1_t']:.1%}/{d['top5_t']:.1%}"
         f" (control top5={d['top5_c']:.1%})"
         f"  winsorised uplift={d['wins_uplift']:+.2%}")
    if not np.isnan(d["power_mde"]):
        s += f"  power@MDE={d['power_mde']:.0%}"
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
        description="Reveal the hidden world + realized diagnostics per student (Lab 9).")
    ap.add_argument("source", help="a student email, or a path to a roster CSV")
    ap.add_argument("--out", help="write the full table to this CSV", default=None)
    ap.add_argument("--country", action="store_true",
                    help="also compute the national streaming readout "
                         "(~15 s per student) and check it against the "
                         "revealed country truth")
    args = ap.parse_args(argv)

    emails = (_read_emails(args.source) if args.source.lower().endswith(".csv")
              else [args.source])
    results = [reference(e, country=args.country) for e in emails]

    w = max((len(r["email"]) for r in results), default=5)
    for r in results:
        print(f"{r['email']:{w}s}  {r['world']:8s} [{r['difficulty']:6s}]  -> {r['decision']}")
        print(f"{'':{w}s}  evidence: {r['evidence']}")
        lines = _fmt_realized(r["realized"]).splitlines()
        print(f"{'':{w}s}  realized: {lines[0]}")
        for line in lines[1:]:
            print(f"{'':{w}s}           {line}")
        if r["country"] is not None:
            cc = r["country"]
            print(f"{'':{w}s}  country:  uplift={cc['uplift']:+.2%}  "
                  f"CI=({cc['lo']:+.2%}, {cc['hi']:+.2%})  "
                  f"true={r['country_uplift']:+.2%}")
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
        print("(grade difficulty-aware: a clean 'ship' is less work than a "
              "'whale'/'under'; the lab is worth 16 points.)")

    if args.out:
        cols = ["email", "world", "difficulty", "decision", "evidence",
                "city", "n_control", "n_test", "user_base", "arpu_c",
                "arpu_t", "zero_share", "welch_p", "naive_lo", "naive_hi",
                "uplift", "ci_lo", "ci_hi", "zone", "p_breakeven", "ev_myr",
                "med_uplift", "med_lo", "med_hi", "top1_t", "top5_t",
                "top5_c", "wins_uplift", "power_mde", "country_uplift_true",
                "country_uplift", "country_lo", "country_hi", "flags"]
        with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=cols)
            wr.writeheader()
            for r in results:
                d = r["realized"]
                wr.writerow({
                    "email": r["email"], "world": r["world"],
                    "difficulty": r["difficulty"],
                    "decision": r["decision"], "evidence": r["evidence"],
                    "city": d["city"], "n_control": d["n_c"],
                    "n_test": d["n_t"], "user_base": d["user_base"],
                    "arpu_c": d["arpu_c"], "arpu_t": d["arpu_t"],
                    "zero_share": d["zero_share"], "welch_p": d["welch_p"],
                    "naive_lo": d["naive_lo"], "naive_hi": d["naive_hi"],
                    "uplift": d["uplift"], "ci_lo": d["lo"],
                    "ci_hi": d["hi"], "zone": d["zone"],
                    "p_breakeven": d["p_breakeven"], "ev_myr": d["ev_myr"],
                    "med_uplift": d["med_uplift"], "med_lo": d["med_lo"],
                    "med_hi": d["med_hi"], "top1_t": d["top1_t"],
                    "top5_t": d["top5_t"], "top5_c": d["top5_c"],
                    "wins_uplift": d["wins_uplift"],
                    "power_mde": d["power_mde"],
                    "country_uplift_true": r["country_uplift"],
                    "country_uplift": (r["country"] or {}).get("uplift", ""),
                    "country_lo": (r["country"] or {}).get("lo", ""),
                    "country_hi": (r["country"] or {}).get("hi", ""),
                    "flags": "; ".join(r["flags"]),
                })
        print(f"\nSaved {len(results)} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
