"""
greenbowl_mc.py — STAT220 *Applied Statistics*, Lab 5 data platform.

Lab 4 handed you a fixed A/B dataset and asked for a business decision.
**Lab 5 is one level up: you audit the *tool itself*.**

The GreenBowl data-engineering team shipped a new significance criterion,
``team_criterion``, that the experimentation platform now runs on **every**
A/B test in the company. Before you trust a single roll-out decision to it,
your job is to **validate it with Monte Carlo** — exactly the way the lecture
taught: generate experiments with a *known* answer and count how often the
criterion is right.

Every student receives a **different** hidden criterion, deterministically
derived from their student id. Some are correct; some are correct only under
certain conditions; some are quietly broken. Which one you got is **HIDDEN**.
You must recover the verdict from simulation alone — is its false-positive
rate calibrated, from what sample size, how does its power compare to the
textbook Welch test, and does it survive a thousand A/A tests on real logs?

Fixed context (the same for everyone, stated in the lab):

    * metric                ARPU (weekly revenue per user, UAH) — skewed
    * baseline ARPU         mu0    = 250 UAH
    * significance level    alpha  = 0.05
    * the criterion under audit is ``team_criterion`` from this module
    * the trusted reference is ``reference_welch`` from this module

Usage (students):

    from greenbowl_mc import (load_my_criterion, load_my_log,
                              reference_welch, arpu_sample, BASELINE, ALPHA)

    team_criterion = load_my_criterion("your.email@kse.org.ua")
    pvalue = team_criterion(test, control, alternative="two-sided")

    log = load_my_log("your.email@kse.org.ua")   # a historical ARPU log

GROUND RULE
-----------
Do **not** read past the line marked ``# === instructor section ===`` and do
**not** call ``_reveal``. The whole point is to decide *without* knowing which
criterion you were handed. Reverse-engineering it from the source instead of
validating it by simulation scores **zero** for interpretation. (Instructors
use ``_reveal`` to grade.)
"""

import hashlib

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["load_my_criterion", "load_my_log", "reference_welch",
           "arpu_sample", "BASELINE", "ALPHA"]

# --- public, documented constants -------------------------------------------
BASELINE = 250.0      # mu0, historical ARPU in UAH
ALPHA = 0.05          # significance level


def arpu_sample(true_mean, n, sigma=0.6, rng=None):
    """A lognormal ARPU sample with a given **mean** ``true_mean``.

    ``sigma`` controls the spread / tail heaviness *without* shifting the
    mean. This is the same skewed-money generator you used in Practice 5;
    it is your "omniscient god" tool — use it to manufacture H0 and H1.

    Pass ``rng`` (a ``np.random.default_rng(...)``) for reproducibility, or
    leave it ``None`` to use the global NumPy RNG (so ``np.random.seed``
    still works).
    """
    mu = np.log(true_mean) - sigma ** 2 / 2
    if rng is None:
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def reference_welch(test, control, alternative="two-sided"):
    """The trusted, textbook two-sample **Welch** t-test (the reference).

    Use this as the criterion you *trust* when comparing power: a candidate
    criterion is only worth adopting if it is (a) valid and (b) at least as
    powerful as this one.
    """
    return stats.ttest_ind(test, control, equal_var=False,
                           alternative=alternative).pvalue


def load_my_criterion(student_id):
    """Return **your** hidden ``team_criterion(test, control, alternative)``.

    The returned object is a function with the signature

        team_criterion(test, control, alternative="two-sided") -> p_value

    where ``alternative`` is about ``test - control`` (so ``"greater"`` means
    "test beats control"), matching ``scipy.stats.ttest_ind``. The two arrays
    may have different lengths. **What the criterion does inside is hidden** —
    validate it, do not dissect it.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    world = _world(student_id)
    impl = _CRITERIA[world]

    def team_criterion(test, control, alternative="two-sided"):
        return impl(np.asarray(test, float), np.asarray(control, float),
                    alternative)

    team_criterion.__doc__ = ("The candidate criterion shipped by the "
                              "GreenBowl data team. Validate it with Monte "
                              "Carlo; do not read its source.")
    return team_criterion


def load_my_log(student_id):
    """Return a 'historical' ARPU log for ``student_id`` (city x cuisine).

    A ``pandas.DataFrame`` with columns ``city``, ``cuisine``, ``arpu`` —
    your real-looking GreenBowl logs, to run a thousand A/A tests on. The
    per-slice means and skewness vary; the log is unique to your id.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    rng = np.random.default_rng(_seed_from_id(student_id) + 99)
    cities = [f"city_{i}" for i in range(40)]
    cuisines = ["pizza", "sushi", "burger", "vegan", "asian",
                "bbq", "dessert", "salad"]
    rows = []
    for city in cities:
        base = rng.uniform(150, 400)              # its own average ARPU
        for cuisine in cuisines:
            n = int(rng.integers(150, 500))
            sig = rng.uniform(0.5, 0.9)           # its own skewness
            arpu = arpu_sample(base, n, sigma=sig, rng=rng)
            rows.append(pd.DataFrame({"city": city, "cuisine": cuisine,
                                      "arpu": arpu}))
    return pd.concat(rows, ignore_index=True)


# ============================================================================
# === instructor section =====================================================
# Students: stop reading here. Everything below defines (and can reveal) the
# hidden ground truth used for grading.
# ============================================================================

# Six candidate criteria. A correct lab reaches a *different* verdict for each,
# and the only way to tell them apart is by simulation. Verdicts are framed as
# APPROVE / APPROVE-WITH-CAVEAT / REJECT, plus a power comparison vs Welch.
#
#   welch         correct Welch                 -> APPROVE (valid + full power)
#   pooled        Student's pooled t            -> CAVEAT: valid only on
#                                                  balanced / equal-variance
#                                                  designs; inflated FPR when
#                                                  groups differ in size & spread
#   zapprox       Welch stat, NORMAL quantiles  -> CAVEAT: inflated FPR at small
#                                                  n (ignores df); self-heals
#                                                  once n is large
#   conservative  Welch with SE inflated x1.20  -> APPROVE-BUT-WEAK: FPR < alpha
#                                                  (safe) but loses power; the
#                                                  reference Welch dominates
#   inflated      Welch with SE deflated x0.80  -> REJECT: anti-conservative at
#                                                  EVERY n (FPR ~ 0.12)
#   dredge        min p of two random halves    -> REJECT: data-dredging bug,
#                                                  FPR ~ 0.10 at every n
#                                                  (independent of n); its
#                                                  power looks fine only
#                                                  because it over-rejects
_WORLDS = ("welch", "pooled", "zapprox", "conservative", "inflated", "dredge")


def _seed_from_id(student_id):
    """Deterministic 32-bit seed from a normalised id."""
    norm = student_id.strip().lower().encode("utf-8")
    return int(hashlib.sha256(norm).hexdigest()[:8], 16)


def _world(student_id):
    """Pick the hidden criterion world for this id."""
    return _WORLDS[_seed_from_id(student_id) % len(_WORLDS)]


# --- the criteria themselves (test - control convention) --------------------

def _welch_parts(test, control):
    """Welch t-statistic, Welch-Satterthwaite df, and the diff/SE."""
    na, nb = len(test), len(control)
    ma, mb = test.mean(), control.mean()
    va, vb = test.var(ddof=1), control.var(ddof=1)
    se = np.sqrt(va / na + vb / nb)
    t = (ma - mb) / se
    df = (va / na + vb / nb) ** 2 / (
        (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    return t, df, se, (ma - mb)


def _p_from(t, dist, alternative):
    """Tail probability for statistic ``t`` under ``dist`` (t or normal)."""
    if alternative == "two-sided":
        return 2.0 * dist.sf(abs(t))
    if alternative == "greater":
        return dist.sf(t)
    if alternative == "less":
        return dist.cdf(t)
    raise ValueError(f"bad alternative: {alternative}")


def _c_welch(test, control, alternative):
    return stats.ttest_ind(test, control, equal_var=False,
                           alternative=alternative).pvalue


def _c_pooled(test, control, alternative):
    return stats.ttest_ind(test, control, equal_var=True,
                           alternative=alternative).pvalue


def _c_zapprox(test, control, alternative):
    # Welch statistic, but p-values from the NORMAL (ignores finite df).
    t, _df, _se, _d = _welch_parts(test, control)
    return _p_from(t, stats.norm, alternative)


def _c_conservative(test, control, alternative):
    # Welch, but the standard error is over-stated by 20% -> t shrinks ->
    # larger p -> fewer rejections -> FPR below alpha at every n.
    t, df, _se, _d = _welch_parts(test, control)
    return _p_from(t / 1.20, stats.t(df), alternative)


def _c_inflated(test, control, alternative):
    # Welch, but the standard error is under-stated by 20% -> t inflates ->
    # smaller p -> too many rejections -> FPR above alpha at every n.
    t, df, _se, _d = _welch_parts(test, control)
    return _p_from(t / 0.80, stats.t(df), alternative)


def _c_dredge(test, control, alternative):
    # A "smarter" criterion that secretly splits each group in two and reports
    # the more significant half -> classic data-dredging -> inflated FPR that
    # does not depend on n, and LOWER power (each half sees half the data).
    ha, hb = len(test) // 2, len(control) // 2
    p1 = _c_welch(test[:ha], control[:hb], alternative)
    p2 = _c_welch(test[ha:2 * ha], control[hb:2 * hb], alternative)
    return min(p1, p2)


_CRITERIA = {
    "welch": _c_welch,
    "pooled": _c_pooled,
    "zapprox": _c_zapprox,
    "conservative": _c_conservative,
    "inflated": _c_inflated,
    "dredge": _c_dredge,
}

# Per-world: the verdict the lab expects, and the decisive Monte Carlo evidence.
_EXPECTED = {
    "welch": (
        "APPROVE",
        "FPR ~ alpha at all n; power == reference Welch. The criterion is the "
        "textbook Welch test."),
    "pooled": (
        "APPROVE WITH CAVEAT (balanced/equal-variance only)",
        "Valid (FPR ~ alpha) on balanced equal-variance designs, but FPR is "
        "inflated (~0.10-0.13) when the groups differ in BOTH size and spread "
        "-> it is pooled Student, not Welch. The 50/50 historical A/A passes, "
        "so the synthetic unbalanced stress test is what exposes it."),
    "zapprox": (
        "APPROVE WITH CAVEAT (large n only)",
        "FPR is inflated at small n (uses normal quantiles, ignores df) and "
        "self-heals once n is large; historical n is large enough that the "
        "1000 A/A passes -> only the small-n synthetic check catches it."),
    "conservative": (
        "APPROVE BUT WEAK (valid yet underpowered)",
        "FPR < alpha at all n (safe), but power is strictly below the "
        "reference Welch -> the SE is over-stated; prefer Welch."),
    "inflated": (
        "REJECT (anti-conservative at every n)",
        "FPR ~ 0.11-0.13 even on large balanced equal-variance samples and "
        "does not heal as n grows -> the SE is under-stated; unsafe."),
    "dredge": (
        "REJECT (data-dredging bug)",
        "FPR ~ 0.10 at every n and does not heal as n grows -> it secretly "
        "tests two random halves and keeps the more significant one. Its "
        "'power' is not a real advantage, just a symptom of over-rejection; "
        "an over-sized FPR makes the power number meaningless."),
}


def _reveal(student_id):
    """INSTRUCTOR ONLY. Return the hidden ground truth for ``student_id``."""
    world = _world(student_id)
    verdict, evidence = _EXPECTED[world]
    return {
        "criterion": world,
        "verdict": verdict,
        "evidence": evidence,
    }
