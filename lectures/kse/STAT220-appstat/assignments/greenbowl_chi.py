"""
greenbowl_chi.py — STAT220 *Applied Statistics*, Lab 7 data platform.

Lab 4 handed you a fixed A/B dataset and asked for a business decision.
Lab 5 made you audit the *criterion*. Lab 6 was distribution forensics on a
**numeric** metric. **Lab 7 is categorical A/B forensics: the metric is a
bucket, not a number, and a single "p < 0.05" line is a trap.**

The GreenBowl experimentation platform ran an A/B test of a new **menu
layout**. For every order it logged the **basket tier** the user landed in
(Economy / Standard / Large / Premium / VIP) and the **city** the order came
from. The automated report prints a tidy chi-square "the redesign changed the
basket mix — ship it" line. Your job is to decide whether you can **believe**
it.

Every student receives a **different** hidden experiment, deterministically
derived from their student id. Some got a genuine, business-meaningful shift;
some a difference that is statistically "significant" yet microscopic; some a
**broken split** (sample-ratio mismatch); some a **micro-pilot** too small for
the chi-square approximation; some a **rare category** that quietly invalidates
the test; and some an apparent win that is really a **city confound**. The only
honest way to find out which one you hold is to run the categorical toolkit
from the lecture — goodness-of-fit (the SRM guardrail), the chi-square test of
homogeneity, standardized residuals, Cramer's V, the expected-count rule,
Yates / Fisher on small tables, and a Monte-Carlo validation of your own
guardrail.

Fixed context (the same for everyone, stated in the lab):

    * unit                  one order
    * metric                basket tier  (Economy/Standard/Large/Premium/VIP)
    * covariate             city         (Kyiv/Lviv/Odesa/Kharkiv)
    * arms                  control = old layout, test = new layout
    * intended split        50 / 50
    * significance level    alpha = 0.05

Usage (students):

    from greenbowl_chi import (load_my_experiment, BASKETS, CITIES,
                               INTENDED_SPLIT, ALPHA)

    exp = load_my_experiment("your.email@kse.org.ua")
    exp.data            # tidy DataFrame: one row per order [arm, basket, city]
    exp.n_control, exp.n_test
    exp.table()         # 2 x 5 counts: rows = [control, test], cols = BASKETS

GROUND RULE
-----------
Do **not** read past the line marked ``# === instructor section ===`` and do
**not** call ``_reveal``. The whole point is to *diagnose* the experiment, not
to look up the answer. Naming the hidden pathology from the source instead of
demonstrating it with the toolkit scores **zero** for interpretation.
(Instructors use ``_reveal`` to grade.)
"""

import hashlib

import numpy as np
import pandas as pd

__all__ = ["load_my_experiment", "Experiment",
           "BASKETS", "CITIES", "INTENDED_SPLIT", "ALPHA"]

# --- public, documented constants -------------------------------------------
BASKETS = ["Economy", "Standard", "Large", "Premium", "VIP"]
CITIES = ["Kyiv", "Lviv", "Odesa", "Kharkiv"]
INTENDED_SPLIT = (0.5, 0.5)   # control / test, as designed
ALPHA = 0.05                  # significance level


class Experiment:
    """One hidden categorical A/B experiment, keyed to a student id.

    Attributes
    ----------
    data : pandas.DataFrame
        One row per order, columns ``arm`` ("control"/"test"), ``basket``
        (one of :data:`BASKETS`) and ``city`` (one of :data:`CITIES`).
    intended : tuple(float, float)
        The **designed** control/test allocation (always 50/50). The actually
        delivered arm sizes may or may not match it — whether they do is a
        guardrail you have to check.
    """

    def __init__(self, data, intended=INTENDED_SPLIT):
        self.data = data.reset_index(drop=True)
        self.intended = tuple(intended)

    @property
    def n_control(self):
        return int((self.data["arm"] == "control").sum())

    @property
    def n_test(self):
        return int((self.data["arm"] == "test").sum())

    def table(self, by=None):
        """Counts table. ``by=None`` → 2 x 5 arm x basket (rows control/test).

        Pass ``by="city"`` to get a dict ``{city: 2 x 5 table}`` for stratified
        analysis, or ``by="arm_city"`` for the arm x city table.
        """
        d = self.data
        if by is None:
            ct = pd.crosstab(d["arm"], d["basket"]).reindex(
                index=["control", "test"], columns=BASKETS, fill_value=0)
            return ct.to_numpy()
        if by == "arm_city":
            ct = pd.crosstab(d["arm"], d["city"]).reindex(
                index=["control", "test"], columns=CITIES, fill_value=0)
            return ct.to_numpy()
        if by == "city":
            out = {}
            for city in CITIES:
                sub = d[d["city"] == city]
                ct = pd.crosstab(sub["arm"], sub["basket"]).reindex(
                    index=["control", "test"], columns=BASKETS, fill_value=0)
                out[city] = ct.to_numpy()
            return out
        raise ValueError("by must be None, 'arm_city' or 'city'")

    def __repr__(self):
        return (f"<Experiment n_control={self.n_control} "
                f"n_test={self.n_test} "
                f"intended={self.intended[0]:.0%}/{self.intended[1]:.0%}>")


def load_my_experiment(student_id):
    """Return **your** hidden :class:`Experiment`, keyed to ``student_id``.

    What (if anything) is wrong with the experiment is **hidden**. Diagnose it
    with the categorical toolkit — do not try to read it off the source.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    rng = np.random.default_rng(_seed_from_id(student_id))
    return _BUILD[_world(student_id)](rng)


# ============================================================================
# === instructor section =====================================================
# Students: stop reading here. Everything below defines (and can reveal) the
# hidden ground truth used for grading.
# ============================================================================

# Six hidden experiment "worlds". A correct lab reaches a *different* decision
# for each, and the only way to tell them apart is with the chi-square toolkit:
#
#   win       honest 50/50 split, the new layout genuinely moves baskets up
#             (more Large/Premium)            -> SHIP: SRM passes, homogeneity
#                                                rejects, Cramer's V is non-
#                                                trivial, residuals show a real
#                                                upward shift.
#   trivial   honest split, HUGE n, a real    -> DON'T SHIP (trivial effect):
#             but microscopic shift             p ~ 0 but Cramer's V ~ 0.01 —
#                                               significant != strong.
#   srm       sample-ratio mismatch: the new   -> FIX THE SPLITTER (result void):
#             layout dropped ~7% of test         a chi-square GOF of arm sizes
#             users, and the dropped ones were   vs the intended 50/50 rejects.
#             low-spenders, so test LOOKS like   The basket "win" is an artifact
#             a win                              of differential dropout.
#   pilot     micro-pilot in a new city,       -> INSUFFICIENT EVIDENCE: expected
#             15 vs 15 orders                    counts << 5, naive chi-square
#                                               over-rejects; collapsed to a 2x2
#                                               Premium-or-not, Fisher's exact is
#                                               NOT significant -> collect more.
#   rare      modest n, the apparent effect    -> NO SHIPPABLE EFFECT (test was
#             lives entirely in the rare VIP     invalid): E(VIP) < 5 so the 5-
#             cell                               tier chi-square is invalid;
#                                               merging VIP into Premium makes
#                                               the arms identical -> p large.
#   confound  overall 50/50, but the test arm  -> DON'T TRUST (city confound):
#             over-sampled Lviv (a Premium-      pooled homogeneity rejects, yet
#             heavy city); within each city      arm x city is dependent and the
#             nothing changed                    within-city tables all pass —
#                                               the "win" is covariate imbalance.
_WORLDS = ("win", "trivial", "srm", "pilot", "rare", "confound")

# Base basket probabilities (Economy, Standard, Large, Premium, VIP).
_BP = np.array([0.45, 0.30, 0.155, 0.08, 0.015])
# Base city probabilities (Kyiv, Lviv, Odesa, Kharkiv).
_CP = np.array([0.40, 0.20, 0.22, 0.18])
# City-specific basket tastes (used by the confound world; same map for both
# arms, so there is no treatment effect — only a covariate imbalance).
_BASKET_BY_CITY = {
    "Kyiv":    np.array([0.45, 0.30, 0.155, 0.08, 0.015]),
    "Lviv":    np.array([0.30, 0.28, 0.22, 0.18, 0.02]),   # Premium-heavy
    "Odesa":   np.array([0.42, 0.31, 0.16, 0.09, 0.02]),
    "Kharkiv": np.array([0.55, 0.28, 0.11, 0.05, 0.01]),   # Economy-heavy
}


def _norm(p):
    p = np.asarray(p, float)
    return p / p.sum()


# A per-lab salt makes THIS lab's world assignment statistically INDEPENDENT of
# every other lab's. Without it, world = hash(id) % 6 is identical across any
# two labs that share the hash and have 6 worlds (e.g. Lab 6 / greenbowl_gof),
# so a student who lands on a "hard" index stays on it in every lab — the
# "some people always get the hard scenario" complaint. EACH NEW LAB MUST USE
# ITS OWN DISTINCT SALT so luck does not carry over across the course.
_LAB_SALT = "stat220-lab7-chi-v1"


def _seed_from_id(student_id):
    """Deterministic 32-bit seed from a normalised id (drives the DATA sample)."""
    norm = student_id.strip().lower().encode("utf-8")
    return int(hashlib.sha256(norm).hexdigest()[:8], 16)


def _jit(n, rng, frac=0.03):
    """Jitter an arm size by +/- frac so the absolute size cannot fingerprint
    a world across students. Left untouched for tiny pilots (n < 50)."""
    if n < 50:
        return n
    return int(n + round(rng.uniform(-frac, frac) * n))


def _world(student_id):
    """Pick the hidden experiment world for this id.

    Salted with ``_LAB_SALT`` so the world drawn here does NOT correlate with
    the world drawn in any other lab — no student is consistently lucky or
    unlucky across the course."""
    norm = (student_id.strip().lower() + "|" + _LAB_SALT).encode("utf-8")
    h = int(hashlib.sha256(norm).hexdigest()[:8], 16)
    return _WORLDS[h % len(_WORLDS)]


def _draw_arm(n, arm, basket_by_city, city_p, rng):
    """n orders for one arm: draw city, then basket | city."""
    cities = rng.choice(CITIES, size=n, p=_norm(city_p))
    baskets = np.empty(n, dtype=object)
    for city in CITIES:
        mask = cities == city
        m = int(mask.sum())
        if m:
            baskets[mask] = rng.choice(BASKETS, size=m,
                                       p=_norm(basket_by_city[city]))
    return pd.DataFrame({"arm": arm, "basket": baskets, "city": cities})


def _flat(basket_p):
    """A city-independent basket map (basket ⟂ city)."""
    return {city: basket_p for city in CITIES}


def _experiment(control_df, test_df, intended=INTENDED_SPLIT):
    data = pd.concat([control_df, test_df], ignore_index=True)
    return Experiment(data, intended)


# --- the experiment builders ------------------------------------------------

def _b_win(rng):
    n = _jit(4000, rng)                 # balanced -> SRM passes exactly
    # the new layout genuinely nudges users into Large/Premium.
    tp = np.array([0.34, 0.27, 0.21, 0.16, 0.02])
    c = _draw_arm(n, "control", _flat(_BP), _CP, rng)
    t = _draw_arm(n, "test", _flat(tp), _CP, rng)
    return _experiment(c, t)


def _b_trivial(rng):
    # n must be LARGE enough that even a 5th-percentile Cramer's V (~0.004)
    # still gives p ~ 1e-7: the lesson is "p ~ 0 yet V ~ 0", so the rejection
    # has to be guaranteed. 500k/arm doubles chi^2 vs 250k without touching V.
    n = _jit(500_000, rng)
    # a real but microscopic nudge: +0.3pp into Premium.
    tp = np.array([0.447, 0.30, 0.155, 0.083, 0.015])
    c = _draw_arm(n, "control", _flat(_BP), _CP, rng)
    t = _draw_arm(n, "test", _flat(tp), _CP, rng)
    return _experiment(c, t)


def _b_srm(rng):
    n_c = _jit(4000, rng)
    n_t = int(round(n_c * rng.uniform(0.90, 0.94)))   # ~6-10% of test missing
    # the dropped test users were disproportionately low-spenders, so the
    # SURVIVING test sample looks richer — a textbook SRM trap. The magnitude
    # of the shortfall is jittered so students cannot share one answer.
    tp = np.array([0.34, 0.28, 0.20, 0.16, 0.02])
    c = _draw_arm(n_c, "control", _flat(_BP), _CP, rng)
    t = _draw_arm(n_t, "test", _flat(tp), _CP, rng)
    return _experiment(c, t)


def _b_pilot(rng):
    n = 15                              # micro-pilot in a brand-new city
    # a real but modest nudge (Premium+VIP 0.095 -> 0.15). A properly powered
    # test would catch it; 15 orders per arm cannot — Fisher rejects < 2% of
    # the time, so the honest read is "not enough data", not "no effect".
    tp = np.array([0.40, 0.29, 0.16, 0.12, 0.03])
    c = _draw_arm(n, "control", _flat(_BP), _CP, rng)
    t = _draw_arm(n, "test", _flat(tp), _CP, rng)
    return _experiment(c, t)


def _b_rare(rng):
    n = _jit(200, rng)
    # NO real effect: both arms share the base distribution, including the rare
    # VIP tier (~1.5%). At n~200, E(VIP) ~ 3.0 < 5 reliably, so the 5-tier
    # chi-square is INVALID (its p-value is not to be trusted, whether or not
    # this particular table rejects). Merging VIP into Premium gives a valid
    # test that correctly finds nothing.
    c = _draw_arm(n, "control", _flat(_BP), _CP, rng)
    t = _draw_arm(n, "test", _flat(_BP), _CP, rng)
    return _experiment(c, t)


def _b_confound(rng):
    n = _jit(4000, rng)                # overall split is honest 50/50
    # baskets depend on CITY (same map for both arms => no treatment effect),
    # but the test arm heavily over-samples Lviv (a Premium-heavy city). Pooled,
    # test looks like a win; stratified by city, nothing changed. Lviv is pushed
    # to 0.50 (from 0.20) so the pooled test rejects > 99% of the time.
    cp_test = np.array([0.25, 0.50, 0.15, 0.10])   # Lviv strongly over-represented
    c = _draw_arm(n, "control", _BASKET_BY_CITY, _CP, rng)
    t = _draw_arm(n, "test", _BASKET_BY_CITY, cp_test, rng)
    return _experiment(c, t)


_BUILD = {
    "win": _b_win,
    "trivial": _b_trivial,
    "srm": _b_srm,
    "pilot": _b_pilot,
    "rare": _b_rare,
    "confound": _b_confound,
}

# Per-world: the decision the lab expects, and the decisive categorical evidence.
_EXPECTED = {
    "win": (
        "SHIP — genuine, business-meaningful basket shift",
        "SRM passes (arm sizes ~50/50). The homogeneity chi-square rejects, "
        "Cramer's V is non-trivial (~0.15-0.25, weak-moderate), and the "
        "standardized residuals show test users moving up into Large/Premium. "
        "No city confound (arm x city is balanced). The Premium+Large share "
        "rises by a meaningful number of points."),
    "trivial": (
        "DON'T SHIP — significant but trivial",
        "SRM passes. With n ~ 500k/arm the homogeneity chi-square gives "
        "p << 0.05 (typically ~1e-3 on a low-V seed down to ~1e-7), but "
        "Cramer's V ~ 0.005-0.01 ('practically none'). The Premium share moves "
        "a fraction of a point. Classic big-n trap: significance is guaranteed "
        "and says nothing about strength."),
    "srm": (
        "FIX THE SPLITTER — result is void",
        "A chi-square goodness-of-fit of the delivered arm sizes against the "
        "intended 50/50 REJECTS (test arm is ~7-8% short). The basket "
        "'win' you would otherwise report is an artifact of differential "
        "dropout (low-spenders missing from test). No effect can be trusted "
        "until the allocation is fixed."),
    "pilot": (
        "INSUFFICIENT EVIDENCE — collect more",
        "Only 15 vs 15 orders: expected counts are ~1, and roughly half the "
        "time at least one basket column is all-zero, so the 5-tier "
        "chi2_contingency RAISES and cannot be run at all. Collapse to a "
        "business 2x2 (e.g. Premium-or-not) and use Fisher's exact test: it is "
        "NOT significant for ~97% of seeds. A modest real nudge exists "
        "underneath (Premium+VIP 0.095 -> 0.15), but 15/arm cannot detect it. "
        "NB a large Cramer's V here (~0.3) is small-sample inflation, not a "
        "real effect; the naive chi-square's over-rejection is a Monte-Carlo "
        "(FPR) fact, not something a single 15-vs-15 table displays."),
    "rare": (
        "NO SHIPPABLE EFFECT — and the raw 5-tier test is statistically invalid",
        "True A/A: both arms share the base mix, so there is no real effect. "
        "The teaching point is the expected-count rule — at n~200 the rare VIP "
        "tier has E(VIP) ~ 3 < 5, so the 5-tier chi-square's p-value is NOT "
        "trustworthy (this particular table may or may not reject by chance — "
        "that unpredictability IS the problem). The correct handling is to "
        "MERGE VIP into Premium (or run a Monte-Carlo p-value); the valid test "
        "then clearly finds nothing (p large, V ~ 0). The over-rejection of "
        "the invalid test is demonstrated in the Monte-Carlo FPR section (make "
        "VIP rare at small n and watch FPR leave alpha), not from one table."),
    "confound": (
        "DON'T TRUST — city confound (covariate imbalance)",
        "Overall SRM passes (50/50), and the pooled homogeneity chi-square "
        "rejects (>99% of seeds) with a real-looking Cramer's V. BUT arm x "
        "city is strongly dependent (test heavily over-sampled Lviv, a "
        "Premium-heavy city). Stratify by city: there is NO real within-city "
        "effect (the same basket map drives both arms), so the within-city "
        "tables show no consistent shift — a lone city dipping below 0.05 is "
        "multiplicity across four tests (think Bonferroni), not signal. The "
        "pooled 'win' is covariate imbalance, not the layout — fix the split "
        "/ control for city."),
}


# Rough effort tier per world. The worlds are NOT equally hard: a clean "win"
# is a short, direct ship verdict, whereas "pilot" (sparse table -> collapse to
# 2x2 -> Fisher -> small-n caveats) and "confound" (independence + stratify +
# multiplicity) take materially more work. Surfaced by grade_chi.py so the
# verdict can be graded **difficulty-aware** — the salt removes cross-lab
# correlation, but within a single lab the draws still differ in effort, and an
# equal-points rubric should account for that.
_DIFFICULTY = {
    "win": "easy",
    "trivial": "medium",
    "srm": "medium",
    "rare": "medium",
    "pilot": "hard",
    "confound": "hard",
}


def _reveal(student_id):
    """INSTRUCTOR ONLY. Return the hidden ground truth for ``student_id``."""
    world = _world(student_id)
    decision, evidence = _EXPECTED[world]
    return {
        "world": world,
        "difficulty": _DIFFICULTY[world],
        "decision": decision,
        "evidence": evidence,
    }
