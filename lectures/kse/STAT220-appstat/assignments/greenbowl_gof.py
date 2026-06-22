"""
greenbowl_gof.py — STAT220 *Applied Statistics*, Lab 6 data platform.

Lab 4 handed you a fixed A/B dataset and asked for a business decision.
Lab 5 made you audit the *criterion* that makes those decisions.
**Lab 6 is distribution forensics: the mean-based verdict is not enough.**

The GreenBowl experimentation platform reports a clean ``t``-test result for
every A/B test — "ship / don't ship the new menu". But a ``t``-test only
looks at the **mean**. Some experiments hide a problem the mean cannot see: a
randomization bug, an effect that lives in the *shape* of the distribution, a
wave of churned (zero-spend) users, a metric that is simply not normal on a
small sample, or a *paired* design quietly analysed as two independent groups.

Every student receives a **different** hidden experiment, deterministically
derived from their student id. Your job is to **diagnose what is really going
on** with the goodness-of-fit toolkit from the lecture — normality (Shapiro +
QQ), the one/two-sample Kolmogorov(-Smirnov) test, the empirical CDF, and a
Monte-Carlo validation of your own guardrail — and to deliver the verdict the
mean alone would have got wrong.

Fixed context (the same for everyone, stated in the lab):

    * metric                ARPU (weekly revenue per user, UAH) — skewed
    * baseline ARPU         mu0    = 250 UAH
    * significance level    alpha  = 0.05

Usage (students):

    from greenbowl_gof import (load_my_experiment, load_my_log,
                               arpu_sample, BASELINE, ALPHA)

    exp = load_my_experiment("your.email@kse.org.ua")
    exp.control, exp.test            # ARPU during the experiment
    exp.pre_control, exp.pre_test    # ARPU during the pre-period (no treatment)
    exp.uid_control, exp.uid_test    # the user ids behind each arm

    log = load_my_log("your.email@kse.org.ua")   # historical ARPU log

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

__all__ = ["load_my_experiment", "load_my_log", "arpu_sample",
           "Experiment", "BASELINE", "ALPHA"]

# --- public, documented constants -------------------------------------------
BASELINE = 250.0      # mu0, historical ARPU in UAH
ALPHA = 0.05          # significance level


def arpu_sample(true_mean, n, sigma=0.6, rng=None):
    """A lognormal ARPU sample with a given **mean** ``true_mean``.

    ``sigma`` controls the spread / tail heaviness *without* shifting the
    mean (E[X] = ``true_mean`` for any ``sigma``). This is the same skewed
    "money" generator you used in Practices 5–6; it is your "omniscient god"
    tool — use it to manufacture H0 and H1 when you validate your guardrail.

    Pass ``rng`` (a ``np.random.default_rng(...)``) for reproducibility, or
    leave it ``None`` to use the global NumPy RNG (so ``np.random.seed`` still
    works).
    """
    mu = np.log(true_mean) - sigma ** 2 / 2
    if rng is None:
        return np.random.lognormal(mean=mu, sigma=sigma, size=n)
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


class Experiment:
    """One hidden A/B experiment, keyed to a student id.

    Attributes
    ----------
    control, test : np.ndarray
        ARPU of each arm **during** the experiment.
    pre_control, pre_test : np.ndarray
        ARPU of each arm **before** the experiment started (the pre-period,
        when no treatment can have had any effect yet). A clean A/A guardrail
        on the pre-period is a necessary condition for trusting the result.
    uid_control, uid_test : np.ndarray
        The user ids behind each arm. Whether the two arms share users is
        something you may need to check — it decides whether a two-sample test
        is even applicable.
    """

    def __init__(self, control, test, pre_control, pre_test,
                 uid_control, uid_test):
        self.control = np.asarray(control, float)
        self.test = np.asarray(test, float)
        self.pre_control = np.asarray(pre_control, float)
        self.pre_test = np.asarray(pre_test, float)
        self.uid_control = np.asarray(uid_control)
        self.uid_test = np.asarray(uid_test)

    @property
    def n_control(self):
        return len(self.control)

    @property
    def n_test(self):
        return len(self.test)

    def __repr__(self):
        return (f"<Experiment n_control={self.n_control} "
                f"n_test={self.n_test} "
                f"pre=({len(self.pre_control)},{len(self.pre_test)})>")


def load_my_experiment(student_id):
    """Return **your** hidden :class:`Experiment`, keyed to ``student_id``.

    The arms are ARPU samples; what (if anything) is wrong with the experiment
    is **hidden**. Diagnose it with the goodness-of-fit toolkit — do not try to
    read it off the source.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    rng = np.random.default_rng(_seed_from_id(student_id))
    return _BUILD[_world(student_id)](rng)


def load_my_log(student_id):
    """Return a 'historical' ARPU log for ``student_id`` (city x cuisine).

    A ``pandas.DataFrame`` with columns ``city``, ``cuisine``, ``arpu`` —
    real-looking GreenBowl logs you can slice and run many A/A tests on when
    you validate your guardrail on real data. Unique to your id.
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

# Six hidden experiment "worlds". A correct lab reaches a *different* diagnosis
# for each, and the only way to tell them apart is with the GOF toolkit:
#
#   clean   honest split, genuine +10% mean lift     -> SHIP (mean win): pre
#           A/A passes; both the t-test and KS reject  on the experiment.
#   shape   same mean, treatment WIDENS the spread   -> INVESTIGATE THE SHAPE:
#           (sigma 0.6 -> 1.0)                          t-test is silent, KS
#                                                       rejects (effect is in
#                                                       the distribution, not
#                                                       the mean).
#   srm     randomization bug present already in the  -> DON'T TRUST THE RESULT:
#           pre-period (test arm heavier-tailed)        the pre-period A/A (KS)
#                                                       already rejects -> the
#                                                       split is broken.
#   zeros   ~8% churned (zero-spend) users in test,   -> INVESTIGATE CHURN:
#           higher spend among the payers              the arithmetic mean is
#                                                       ~flat, but KS + the ECDF
#                                                       jump at 0 (and a z-test
#                                                       on the share of zeros)
#                                                       expose zero-inflation.
#   paired  control/test are BEFORE/AFTER on the SAME -> WRONG METHOD: the arms
#           users (shared uids), real within-user lift  share users -> not
#                                                       independent; two-sample
#                                                       KS is conservative and
#                                                       misses it; per-user
#                                                       differences reveal a
#                                                       clear effect.
#   skew    heavy skew on a SMALL, UNBALANCED design  -> USE THE LOG SCALE: raw
#           (n_control=60, n_test=12), +100% lift       ARPU fails Shapiro / the
#                                                       QQ curves AND the raw
#                                                       small/unbalanced t-test
#                                                       is miscalibrated; log
#                                                       (ARPU) is normal, the
#                                                       log Welch is calibrated
#                                                       and shows a clear win.
_WORLDS = ("clean", "shape", "srm", "zeros", "paired", "skew")


def _seed_from_id(student_id):
    """Deterministic 32-bit seed from a normalised id."""
    norm = student_id.strip().lower().encode("utf-8")
    return int(hashlib.sha256(norm).hexdigest()[:8], 16)


def _world(student_id):
    """Pick the hidden experiment world for this id."""
    return _WORLDS[_seed_from_id(student_id) % len(_WORLDS)]


# --- the experiment builders ------------------------------------------------

def _indep(control, test, pre_control, pre_test):
    """Wrap two INDEPENDENT arms (disjoint user ids)."""
    nc, nt = len(control), len(test)
    uid_c = np.arange(nc)
    uid_t = np.arange(nc, nc + nt)
    return Experiment(control, test, pre_control, pre_test, uid_c, uid_t)


def _b_clean(rng):
    n = 1500
    pc = arpu_sample(250, n, 0.6, rng)
    pt = arpu_sample(250, n, 0.6, rng)
    c = arpu_sample(250, n, 0.6, rng)
    t = arpu_sample(275, n, 0.6, rng)          # genuine +10% mean lift
    return _indep(c, t, pc, pt)


def _b_shape(rng):
    n = 1800
    pc = arpu_sample(250, n, 0.6, rng)
    pt = arpu_sample(250, n, 0.6, rng)
    c = arpu_sample(250, n, 0.6, rng)
    t = arpu_sample(250, n, 1.0, rng)          # SAME mean, wider spread
    return _indep(c, t, pc, pt)


def _b_srm(rng):
    n = 1600
    pc = arpu_sample(250, n, 0.6, rng)
    pt = arpu_sample(250, n, 0.95, rng)        # bug already in the pre-period
    c = arpu_sample(250, n, 0.6, rng)
    t = arpu_sample(250, n, 0.95, rng)
    return _indep(c, t, pc, pt)


def _b_zeros(rng):
    n = 1700
    churn_rate = 0.08
    pc = arpu_sample(250, n, 0.6, rng)
    pt = arpu_sample(250, n, 0.6, rng)
    c = arpu_sample(250, n, 0.6, rng)
    # payers spend exactly enough more that the ~8% churn-to-zero leaves the
    # ARPU mean flat at 250 -> the t-test on the mean stays silent by design.
    payers = arpu_sample(250 / (1 - churn_rate), n, 0.6, rng)
    churn = rng.random(n) < churn_rate
    t = np.where(churn, 0.0, payers)
    return _indep(c, t, pc, pt)


def _b_paired(rng):
    n = 1200
    base = arpu_sample(250, n, 0.55, rng)              # the user's own ARPU
    before = base * np.exp(rng.normal(0.00, 0.15, n))  # week before
    after = base * np.exp(rng.normal(0.04, 0.15, n))   # +~4% within-user lift
    pre_b = base * np.exp(rng.normal(0.00, 0.15, n))   # an even earlier week
    pre_a = base * np.exp(rng.normal(0.00, 0.15, n))
    uid = np.arange(n)
    # SAME user ids in both arms -> dependent (before/after), not two groups.
    return Experiment(before, after, pre_b, pre_a, uid, uid)


def _b_skew(rng):
    # Heavy skew on a SMALL, UNBALANCED design: balance would let Welch cancel
    # the skew in the numerator (so a balanced raw test stays calibrated). The
    # imbalance is what makes the raw two-sample t-test miscalibrated, so the
    # log scale is genuinely the right call here, not just cosmetics.
    nc, nt = 60, 12
    pc = arpu_sample(250, nc, 0.7, rng)
    pt = arpu_sample(250, nt, 0.7, rng)
    c = arpu_sample(250, nc, 0.7, rng)
    t = arpu_sample(500, nt, 0.7, rng)         # genuine +100% lift, small arm
    return _indep(c, t, pc, pt)


_BUILD = {
    "clean": _b_clean,
    "shape": _b_shape,
    "srm": _b_srm,
    "zeros": _b_zeros,
    "paired": _b_paired,
    "skew": _b_skew,
}

# Per-world: the diagnosis the lab expects, and the decisive GOF evidence.
_EXPECTED = {
    "clean": (
        "SHIP — genuine mean lift",
        "Pre-period two-sample KS passes (honest split). On the experiment "
        "both the t-test AND the KS reject: a real ~+10% ARPU lift. The "
        "distribution shape is otherwise unchanged."),
    "shape": (
        "INVESTIGATE THE SHAPE — effect is NOT in the mean",
        "Pre-period KS passes. On the experiment the t-test is silent (means "
        "are equal) but KS rejects: the treatment widened the distribution "
        "(sigma 0.6 -> 1.0). The ECDFs cross; mean-only reporting misses it."),
    "srm": (
        "DON'T TRUST THE RESULT — randomization is broken",
        "The PRE-period two-sample KS already rejects (test arm is heavier-"
        "tailed before any treatment). No effect can exist pre-period, so this "
        "is a splitter / SRM-by-shape bug; the experiment verdict is void "
        "until the randomization is fixed."),
    "zeros": (
        "INVESTIGATE CHURN — zero-inflation hidden by a flat mean",
        "Pre-period KS passes. The arithmetic mean is ~flat, so the t-test "
        "sees little, but KS rejects and the ECDF has a jump at 0; a z-test on "
        "the share of zero-spend users is highly significant (~8% vs ~0%). The "
        "treatment churned users while raising spend among the payers."),
    "paired": (
        "WRONG METHOD — samples are dependent (before/after)",
        "uid_control and uid_test are identical -> the arms are the SAME users "
        "before/after, not two independent groups. Naive two-sample KS is "
        "conservative and misses the effect; per-user differences d = after - "
        "before (or on the log scale) show a clear ~+4% within-user lift."),
    "skew": (
        "USE THE LOG SCALE — raw metric non-normal on a small, unbalanced sample",
        "Heavy skew on a small, unbalanced design (n_control=60, n_test=12): raw "
        "ARPU fails Shapiro and the QQ-plot curves, while log(ARPU) is normal. "
        "On this design the raw two-sample t-test is MISCALIBRATED (a Monte-Carlo "
        "A/A gives FPR ~0.08 > alpha two-sided, and it is conservative & under-"
        "powered one-sided), whereas the log-scale Welch test is calibrated "
        "(~0.05) and more powerful; on the log scale it shows a clear geometric-"
        "mean win (~+100% on the raw scale). Balance would have let Welch cancel "
        "the skew, so the imbalance is the point."),
}


def _reveal(student_id):
    """INSTRUCTOR ONLY. Return the hidden ground truth for ``student_id``."""
    world = _world(student_id)
    diagnosis, evidence = _EXPECTED[world]
    return {
        "world": world,
        "diagnosis": diagnosis,
        "evidence": evidence,
    }
