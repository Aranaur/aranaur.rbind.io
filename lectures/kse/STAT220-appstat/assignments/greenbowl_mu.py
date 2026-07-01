"""
greenbowl_mu.py — STAT220 *Applied Statistics*, Lab 8 data platform.

Lab 4 handed you a fixed A/B dataset and asked for a business decision.
Lab 5 made you audit the *criterion*. Lab 6 was distribution forensics on a
numeric metric. Lab 7 was categorical A/B forensics. **Lab 8 returns to a
numeric metric — delivery time — and takes away the crutch: the mean (and the
t-test built on it) can no longer be trusted, and the platform's headline
verdict is exactly that t-test.**

The GreenBowl experimentation platform ran an A/B test of a new **courier
dispatcher**. For every order it logged the **delivery time in minutes**
(lower = better) and the **courier** who carried it. Delivery time is heavily
right-skewed — most orders arrive in half an hour, a few get stuck in traffic
for hours. The automated report prints a tidy **Welch t-test on the means**
and a "ship / don't ship" line. Your job is to decide whether you can
**believe** it, using the nonparametric toolkit from the lecture: the sign
test, the Wilcoxon signed-rank test, Mann–Whitney, the probability of
superiority P(A<B), the Hodges–Lehmann shift with its CI, Brunner–Munzel, and
a Monte-Carlo validation of whatever criterion your verdict rests on.

Every student receives a **different** hidden experiment, deterministically
derived from their student id. Some got a genuine, business-meaningful
speed-up; some an honest null; some a mean "win" manufactured by a handful of
stuck-order **outliers**; some a microscopic shift made "significant" by an
enormous sample; some a dispatcher that keeps the **median** but wrecks the
**tail**; and some a **crossover** design where the same couriers appear in
both arms — so every independent-samples test is invalid from the start. The
only honest way to find out which one you hold is to run the diagnostics.

Fixed context (the same for everyone, stated in the lab):

    * unit                  one order
    * metric                delivery time, minutes (lower = better)
    * arms                  control = old dispatcher, test = new dispatcher
    * SLA                   the contract promises a median <= SLA_MIN minutes
    * business threshold    a median speed-up below BUSINESS_MDE minutes is
                            not worth the rollout risk
    * significance level    alpha = 0.05

Usage (students):

    from greenbowl_mu import load_my_experiment, ALPHA, SLA_MIN, BUSINESS_MDE

    exp = load_my_experiment("your.email@kse.org.ua")
    exp.data            # tidy DataFrame: one row per order
                        #   [courier_id, arm, minutes]
    exp.n_control, exp.n_test
    exp.control         # np.array of control delivery times, minutes
    exp.test            # np.array of test delivery times, minutes

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
           "ALPHA", "SLA_MIN", "BUSINESS_MDE"]

# --- public, documented constants -------------------------------------------
ALPHA = 0.05          # significance level
SLA_MIN = 45          # contractual SLA: median delivery time <= 45 min
BUSINESS_MDE = 3      # a median speed-up below 3 min is not worth shipping


class Experiment:
    """One hidden delivery-time A/B experiment, keyed to a student id.

    Attributes
    ----------
    data : pandas.DataFrame
        One row per order, columns ``courier_id`` (who carried the order),
        ``arm`` ("control"/"test") and ``minutes`` (delivery time, lower =
        better). What the ``courier_id`` column can tell you about the
        experiment's *design* is part of the lab.
    """

    def __init__(self, data):
        self.data = data.reset_index(drop=True)

    @property
    def n_control(self):
        return int((self.data["arm"] == "control").sum())

    @property
    def n_test(self):
        return int((self.data["arm"] == "test").sum())

    @property
    def control(self):
        """Delivery times (minutes) of the control arm, as an array."""
        d = self.data
        return d.loc[d["arm"] == "control", "minutes"].to_numpy()

    @property
    def test(self):
        """Delivery times (minutes) of the test arm, as an array."""
        d = self.data
        return d.loc[d["arm"] == "test", "minutes"].to_numpy()

    def __repr__(self):
        return (f"<Experiment n_control={self.n_control} "
                f"n_test={self.n_test} metric=delivery minutes>")


def load_my_experiment(student_id):
    """Return **your** hidden :class:`Experiment`, keyed to ``student_id``.

    What (if anything) is wrong with the experiment is **hidden**. Diagnose it
    with the nonparametric toolkit — do not try to read it off the source.
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
# for each, and the only way to tell them apart is with the nonparametric
# toolkit:
#
#   win       honest independent A/B, the new    -> SHIP: Welch t and MW both
#             dispatcher genuinely cuts the         reject, P(test<control) ~
#             median 40 -> 34 min                   0.58, HL shift ~ -6 min
#                                                   with a CI clear of 0 and
#                                                   past BUSINESS_MDE.
#   null      honest A/A, no effect; minutes     -> NO EFFECT (a valid, useful
#             logged as whole numbers -> heavy      finding): every test keeps
#             ties                                  H0; the MC power section
#                                                   says what WAS detectable.
#   outlier   no real effect, but ~2% of         -> DON'T SHIP (mean artifact):
#             control orders got stuck for          Welch t screams "test is
#             hours; the control MEAN inflates      faster"; MW / HL see
#                                                   nothing. Ranks tell truth.
#   trivial   n ~ 150k/arm, a real but           -> DON'T SHIP (trivial):
#             microscopic ~0.3-min speed-up         MW p ~ 0 but P(t<c) ~ 0.503
#                                                   and HL ~ -0.3 min << 3-min
#                                                   business threshold.
#   shape     medians EQUAL, but the new         -> RELIABILITY REGRESSION:
#             dispatcher batches orders — some      Welch t rejects ("slower on
#             very early, some very late; the       average!"), MW keeps H0
#             mean rises with the tail              (P(t<c) = 0.5). Neither
#                                                   headline is right: the P90
#                                                   tail explodes. Quantiles /
#                                                   ECDF settle it.
#   paired    crossover: the SAME ~26 couriers   -> SHIP, but only the PAIRED
#             ride a week on each dispatcher;      path shows it: pooled t and
#             big between-courier spread hides      MW keep H0; per-courier
#             a real ~7% within-courier             differences + sign /
#             speed-up                              signed-rank clearly reject.
_WORLDS = ("win", "null", "outlier", "trivial", "shape", "paired")


# A per-lab salt makes THIS lab's world assignment statistically INDEPENDENT of
# every other lab's. Without it, world = hash(id) % 6 is identical across any
# two labs that share the hash and have 6 worlds (e.g. Lab 7 / greenbowl_chi),
# so a student who lands on a "hard" index stays on it in every lab — the
# "some people always get the hard scenario" complaint. EACH NEW LAB MUST USE
# ITS OWN DISTINCT SALT so luck does not carry over across the course.
_LAB_SALT = "stat220-lab8-mu-v1"


def _seed_from_id(student_id):
    """Deterministic 32-bit seed from a normalised id (drives the DATA sample)."""
    norm = student_id.strip().lower().encode("utf-8")
    return int(hashlib.sha256(norm).hexdigest()[:8], 16)


def _jit(n, rng, frac=0.03):
    """Jitter an arm size by +/- frac so the absolute size cannot fingerprint
    a world across students. Left untouched for tiny samples (n < 50)."""
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


def _delivery(median, sigma, n, rng):
    """Delivery time (min) — right-skewed: lognormal with a given median."""
    return rng.lognormal(np.log(median), sigma, n)


def _ids(n, rng, width=7):
    """n unique courier ids, drawn from a wide range so neither the range nor
    the ordering fingerprints an arm or a world."""
    pool = rng.choice(10 ** width, size=n, replace=False)
    return np.array([f"c_{i:0{width}d}" for i in pool])


def _frame(ids, arm, minutes, decimals=1):
    minutes = np.round(np.maximum(np.asarray(minutes, float), 1.0), decimals)
    return pd.DataFrame({"courier_id": ids, "arm": arm, "minutes": minutes})


def _experiment(control_df, test_df):
    data = pd.concat([control_df, test_df], ignore_index=True)
    return Experiment(data)


# --- the experiment builders ------------------------------------------------

def _b_win(rng):
    n_c, n_t = _jit(600, rng), _jit(600, rng)
    # the new dispatcher genuinely cuts the median 40 -> 34 min (same shape).
    c = _delivery(40, 0.55, n_c, rng)
    t = _delivery(34, 0.55, n_t, rng)
    ids = _ids(n_c + n_t, rng)
    return _experiment(_frame(ids[:n_c], "control", c),
                       _frame(ids[n_c:], "test", t))


def _b_null(rng):
    n_c, n_t = _jit(700, rng), _jit(700, rng)
    # honest A/A: both arms share the law. Minutes are logged as WHOLE numbers
    # (a real logging pipeline quirk) -> heavy ties, which the rank tests must
    # handle via the tie correction. There is nothing to find — and saying so,
    # with a power analysis of what WAS detectable, is the correct verdict.
    c = _delivery(40, 0.55, n_c, rng)
    t = _delivery(40, 0.55, n_t, rng)
    ids = _ids(n_c + n_t, rng)
    return _experiment(_frame(ids[:n_c], "control", c, decimals=0),
                       _frame(ids[n_c:], "test", t, decimals=0))


def _b_outlier(rng):
    n_c, n_t = _jit(500, rng), _jit(500, rng)
    # NO real effect — both arms share the law. But ~2% of CONTROL orders got
    # stuck (courier app crash, order re-dispatched hours later): multiply a
    # fixed handful by 10-22x. The control MEAN inflates by ~10 min, so the
    # headline Welch t "discovers" that the new dispatcher is faster. The
    # deterministic outlier count keeps the signature stable across seeds.
    c = _delivery(38, 0.45, n_c, rng)
    t = _delivery(38, 0.45, n_t, rng)
    k = max(6, int(round(0.02 * n_c)))
    idx = rng.choice(n_c, size=k, replace=False)
    c[idx] *= rng.uniform(10, 22, size=k)
    ids = _ids(n_c + n_t, rng)
    return _experiment(_frame(ids[:n_c], "control", c),
                       _frame(ids[n_c:], "test", t))


def _b_trivial(rng):
    # n must be LARGE enough that a ~0.75% median shift is all but guaranteed
    # to reject (the lesson is "p ~ 0 yet the effect is ~0.3 min", so the
    # rejection must be reliable): at 150k/arm the MW z is ~ 4.
    n_c, n_t = _jit(150_000, rng), _jit(150_000, rng)
    c = _delivery(40.0, 0.5, n_c, rng)
    t = _delivery(39.7, 0.5, n_t, rng)   # real but microscopic speed-up
    ids = _ids(n_c + n_t, rng, width=8)
    return _experiment(_frame(ids[:n_c], "control", c),
                       _frame(ids[n_c:], "test", t))


def _b_shape(rng):
    # medians EQUAL (40 vs 40), but the new dispatcher batches orders: far
    # more spread (sigma 0.2 -> 0.75). The lognormal MEAN rises with sigma, so
    # Welch t reliably says "test is SLOWER on average — roll back", while
    # Mann-Whitney sees P(t<c) = 0.5 and keeps H0. Neither is the right read:
    # the median user is unaffected but the P90 tail (and SLA breaches)
    # explode. Arms are UNBALANCED so the practice's MW-variance caveat about
    # unequal shapes + unequal n applies to the student's own data.
    n_c, n_t = _jit(150, rng), _jit(600, rng)
    c = _delivery(40, 0.2, n_c, rng)
    t = _delivery(40, 0.75, n_t, rng)
    ids = _ids(n_c + n_t, rng)
    return _experiment(_frame(ids[:n_c], "control", c),
                       _frame(ids[n_c:], "test", t))


def _b_paired(rng):
    n = 26
    # CROSSOVER: the same n couriers ride a week on the old dispatcher and a
    # week on the new one. Couriers differ a lot from each other (sigma 0.45
    # between couriers), but each courier is consistent with themselves
    # (sigma 0.06 within), and the new dispatcher makes each courier ~7%
    # faster. Pooled independent tests are (a) invalid — the arms share the
    # couriers — and (b) blind: the between-courier spread swamps the shift.
    # The paired path (per-courier differences -> sign / signed-rank) sees it.
    base = _delivery(40, 0.45, n, rng)                       # courier baselines
    c = base * rng.lognormal(0.0, 0.06, n)                   # old dispatcher
    t = base * rng.lognormal(np.log(0.93), 0.06, n)          # new: ~7% faster
    ids = _ids(n, rng, width=3)
    return _experiment(_frame(ids, "control", c),
                       _frame(ids, "test", t))


_BUILD = {
    "win": _b_win,
    "null": _b_null,
    "outlier": _b_outlier,
    "trivial": _b_trivial,
    "shape": _b_shape,
    "paired": _b_paired,
}

# Per-world: the decision the lab expects, and the decisive evidence.
_EXPECTED = {
    "win": (
        "SHIP — genuine, business-meaningful speed-up",
        "Independent design (no courier appears twice). Both Welch t and "
        "Mann-Whitney reject; P(test<control) ~ 0.58 and the Hodges-Lehmann "
        "shift is ~ -6 min with a 95% CI clear of zero and past the 3-min "
        "business threshold. The distributions have the same shape, so the "
        "MW rejection reads cleanly as a median shift."),
    "null": (
        "NO DETECTABLE EFFECT — and that is a valid, reportable finding",
        "Honest A/A on ~500/arm. Welch t, Mann-Whitney and Brunner-Munzel all "
        "keep H0; P(test<control) ~ 0.5 and the HL CI covers 0. Minutes are "
        "logged as integers -> heavy ties, handled by the tie correction "
        "(scipy does it automatically). The honest deliverable is the MC "
        "power curve: what median shift WOULD have been detectable at this n "
        "(a ~3-min shift is comfortably within reach, so 'no effect' is "
        "informative, not just 'not enough data')."),
    "outlier": (
        "DON'T SHIP — the mean 'win' is an outlier artifact",
        "No real effect: both arms share the law, but ~2% of CONTROL orders "
        "are stuck-order outliers (hours long). The headline Welch t rejects "
        "and claims the new dispatcher is faster — the control MEAN is "
        "hostage to the tail. Mann-Whitney keeps H0 (P(t<c) ~ 0.49), the HL "
        "shift is ~ 0 min. Removing or winsorising the outliers kills the t "
        "verdict too. Ranks tell the truth; the platform's mean-based "
        "headline is the bug."),
    "trivial": (
        "DON'T SHIP — significant but trivial",
        "n ~ 150k/arm. Mann-Whitney (and Welch t) give p << 0.05 — a real "
        "shift exists — but P(test<control) ~ 0.503 and the Hodges-Lehmann "
        "shift is ~ -0.3 min: a tenth of the 3-min business threshold. "
        "Classic big-n trap: significance is guaranteed at this sample size "
        "and says nothing about business relevance. (HL on 150k x 150k pairs "
        "must be computed on a random subsample.)"),
    "shape": (
        "RELIABILITY REGRESSION — same median, exploded tail; don't read "
        "either headline",
        "Medians are equal (~40 vs ~40) and P(test<control) = 0.5, so "
        "Mann-Whitney keeps H0; but Welch t REJECTS because the lognormal "
        "mean rises with the spread — the platform prints 'slower on "
        "average, roll back'. Neither read is right: the new dispatcher "
        "keeps the median but batches orders, so the P90 tail (~50 -> ~100+ "
        "min) and SLA breaches explode. ECDF / quantile table settles it. "
        "Also the arms are unbalanced with unequal shapes — exactly the "
        "practice's warning zone for MW's calibration; Brunner-Munzel is the "
        "honest rank test here (it also keeps H0: stochastic equality holds)."),
    "paired": (
        "SHIP — but only the paired analysis can see it",
        "The SAME ~26 couriers appear in both arms (crossover) — check "
        "courier_id! Every independent-samples test is invalid by design, "
        "and also blind: pooled Welch t and MW keep H0 because the "
        "between-courier spread swamps the ~7% within-courier speed-up. "
        "Collapse to per-courier differences: ~75-80% of couriers got "
        "faster; the sign test and Wilcoxon signed-rank both reject (check "
        "the symmetry of the differences before trusting signed-rank). The "
        "median per-courier saving is ~2.5-3 min — read it against the "
        "3-min threshold honestly."),
}


# Rough effort tier per world. The worlds are NOT equally hard: a clean "win"
# is a short, direct ship verdict, whereas "shape" (two wrong headlines ->
# quantile forensics -> BM) and "paired" (design detection -> differences ->
# symmetry -> sign/signed-rank) take materially more work. Surfaced by
# grade_mu.py so the verdict can be graded **difficulty-aware** — the salt
# removes cross-lab correlation, but within a single lab the draws still
# differ in effort, and an equal-points rubric should account for that.
_DIFFICULTY = {
    "win": "easy",
    "null": "easy",
    "outlier": "medium",
    "trivial": "medium",
    "shape": "hard",
    "paired": "hard",
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
