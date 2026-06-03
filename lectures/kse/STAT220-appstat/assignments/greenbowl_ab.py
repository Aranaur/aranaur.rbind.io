"""
greenbowl_ab.py — STAT220 *Applied Statistics*, Lab 4 data platform.

A tiny simulation of a *live* A/B test for the **GreenBowl** food-delivery
app. Every student receives a **unique** dataset, deterministically derived
from their student id. The true data-generating parameters — the effect
size, the variances, the group sizes, the skewness, and even *whether there
is any effect at all* — are **HIDDEN**. Your job in Lab 4 is to recover the
business conclusion from the data alone, exactly as you would on a real
product team where nobody hands you the ground truth.

Fixed business context (the same for everyone, stated in the lab):

    * baseline ARPU            mu0          = 250 UAH  (3 months of history)
    * break-even               +15 UAH      (DECISION threshold: a smaller
                                             lift is not worth shipping)
    * design MDE               +25 UAH      (PLANNING threshold: the "clear
                                             win" experiments are sized for)
    * significance level       alpha        = 0.05
    * direction                one-sided "greater" (ship only if test > control)

Usage (students):

    from greenbowl_ab import load_my_ab
    control, test = load_my_ab("your.email@kse.org.ua")

    control -> weekly ARPU (UAH) for the OLD menu (cohort A)
    test    -> weekly ARPU (UAH) for the NEW menu (cohort B)

GROUND RULE
-----------
Do **not** read past the line marked ``# === instructor section ===`` and do
**not** call ``_reveal``. The entire point of the lab is to decide *without*
knowing the truth — your data might contain a large effect, a tiny one, or
none at all; the variances might be equal or wildly different; the sample
might be plenty or far too small. You must diagnose which world you are in.
(Instructors use ``_reveal`` to grade.)
"""

import hashlib

import numpy as np

__all__ = ["load_my_ab", "BASELINE", "BREAKEVEN", "DESIGN_MDE",
           "BUSINESS_MDE", "ALPHA"]

# --- public, documented constants -------------------------------------------
BASELINE = 250.0      # mu0, historical ARPU in UAH
BREAKEVEN = 15.0      # DECISION threshold: smallest lift (UAH) worth shipping
DESIGN_MDE = 25.0     # PLANNING threshold: the "clear win" we size N for
BUSINESS_MDE = BREAKEVEN   # backward-compatible alias (== break-even)
ALPHA = 0.05          # significance level


def load_my_ab(student_id):
    """Return ``(control, test)`` weekly-ARPU samples for ``student_id``.

    Parameters
    ----------
    student_id : str
        Your KSE email or student id. Case- and whitespace-insensitive, so
        ``"Ihor.M@kse.org.ua"`` and ``" ihor.m@kse.org.ua "`` give the same
        dataset. Use **one** id consistently throughout the lab.

    Returns
    -------
    (control, test) : tuple of np.ndarray
        ``control`` = ARPU under the old menu, ``test`` = ARPU under the new
        menu. Positive, right-skewed money values. The two arrays may have
        **different** lengths.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    p = _params(student_id)
    rng = np.random.default_rng(p["seed"])
    control = _arpu(p["mean_c"], p["n_c"], p["sigma_c"], rng)
    test = _arpu(p["mean_t"], p["n_t"], p["sigma_t"], rng)
    return control, test


# ============================================================================
# === instructor section =====================================================
# Students: stop reading here. Everything below defines (and can reveal) the
# hidden ground truth used for grading.
# ============================================================================

# Six worlds. A correct lab reaches a *different* business decision in each,
# and the only way to tell them apart is from the data + the diagnostics.
# Decisions are made vs the +15 break-even; adequacy vs the +25 design-MDE:
#
#   clear_win       large effect, large n            -> SHIP (lower bound > break-even)
#   marginal_win    p<alpha but lift below break-even-> real, but not worth shipping
#   no_effect       a disguised A/A                  -> do not reject
#   underpowered    real effect >= break-even, tiny n-> extend the test
#   unequal_var     very unequal variances & sizes   -> Welch, not pooled
#   skewed_small_n  heavy skew + small n             -> t unreliable, bootstrap
_SCENARIOS = (
    "clear_win",
    "marginal_win",
    "no_effect",
    "underpowered",
    "unequal_var",
    "skewed_small_n",
)


def _seed_from_id(student_id):
    """Deterministic 32-bit seed from a normalised id."""
    norm = student_id.strip().lower().encode("utf-8")
    return int(hashlib.sha256(norm).hexdigest()[:8], 16)


def _params(student_id):
    """Pick a hidden world and concrete (jittered) parameters for this id."""
    seed = _seed_from_id(student_id)
    pick = np.random.default_rng(seed + 1)        # for parameter jitter
    scenario = _SCENARIOS[seed % len(_SCENARIOS)]

    mean_c = BASELINE                             # control = old menu = baseline
    sigma_c = sigma_t = 0.6                       # default lognormal shape

    if scenario == "clear_win":
        effect = float(pick.uniform(24, 32))      # well above MDE
        n_c = n_t = int(pick.integers(2200, 3200))

    elif scenario == "marginal_win":
        effect = float(pick.uniform(9, 13))       # below the +15 break-even...
        n_c = n_t = int(pick.integers(4500, 6500))  # ...but n large enough to be significant

    elif scenario == "no_effect":
        effect = 0.0                              # disguised A/A
        n_c = n_t = int(pick.integers(2000, 3000))

    elif scenario == "underpowered":
        effect = float(pick.uniform(22, 30))      # real and above break-even...
        n_c = n_t = int(pick.integers(95, 175))   # ...but far too few users

    elif scenario == "unequal_var":
        effect = float(pick.uniform(11, 18))
        sigma_c = 0.35                            # tight control...
        sigma_t = 0.95                            # ...noisy test
        n_c = int(pick.integers(2600, 3400))      # and unbalanced sizes
        n_t = int(pick.integers(260, 420))

    else:  # skewed_small_n
        effect = float(pick.uniform(14, 22))
        sigma_c = sigma_t = 1.1                   # heavy right tail
        n_c = n_t = int(pick.integers(40, 75))    # CLT has not kicked in

    return {
        "scenario": scenario,
        "seed": seed,
        "mean_c": mean_c,
        "mean_t": mean_c + effect,
        "effect": effect,
        "sigma_c": sigma_c,
        "sigma_t": sigma_t,
        "n_c": n_c,
        "n_t": n_t,
    }


def _arpu(true_mean, n, sigma, rng):
    """Lognormal ARPU sample with mean ``true_mean`` and log-shape ``sigma``."""
    mu = np.log(true_mean) - sigma ** 2 / 2
    return rng.lognormal(mean=mu, sigma=sigma, size=n)


def _reveal(student_id):
    """INSTRUCTOR ONLY. Return the hidden ground truth for ``student_id``."""
    p = _params(student_id)
    return {
        "scenario": p["scenario"],
        "true_control_mean": round(p["mean_c"], 2),
        "true_test_mean": round(p["mean_t"], 2),
        "true_effect_UAH": round(p["effect"], 2),
        "above_breakeven": p["effect"] >= BREAKEVEN,
        "above_design_mde": p["effect"] >= DESIGN_MDE,
        "sigma_control": p["sigma_c"],
        "sigma_test": p["sigma_t"],
        "n_control": p["n_c"],
        "n_test": p["n_t"],
    }
