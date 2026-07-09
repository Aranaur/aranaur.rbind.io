"""
greenbowl_boot.py — STAT220 *Applied Statistics*, Lab 9 (FINAL) data platform.

Lab 4 handed you a fixed A/B dataset and asked for a business decision. Lab 5
made you audit the *criterion*. Lab 6 was distribution forensics. Lab 7 was
categorical A/B forensics. Lab 8 took away the mean. **Lab 9 takes away the
last crutch: someone else's decision layer.** You are the lead analyst who
signs the go/no-go memo, and the number the CFO reads is not a p-value — it
is *"by how many percent did revenue grow, and does the feature pay for
itself?"*

GreenBowl is rolling **"Free delivery on orders over 300 UAH"** out to a new
city — *your* city. The experimentation platform logged, for every user in
the two-week test, their **total revenue over the two weeks (2-week ARPU,
UAH)** — zeros for the users who ordered nothing, a long right tail for the
ones who ordered a lot. The platform's automated report prints a **naive
headline**: a Welch t-test against zero plus a "percent CI" made by dividing
the UAH-difference CI by the frozen control mean — both traps you already
know. Your job is to rebuild the honest pipeline from the bootstrap practice
(`uplift_ci` → `uplift_ci_pois` → `rate_uplift` → `power_sim` → `ab_report`),
validate it, and read the result through the **economics**: the delivery
subsidy costs ≈3% of revenue, so the feature pays off only above
**breakeven = +3%** ARPU uplift.

Every student receives a **different** hidden experiment, deterministically
derived from their student id. Some cities got a clear win; some a feature
that actively hurts revenue; some a result stuck between "significant" and
"profitable"; some a country-scale sample where significance is guaranteed
and meaningless; some a handful of corporate accounts that manufactured the
whole "uplift"; and some an experiment that never had a chance to answer the
question it was asked. **The only honest way to find out which one you hold
is to run the pipeline.**

Fixed context (the same for everyone, stated in the lab):

    * unit                  one user
    * metric                2-week ARPU, UAH (zeros included; higher = better)
    * arms                  control = old delivery pricing,
                            test = free delivery over 300 UAH
    * breakeven             the subsidy costs ~3% of revenue -> the feature
                            pays off only if ARPU uplift > BREAKEVEN (+3%);
                            this is the DECISION threshold
    * design MDE            the team plans experiments to reliably catch a
                            DESIGN_MDE (+5%) effect; this is the PLANNING
                            threshold — do not confuse the two
    * significance level    alpha = 0.05

Usage (students):

    from greenbowl_boot import load_my_experiment, ALPHA, BREAKEVEN, DESIGN_MDE

    exp = load_my_experiment("your.email@kse.org.ua")
    exp.data             # tidy DataFrame: one row per user
                         #   [user_id, arm, revenue]
    exp.n_control, exp.n_test
    exp.control          # np.array of control 2-week revenues, UAH
    exp.test             # np.array of test 2-week revenues, UAH
    exp.city             # which city you are launching
    exp.user_base        # the city's user base the feature would roll out to
    exp.annual_revenue   # user_base * control ARPU * 26 -> feeds the EV line

    # the NATIONAL warehouse (for the streaming task): a parallel country-
    # wide experiment, exported one shard at a time — see CountryStream.
    stream = load_country_stream("your.email@kse.org.ua")
    stream.n_shards, stream.n_control, stream.n_test
    c_chunk, t_chunk = stream.shard(0)     # or:  for c_chunk, t_chunk in stream:

GROUND RULE
-----------
Do **not** read past the line marked ``# === instructor section ===`` and do
**not** call ``_reveal``. The whole point is to *diagnose* the experiment, not
to look up the answer. Naming the hidden scenario from the source instead of
demonstrating it with the pipeline scores **zero** for interpretation.
(Instructors use ``_reveal`` to grade.)
"""

import hashlib

import numpy as np
import pandas as pd

__all__ = ["load_my_experiment", "load_country_stream",
           "Experiment", "CountryStream",
           "ALPHA", "BREAKEVEN", "DESIGN_MDE"]

# --- public, documented constants -------------------------------------------
ALPHA = 0.05          # significance level
BREAKEVEN = 0.03      # decision threshold: the subsidy costs ~3% of revenue
DESIGN_MDE = 0.05     # planning threshold: the effect experiments are sized for


class Experiment:
    """One hidden revenue A/B experiment, keyed to a student id.

    Attributes
    ----------
    data : pandas.DataFrame
        One row per user, columns ``user_id``, ``arm`` ("control"/"test") and
        ``revenue`` (2-week revenue in UAH; zero = the user ordered nothing).
    city : str
        The city this launch targets (each student gets their own).
    user_base : int
        How many users the feature would roll out to in this city — the
        scale that turns an uplift into money.
    """

    def __init__(self, data, city, user_base):
        self.data = data.reset_index(drop=True)
        self.city = city
        self.user_base = int(user_base)

    @property
    def n_control(self):
        return int((self.data["arm"] == "control").sum())

    @property
    def n_test(self):
        return int((self.data["arm"] == "test").sum())

    @property
    def control(self):
        """2-week revenues (UAH) of the control arm, as an array."""
        d = self.data
        return d.loc[d["arm"] == "control", "revenue"].to_numpy()

    @property
    def test(self):
        """2-week revenues (UAH) of the test arm, as an array."""
        d = self.data
        return d.loc[d["arm"] == "test", "revenue"].to_numpy()

    @property
    def annual_revenue(self):
        """City-scale annual revenue at the control ARPU: base * ARPU * 26.

        Multiply this by an uplift to turn percent into UAH per year — the
        number the EV line of your report is denominated in.
        """
        return float(self.user_base * self.control.mean() * 26)

    def __repr__(self):
        return (f"<Experiment city={self.city!r} n_control={self.n_control} "
                f"n_test={self.n_test} user_base={self.user_base} "
                f"metric=2-week ARPU (UAH)>")


def load_my_experiment(student_id):
    """Return **your** hidden :class:`Experiment`, keyed to ``student_id``.

    What (if anything) the experiment can honestly support is **hidden**.
    Diagnose it with the bootstrap pipeline — do not try to read it off the
    source.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    rng = np.random.default_rng(_seed_from_id(student_id))
    return _BUILD[_world(student_id)](rng)


class CountryStream:
    """The national experimentation warehouse — exported one shard at a time.

    While your city ran its two-week test, the same feature ran as a parallel
    **country-wide** experiment. The full log lives in a sharded warehouse:
    ~24 shards, ~50k users per arm each — over a million users per arm in
    total. **The warehouse contract:** it exports **one shard at a time**.
    Your analysis code may hold one shard plus O(B) accumulators — never the
    concatenated dataset. (Pretend each shard lives on its own machine; at
    true warehouse scale even the raw concatenation would not fit, and the
    export API simply does not offer it.)

    Shards are generated on demand and **deterministically**: asking for the
    same shard twice — or iterating the stream again, in any order — yields
    identical data. That is what makes a one-pass weighted bootstrap
    reproducible and shard-parallel.

    Attributes / API
    ----------------
    n_shards : int          how many shards the warehouse holds
    n_control, n_test : int total users per arm across all shards
    shard(i) -> (control_chunk, test_chunk)   the i-th shard's revenues, UAH
    iter(stream)            yields shard(0), shard(1), ... in order
    """

    def __init__(self, student_id):
        seed = _seed_from_id(student_id, extra="country")
        rng = np.random.default_rng(seed)
        self._seed = seed
        self.n_shards = int(rng.integers(22, 27))
        self._pp = _params(rng)
        self._uplift = float(rng.uniform(0.025, 0.055))   # hidden truth
        self._sizes = [(int(rng.integers(45_000, 55_001)),
                        int(rng.integers(45_000, 55_001)))
                       for _ in range(self.n_shards)]

    @property
    def n_control(self):
        return sum(s[0] for s in self._sizes)

    @property
    def n_test(self):
        return sum(s[1] for s in self._sizes)

    def shard(self, i):
        """Return shard ``i`` as ``(control_chunk, test_chunk)`` arrays."""
        if not 0 <= i < self.n_shards:
            raise IndexError(f"shard index must be in [0, {self.n_shards})")
        rng = np.random.default_rng((self._seed, i))
        n_c, n_t = self._sizes[i]
        c = np.round(_arpu(n_c, rng, self._pp), 2)
        t = np.round(_arpu(n_t, rng, self._pp, uplift=self._uplift), 2)
        return c, t

    def __iter__(self):
        return (self.shard(i) for i in range(self.n_shards))

    def __len__(self):
        return self.n_shards

    def __repr__(self):
        return (f"<CountryStream n_shards={self.n_shards} "
                f"n_control={self.n_control:,} n_test={self.n_test:,} "
                f"(one shard at a time!)>")


def load_country_stream(student_id):
    """Return **your** national warehouse stream, keyed to ``student_id``.

    The country-wide experiment is separate from your city's: its own users,
    its own (hidden) effect. The honour rule applies here too: the true
    national uplift is diagnosed with your streaming engine, not read off
    the object's internals.
    """
    if not isinstance(student_id, str) or not student_id.strip():
        raise ValueError("student_id must be a non-empty string (your KSE email/id)")
    return CountryStream(student_id)


# ============================================================================
# === instructor section =====================================================
# Students: stop reading here. Everything below defines (and can reveal) the
# hidden ground truth used for grading.
# ============================================================================

# Six hidden experiment "worlds". A correct lab reaches a *different* final
# memo for each, and the only way to tell them apart is with the honest
# pipeline + the decision layer:
#
#   ship      n ~ 9k/arm, true uplift ~ +8%   -> SHIP: the honest CI sits
#                                                entirely above breakeven;
#                                                P(profitable) ~ 99%, EV
#                                                strongly positive.
#   harm      n ~ 6k/arm, true uplift ~ -6%   -> KILL: the CI is entirely
#             (the subsidy attracts tiny         below zero. The naive
#             baskets that cannibalise           headline also says "don't
#             normal orders)                     ship" — but for the wrong
#                                                reason (vs zero, not vs
#                                                breakeven).
#   grey      n ~ 6k/arm, true uplift ~ +4.5% -> GREY ZONE: significant vs
#                                                zero, but breakeven sits
#                                                inside the CI. The rational
#                                                move: BUY PRECISION —
#                                                power_sim on the current
#                                                data says how much more.
#   trivial   n ~ 150k/arm, true uplift        -> KILL BY ECONOMICS: p ~ 0
#             ~ +1.2%                             vs zero, yet the whole CI
#                                                is below breakeven. Also
#                                                forces the memory-aware /
#                                                Poisson path: a full B x N
#                                                resample matrix is GBs.
#   whale     n ~ 5k/arm, NO broad effect,    -> DON'T SHIP THE MEAN VERDICT:
#             6-8 corporate-catering            the "uplift" is manufactured
#             accounts landed in test           by a handful of users. Median
#                                               uplift ~ 0, top-5 users hold
#                                               a huge share of arm revenue,
#                                               winsorising kills the effect.
#                                               The CI is only as good as
#                                               F-hat ~ F — a tail the sample
#                                               barely touches breaks the
#                                               promise.
#   under     n ~ 600/arm, true uplift ~ +5%  -> NO CONCLUSION POSSIBLE: the
#                                               CI covers everything from
#                                               "harmful" to "great". A-priori
#                                               power at DESIGN_MDE is ~10%;
#                                               the launch never had a chance.
#                                               No post-hoc power — compute
#                                               the n a relaunch needs.
_WORLDS = ("ship", "harm", "grey", "trivial", "whale", "under")


# A per-lab salt makes THIS lab's world assignment statistically INDEPENDENT
# of every other lab's. Without it, world = hash(id) % 6 is identical across
# any two labs that share the hash and have 6 worlds, so a student who lands
# on a "hard" index stays on it in every lab. EACH NEW LAB MUST USE ITS OWN
# DISTINCT SALT so luck does not carry over across the course.
_LAB_SALT = "stat220-lab9-boot-v1"

_CITIES = ("Odesa", "Kharkiv", "Dnipro", "Zaporizhzhia", "Vinnytsia",
           "Poltava", "Chernihiv", "Cherkasy", "Ternopil", "Rivne",
           "Mykolaiv", "Ivano-Frankivsk", "Uzhhorod", "Sumy",
           "Khmelnytskyi", "Zhytomyr")


def _seed_from_id(student_id, extra=""):
    """Deterministic 32-bit seed from a normalised id (drives the DATA sample).

    ``extra`` derives an independent sub-seed for a side dataset (the
    country stream) so it does not share randomness with the city data."""
    norm = student_id.strip().lower()
    if extra:
        norm += "|" + extra
    return int(hashlib.sha256(norm.encode("utf-8")).hexdigest()[:8], 16)


def _world(student_id):
    """Pick the hidden experiment world for this id.

    Salted with ``_LAB_SALT`` so the world drawn here does NOT correlate with
    the world drawn in any other lab — no student is consistently lucky or
    unlucky across the course."""
    norm = (student_id.strip().lower() + "|" + _LAB_SALT).encode("utf-8")
    h = int(hashlib.sha256(norm).hexdigest()[:8], 16)
    return _WORLDS[h % len(_WORLDS)]


def _jit(n, rng, frac=0.03):
    """Jitter an arm size by +/- frac so the absolute size cannot fingerprint
    a world across students. Left untouched for tiny samples (n < 50)."""
    if n < 50:
        return n
    return int(n + round(rng.uniform(-frac, frac) * n))


def _params(rng):
    """Per-student revenue-law parameters, jittered so no absolute number
    (ARPU, zero share) can fingerprint a world."""
    return {"p_active": rng.uniform(0.72, 0.78),   # share who ordered at all
            "lam": rng.uniform(1.1, 1.3),          # extra orders ~ Pois(lam)
            "med": rng.uniform(220, 260),          # median basket, UAH
            "sig": rng.uniform(0.36, 0.44)}        # basket lognormal sigma


def _arpu(n, rng, pp, uplift=0.0):
    """2-week revenue per user, UAH: zeros + right skew (as in Practice 9)."""
    active = rng.random(n) < pp["p_active"]
    orders = 1 + rng.poisson(pp["lam"], n)
    check = rng.lognormal(np.log(pp["med"]), pp["sig"], n)
    return np.where(active, orders * check * (1 + uplift), 0.0)


def _uplift_delta(c, t, z=1.96):
    """Delta-method CI for t.mean()/c.mean() - 1. Build-time acceptance check
    only — a cheap stand-in for the bootstrap CI the student will compute."""
    mc, mt = c.mean(), t.mean()
    vc, vt = c.var(ddof=1) / c.size, t.var(ddof=1) / t.size
    hat = mt / mc - 1
    se = np.sqrt(vt / mc**2 + mt**2 * vc / mc**4)
    return hat, hat - z * se, hat + z * se


def _ids(n, rng, width=8):
    """n unique user ids, drawn from a wide range so neither the range nor
    the ordering fingerprints an arm or a world."""
    pool = rng.choice(10 ** width, size=n, replace=False)
    return np.array([f"u_{i:0{width}d}" for i in pool])


def _experiment(rng, c, t, base_mult):
    n_c, n_t = c.size, t.size
    ids = _ids(n_c + n_t, rng)
    data = pd.DataFrame({
        "user_id": np.concatenate([ids[:n_c], ids[n_c:]]),
        "arm": np.array(["control"] * n_c + ["test"] * n_t),
        "revenue": np.round(np.concatenate([c, t]), 2),
    })
    city = rng.choice(_CITIES)
    user_base = int(round((n_c + n_t) * rng.uniform(*base_mult), -2))
    return Experiment(data, city, user_base)


# --- the experiment builders ------------------------------------------------
# Each builder redraws (up to _TRIES times) until the realized sample matches
# its archetype by the delta-method check, so almost every student sees the
# signature their world was designed to teach. The rare escapee is caught by
# grade_boot.py's ⚠ flag — a student who honestly reported an off-archetype
# sample must NOT be failed against the intended verdict.
_TRIES = 40


def _b_ship(rng):
    n_c, n_t = _jit(9000, rng), _jit(9000, rng)
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp, uplift=0.08)
        _, lo, hi = _uplift_delta(c, t)
        if lo > BREAKEVEN + 0.005:
            break
    return _experiment(rng, c, t, base_mult=(8, 14))


def _b_harm(rng):
    n_c, n_t = _jit(6000, rng), _jit(6000, rng)
    # free delivery attracts a swarm of sub-300-UAH baskets that cannibalise
    # normal orders: the active users' revenue DROPS ~6%.
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp, uplift=-0.06)
        _, lo, hi = _uplift_delta(c, t)
        if hi < -0.005:
            break
    return _experiment(rng, c, t, base_mult=(8, 14))


def _b_grey(rng):
    n_c, n_t = _jit(6000, rng), _jit(6000, rng)
    # real ~+4.5%: significant against zero, but breakeven stays inside the
    # CI — the Practice 9 cliffhanger as the student's own final call.
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp, uplift=0.045)
        _, lo, hi = _uplift_delta(c, t)
        if 0.002 < lo < BREAKEVEN - 0.004 and hi > BREAKEVEN + 0.006:
            break
    return _experiment(rng, c, t, base_mult=(8, 14))


def _b_trivial(rng):
    # n must be LARGE enough that a ~+1.2% uplift is all but guaranteed to be
    # "significant" (the lesson is "p ~ 0 yet the CI never reaches
    # breakeven"), and large enough that the naive B x N resample matrix
    # from Practice 9 does not fit in memory — forcing the Poisson / one-pass
    # path.
    n_c, n_t = _jit(150_000, rng), _jit(150_000, rng)
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp, uplift=0.012)
        _, lo, hi = _uplift_delta(c, t)
        if lo > 0.002 and hi < BREAKEVEN - 0.004:
            break
    return _experiment(rng, c, t, base_mult=(2, 4))


def _b_whale(rng):
    n_c, n_t = _jit(5000, rng), _jit(5000, rng)
    # NO broad effect — both arms share the law. But 6-8 corporate-catering
    # accounts (25-40k UAH over two weeks vs a typical ~800) landed in the
    # TEST arm by chance. The mean "uplift" they manufacture clears the
    # naive headline, while the median and the winsorised mean see nothing.
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp)
        k = int(rng.integers(6, 9))
        idx = rng.choice(n_t, size=k, replace=False)
        t[idx] = rng.uniform(25_000, 40_000, size=k)
        # acceptance: the headline "wins", the robust reads do not
        _, lo, _hi = _uplift_delta(c, t)
        med_up = np.median(t) / np.median(c) - 1
        cap_c, cap_t = np.percentile(c, 99), np.percentile(t, 99)
        wins_up = (np.minimum(t, cap_t).mean()
                   / np.minimum(c, cap_c).mean() - 1)
        if lo > 0.005 and abs(med_up) < 0.03 and abs(wins_up) < 0.02:
            break
    return _experiment(rng, c, t, base_mult=(8, 14))


def _b_under(rng):
    n_c, n_t = _jit(600, rng), _jit(600, rng)
    # the effect is REAL (~+5%) — but at 600/arm the CI covers everything
    # from "harmful" to "great", and a-priori power at DESIGN_MDE is ~10%.
    # The only honest memo: the launch never had a chance; here is the n a
    # relaunch needs.
    for _ in range(_TRIES):
        pp = _params(rng)
        c = _arpu(n_c, rng, pp)
        t = _arpu(n_t, rng, pp, uplift=0.05)
        _, lo, hi = _uplift_delta(c, t)
        if lo < -0.005 and hi > BREAKEVEN + 0.01:
            break
    return _experiment(rng, c, t, base_mult=(25, 45))


_BUILD = {
    "ship": _b_ship,
    "harm": _b_harm,
    "grey": _b_grey,
    "trivial": _b_trivial,
    "whale": _b_whale,
    "under": _b_under,
}

# Per-world: the decision the lab expects, and the decisive evidence.
_EXPECTED = {
    "ship": (
        "SHIP — the whole CI clears breakeven",
        "Honest uplift ~ +8% with a 95% CI entirely above +3%: profitability "
        "is proven, not just significance. P(uplift >= breakeven) ~ 99%, EV "
        "strongly positive at the city's annual revenue. The naive %-CI "
        "roughly agrees here (small-effect luck — Practice 9 Part I); the "
        "correct memo still quotes the honest interval and the three-zone "
        "read against breakeven, not against zero."),
    "harm": (
        "KILL — switch the feature off; it hurts revenue",
        "Honest uplift ~ -6% with a 95% CI entirely below zero: free "
        "delivery attracted small baskets that cannibalised normal orders. "
        "The headline also prints 'don't ship', but for the wrong reason — "
        "it compares to zero and says nothing about how much money continuing "
        "the rollout would burn. EV is sharply negative; P(profitable) ~ 0. "
        "The memo must state the direction and size of the harm, not just "
        "'not significant enough to ship'."),
    "grey": (
        "GREY ZONE — significant, not yet profitable; buy precision",
        "Honest uplift ~ +4.5%, CI above zero but with breakeven (+3%) "
        "INSIDE the interval: the growth is real, profitability is unproven. "
        "P(uplift >= breakeven) ~ 70-85%, EV positive — killing the feature "
        "throws away expected money, shipping it bets on an unproven margin. "
        "The rational memo: extend the experiment, with the extra n/weeks "
        "computed by power_sim using the current data as the pilot (never a "
        "gut-feel 'run it a bit longer')."),
    "trivial": (
        "KILL BY ECONOMICS — significant but below breakeven",
        "n ~ 150k/arm: p << 0.05 against zero, yet the entire CI (~ +0.8% "
        "to +1.6%) sits BELOW the +3% breakeven — the feature demonstrably "
        "does not pay for its subsidy. 'Statistically significant' and "
        "'profitable' are different sentences; at this n significance is "
        "guaranteed and meaningless. Bonus signature: the sample is too big "
        "for the naive B x N resample matrix — the report must use the "
        "Poisson / one-pass engine (or explicit chunking) and say so."),
    "whale": (
        "DON'T SHIP THE MEAN VERDICT — the uplift is a handful of users",
        "No broad effect: 6-8 corporate-catering accounts (25-40k UAH) "
        "landed in the test arm and manufactured the whole mean 'uplift'. "
        "Diagnostics that catch it: top-5 users hold a large share of the "
        "test arm's revenue (vs ~1% in control); the median-uplift CI "
        "covers zero; winsorising at P99 (or dropping the whales) kills the "
        "effect; the bootstrap star distribution is lumpy — resamples that "
        "include the whales twice tell a different story than resamples "
        "that miss them. The lecture's edge-statistic warning in A/B "
        "costume: the CI's promise rests on F-hat ~ F, and a tail the "
        "sample barely touches breaks it. Memo: do not ship on ARPU-mean; "
        "either cap/winsorise the metric by design or rerun longer and "
        "check whether whales are reproducible revenue."),
    "under": (
        "NO CONCLUSION — the experiment never had a chance; relaunch with a "
        "computed n",
        "n ~ 600/arm: the CI spans roughly -4% to +13% — it covers zero, "
        "breakeven and everything else, so no decision is supported. The "
        "trap to refuse: 'power at the observed effect' (post-hoc power) — "
        "a deterministic function of the p-value, zero new information. The "
        "honest deliverable: a-priori power at DESIGN_MDE for this n via "
        "power_sim on the control as pilot (~10-15%), the verdict that the "
        "launch should never have happened at this size, and the n (~8000/"
        "arm) a relaunch needs for 80% power."),
}


# Rough effort tier per world. The worlds are NOT equally hard: a clean
# "ship" is a short, direct verdict, whereas "whale" (fragility forensics ->
# robust reruns -> a metric-design recommendation) and "under" (refusing the
# post-hoc-power excuse -> a-priori power -> relaunch sizing) take materially
# more work. Surfaced by grade_boot.py so the verdict can be graded
# difficulty-aware.
_DIFFICULTY = {
    "ship": "easy",
    "harm": "easy",
    "grey": "medium",
    "trivial": "medium",
    "whale": "hard",
    "under": "hard",
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
        # the streaming task's separate hidden truth (same for every world)
        "country_uplift": CountryStream(student_id)._uplift,
    }
