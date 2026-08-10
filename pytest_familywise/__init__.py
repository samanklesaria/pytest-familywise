"""Pytest plugin for family-wise error rate control of randomized tests.

Tests register a *sampler* via the ``assertNotReject`` fixture: a callable that
draws a group of data from the generator it is handed and returns the p-value
for that group.  After all tests run, the plugin applies a step-down
multiple-comparison procedure to control the family-wise error rate (FWER).  A
test "passes" when its p-value is too large to reject the null hypothesis after
correction; it "fails" when the null hypothesis is rejected.

Two procedures are available via ``--correction``:

* ``holm`` (default) – Holm-Bonferroni.  Assumes nothing about the dependence
  between tests, and is correspondingly conservative when they are correlated.
* ``westfall-young`` – resampling step-down (minP).  Estimates the *joint* null
  distribution of the p-value vector rather than bounding it, recovering the
  power Holm gives up.  Requires each test to supply a ``null_sample`` callable
  (see ``assertNotReject``).

Both are expressed as adjusted p-values: each test's raw p-value is mapped to an
adjusted value, and the null hypothesis is rejected when ``adjusted <= alpha``.

Group-sequential testing
------------------------
Under ``--groupsize`` the sampler is called once per group rather than once:
every test draws a group, the correction is applied to the whole family, tests
whose null is rejected stop drawing, and the survivors draw another group.
Because each group is fresh independent data, ``Z_k = k**-0.5 * sum_j
Phi^-1(1 - p_j)`` has *exactly* the canonical sequential law -- for any test
statistic -- so an exact Pocock boundary applies.  The boundary is inverted
into a single sequential p-value per test, which then goes through the same
step-down machinery.  See the README for the derivation, and ``_cross_prob``
for the numerics.

Other fixtures expose required-sample-size calculations so tests can be
sized for the desired per-test power before running.
The sample-size fixtures use Bonferroni corrected significance levels rather
than the raw familywise alpha.  The plugin records which tests use
``assertNotReject`` at collection time, and every one of those *m* tests sizes
against ``alpha / m``.  Tests outside that set are never corrected, so they size
against the raw alpha.

Why ``alpha / m`` for all of them rather than Holm's ``alpha / (m - k + 1)``
ladder: a test's rung depends on its p-value *rank*, which is not known until
the suite has run.  A test with a real effect produces a small p-value, so it
lands at (or near) rank 1 and faces ``alpha / m`` whatever its position in the
file.  Sizing any test for a laxer rung is betting it is not the broken one, and
there is nothing to place that bet with.  The cost is mild -- the penalty grows
like ``log m``, roughly 1.7x the raw-alpha sample size at m=10 -- and it is the
only sizing that keeps the requested power for whichever test actually breaks.

Loading
-------
The package registers itself with pytest via the ``pytest11`` entry point in
``pyproject.toml``:

```toml
[project.entry-points."pytest11"]
random = "pytest_familywise"
```

Installing the package is sufficient — pytest discovers the entry point at
startup and loads the plugin automatically.  No ``conftest.py`` import is
required in the project under test.

If you are working from a source checkout without installing the package,
add the following to your project's ``conftest.py`` instead:

```python
pytest_plugins = ["pytest_familywise"]
```
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Set

import numpy as np
from numpy.random import SeedSequence, default_rng
from scipy.signal import fftconvolve
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2
import pytest

# Base seed for Westfall-Young resampling.  Fixed rather than configurable so
# that runs are reproducible by default; draw b uses default_rng(_BASE_SEED + b)
# identically in every test, which is what aligns the null columns.
_BASE_SEED = 0x5EED

# p-values are clamped to this before the probit transform, so that an exact 0
# (permutation tests, underflowed tails) gives a large-but-finite z ~ 37.
_P_FLOOR = 1e-300

# Tail cutoff for the Armitage-McPherson grid, in standard deviations.  The
# neglected mass is ~6e-16, below the recursion's 1e-8 target.
_GRID_TAIL = 8.0
_GRID_STEP = 0.02


def _ztest_n(alpha: float, power: float, effect_size: float, two_sided: bool = True) -> int:
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    return math.ceil(((z_alpha + z_beta) / effect_size) ** 2)

def _chisquare_n(alpha: float, power: float, effect_size: float, df: int) -> int:
    critical = chi2.ppf(1 - alpha, df)

    def shortfall(n: float) -> float:
        return ncx2.sf(critical, df, n * effect_size ** 2) - power

    hi = 4.0
    while shortfall(hi) < 0:
        hi *= 2
    return math.ceil(brentq(shortfall, 1.0, hi))

def _ks_n(alpha: float, power: float, effect_size: float, two_sample: bool = False) -> int:
    beta = 1.0 - power
    delta = effect_size
    n = (math.sqrt(math.log(2.0 / alpha)) + math.sqrt(math.log(2.0 / beta))) ** 2 / (
        2.0 * delta ** 2
    )
    n_ceil = math.ceil(n)
    # For two equal groups the effective n is n_each/2, so double.
    return n_ceil * 2 if two_sample else n_ceil

def _validated(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p-value must be in [0, 1], got {p!r}")
    return float(p)


def _cross_prob(b: float, k_looks: int, drift: float = 0.0) -> float:
    """``P(exists k <= K : Z_k >= b)`` for the canonical sequential statistic.

    ``Z_k = S_k / sqrt(k)`` where ``S_k`` is the sum of *k* i.i.d. ``N(drift, 1)``
    increments, so ``corr(Z_j, Z_k) = sqrt(j / k)``.  With ``drift=0`` this is
    the null crossing probability that defines a Pocock boundary (and, read
    backwards, the sequential p-value); with ``drift=mu`` it is the design's
    power against that alternative.

    Evaluated by the Armitage-McPherson recursion on the sub-density ``f_k`` of
    ``S_k`` restricted to "no earlier look crossed"::

        f_1(s)     = phi(s - drift)                          for s < b
        f_{k+1}(s) = int_{u < b sqrt(k)} f_k(u) phi(s - u - drift) du
        P          = 1 - int_{s < b sqrt(K)} f_K(s) ds

    Independent increments turn the K-dimensional orthant probability into K
    one-dimensional convolutions, which is both exact and far cheaper than
    ``scipy.stats.multivariate_normal.cdf`` -- that routine is randomised
    quasi-Monte-Carlo, so it varies run to run and would make a reported
    p-value irreproducible.
    """
    k = int(k_looks)
    if k < 1:
        raise ValueError(f"k_looks must be >= 1, got {k_looks!r}")
    if k == 1:
        # Exact, and keeps the non-sequential path bit-identical to reporting p.
        return float(norm.sf(b - drift))

    # Above b*sqrt(j) the sub-density is identically zero, so the grid needs to
    # reach only the largest of those cutoffs.  Below there is no boundary, so
    # that edge is a genuine tail cutoff: S_j has mean j*drift and sd sqrt(j),
    # and j*drift - 8*sqrt(j) dips before it rises, so take the minimum.
    roots = np.sqrt(np.arange(1, k + 1))
    hi = float(b * roots[-1]) if b >= 0 else float(b)
    lo = float(np.min(np.arange(1, k + 1) * drift - _GRID_TAIL * roots))
    if hi <= lo:
        return 1.0

    n = int((hi - lo) / _GRID_STEP) | 1  # odd, for Simpson
    n = min(max(n, 401), 8001)
    s, h = np.linspace(lo, hi, n, retstep=True)
    weights = np.ones(n)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    weights *= h / 3.0

    # Truncate with a partial-cell mask rather than snapping to the nearest
    # grid point: each point stands for a cell of width h, and the cutoff
    # b*sqrt(j) generally falls inside one.  Snapping makes the result jump by
    # O(h) as b moves, which shows up directly as noise in a reported p-value.
    def truncated(density: "np.ndarray", cutoff: float) -> "np.ndarray":
        return density * np.clip((cutoff - s) / h + 0.5, 0.0, 1.0)

    kernel = norm.pdf(np.arange(-(n - 1), n) * h - drift)
    f = norm.pdf(s - drift)
    for j in range(1, k):
        f = truncated(f, float(b * roots[j - 1]))
        f = fftconvolve(weights * f, kernel)[n - 1:2 * n - 1]
    f = truncated(f, float(b * roots[-1]))
    return float(min(1.0, max(0.0, 1.0 - float((weights * f).sum()))))


@lru_cache(maxsize=None)
def _pocock_c(alpha: float, k_looks: int) -> float:
    "The constant Pocock boundary: the b with ``_cross_prob(b, K) == alpha``."
    return float(brentq(
        lambda b: _cross_prob(b, k_looks) - alpha, -8.0, 12.0, xtol=1e-9,
    ))


@lru_cache(maxsize=None)
def _cross_prob_grid(k_looks: int) -> tuple:
    """Tabulate ``log _cross_prob(., K)`` for bulk evaluation.

    Only the Westfall-Young null columns go through this -- B values per test
    per look, which is far too many to run the recursion for.  Their precision
    needs are modest: an adjusted p-value from B resamples is a multiple of
    1/B, so interpolation error well under that is invisible.  Observed
    p-values always use the exact recursion.
    """
    grid = np.linspace(-8.0, 10.0, 361)
    logp = np.log(np.maximum(
        [_cross_prob(float(b), k_looks) for b in grid], _P_FLOOR,
    ))
    return tuple(grid), tuple(logp)


def _cross_prob_many(bs: "np.ndarray", k_looks: int) -> "np.ndarray":
    "Vectorised ``_cross_prob`` by log-interpolation; see ``_cross_prob_grid``."
    if k_looks == 1:
        return np.asarray(norm.sf(bs), dtype=float)
    grid, logp = _cross_prob_grid(k_looks)
    return np.exp(np.interp(bs, np.asarray(grid), np.asarray(logp)))


def _group_drift(
    n_at: Callable[[float, float], int], alpha: float, groupsize: int
) -> float:
    """Per-group drift ``mu = E[Phi^-1(1 - p_j)]`` for a group of *groupsize*.

    ``n_at(alpha, power)`` is one of the existing sample-size functions with its
    effect size already bound.  Invert it to find the power a single group
    achieves at level *alpha*, then map that back onto the probit scale.

    ponytail: exact when the probit-transformed p-value is normal with unit
    variance (the location-normal case, i.e. the z-test); an approximation for
    chi-square and KS.  It only selects the number of looks -- a wrong mu costs
    power, never error-rate control.

    Returns 0.0 when a single group carries no usable signal at all, which the
    caller reports as a group size too small for the test.
    """
    lo, hi = alpha + 1e-9, 1.0 - 1e-9
    if n_at(alpha, lo) > groupsize:
        return 0.0
    if n_at(alpha, hi) <= groupsize:
        power = hi
    else:
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if n_at(alpha, mid) <= groupsize:
                lo = mid
            else:
                hi = mid
        power = lo
    return float(norm.ppf(power) + norm.ppf(1.0 - alpha))


def _n_looks(mu: float, alpha: float, power: float, cap: int = 60) -> int:
    "Smallest K whose Pocock design attains *power* against per-group drift mu."
    if mu <= 0.0:
        raise ValueError("per-group drift must be positive")
    for k in range(1, cap + 1):
        if _cross_prob(_pocock_c(alpha, k), k, drift=mu) >= power:
            return k
    raise ValueError(
        f"no group-sequential design with <= {cap} looks reaches power "
        f"{power} at alpha={alpha:g} with this group size; raise --groupsize"
    )

@dataclass
class _LookState:
    """Group-sequential state for one test, accumulated across looks.

    Lives on the plugin rather than on the reporter: later looks re-run the
    test, so the function-scoped ``assertNotReject`` fixture is rebuilt each
    time and cannot carry anything forward.
    """
    k_looks: int = 1
    looks_done: int = 0
    z_sum: float = 0.0
    z_max: float = -math.inf
    value: Optional[float] = None                 # sequential p-value p*
    nulls: Optional["np.ndarray"] = None          # (B,) null p*, WY only
    null_z_sum: Optional["np.ndarray"] = None
    null_z_max: Optional["np.ndarray"] = None
    adjusted: Optional[float] = None
    active: bool = True


class PValueReporter:
    "Callable returned by the ``assertNotReject`` fixture."

    def __init__(self, plugin: "FamilywisePlugin", nodeid: str) -> None:
        self._plugin = plugin
        self._nodeid = nodeid

    def __call__(
        self,
        sampler: Callable[..., float],
        null_sample: Optional[Callable[..., float]] = None,
    ) -> None:
        plugin = self._plugin
        if not callable(sampler):
            raise TypeError(
                "assertNotReject takes a sampler, not a p-value: "
                "assertNotReject(fn), where fn(rng) draws one group of data "
                "from rng and returns the p-value for that group"
            )
        if plugin.correction == "westfall-young" and null_sample is None:
            raise ValueError(
                "the westfall-young correction requires "
                "assertNotReject(fn, null_sample=g), where g(rng) recomputes "
                "this test's p-value on an H0-imposing transform of the "
                "observed data (permute, sign-flip, center-then-bootstrap)"
            )
        state = plugin._state.get(self._nodeid)
        if state is None:
            if plugin.groupsize > 0:
                raise ValueError(
                    "--groupsize needs the number of looks, which comes from a "
                    "sample-size fixture; call ztest_sample_size (or "
                    "chisquare_/ks_) before assertNotReject"
                )
            state = plugin._state.setdefault(self._nodeid, _LookState())

        if plugin._backfill:
            # Reconstructing null columns for looks already taken; the observed
            # statistic must not advance and the sampler must not be called.
            self._resample(state, null_sample, range(1, state.looks_done + 1), reset=True)
            return

        state.looks_done += 1
        look = state.looks_done
        p = _validated(sampler(plugin.rng))
        if state.k_looks == 1:
            state.value = p  # degenerate case: p* == p, exactly
        else:
            state.z_sum += float(norm.isf(min(max(p, _P_FLOOR), 1.0)))
            state.z_max = max(state.z_max, state.z_sum / math.sqrt(look))
            state.value = _cross_prob(state.z_max, state.k_looks)
        if null_sample is not None and plugin._nulls_open:
            self._resample(state, null_sample, [look], reset=False)

    def _resample(self, state, null_sample, looks, reset: bool) -> None:
        "Accumulate the null path for the given looks; see ``_cross_prob``."
        plugin = self._plugin
        if null_sample is None:
            return
        count = plugin.resamples
        if reset or state.null_z_sum is None:
            state.null_z_sum = np.zeros(count)
            state.null_z_max = np.full(count, -np.inf)
        for look in looks:
            draws = np.fromiter(
                (
                    _validated(null_sample(plugin.null_rng(look, b)))
                    for b in range(count)
                ),
                dtype=float,
                count=count,
            )
            if state.k_looks == 1:
                state.nulls = draws
                return
            state.null_z_sum += norm.isf(np.clip(draws, _P_FLOOR, 1.0))
            state.null_z_max = np.maximum(
                state.null_z_max, state.null_z_sum / math.sqrt(look)
            )
        state.nulls = _cross_prob_many(state.null_z_max, state.k_looks)


def _holm_adjusted(
    pvalues: Sequence[float],
    nulls: Sequence[None] = (),  # unused; shared with WY
) -> List[float]:
    """Holm-Bonferroni adjusted p-values for p-values sorted ascending.

    ``adj_k = max(adj_{k-1}, min(1, (m - k + 1) * p_k))`` -- equivalent to
    comparing each ``p_k`` against the threshold ``alpha / (m - k + 1)`` and
    stopping at the first non-rejection.
    """
    m = len(pvalues)
    ranks = m - np.arange(m)  # m - k + 1 for k = 1..m
    return np.maximum.accumulate(
        np.minimum(1.0, ranks * np.asarray(pvalues, dtype=float))
    ).tolist()

def _westfall_young_adjusted(
    pvalues: Sequence[float],
    nulls: Sequence[Optional[Sequence[float]]],
) -> List[float]:
    """Westfall-Young minP adjusted p-values for p-values sorted ascending.

    ``nulls[j]`` holds the resampled null p-values for the test at sorted
    position *j*, all lists the same length *B*.  For each resample *b* the
    successive minima over the tail of the ordering give the null distribution
    of the minimum p-value among the not-yet-rejected hypotheses:

        q[b][k] = min(q[b][k+1], nulls[k][b])
        adj_k   = #{b : q[b][k] <= p_k} / B

    followed by enforced monotonicity.  Note the granularity floor: the
    smallest attainable adjusted p-value is ``1 / B``, so *B* must exceed
    ``1 / alpha`` for any rejection to be possible at all.
    """
    m = len(pvalues)
    if m == 0:
        return []
    columns = [n for n in nulls if n is not None]
    if len(columns) != m:
        raise ValueError("westfall-young requires null samples for every test")
    b_count = len(columns[0])
    if b_count == 0 or any(len(c) != b_count for c in columns):
        raise ValueError("all tests must supply the same number of null samples")

    # Successive minima over the tail of the ordering: reverse, take a running
    # minimum, reverse back.  Then count, per rank, the resamples at or below the
    # observed p-value, and enforce monotonicity with a running maximum.
    matrix = np.asarray(columns, dtype=float)
    tail_min = np.minimum.accumulate(matrix[::-1])[::-1]
    counts = (tail_min <= np.asarray(pvalues, dtype=float)[:, None]).sum(axis=1)
    return np.maximum.accumulate(counts / b_count).tolist()


CORRECTIONS: Dict[str, Callable[..., List[float]]] = {
    "holm": _holm_adjusted,
    "westfall-young": _westfall_young_adjusted,
}

CORRECTION_NAMES = {
    "holm": "Holm-Bonferroni",
    "westfall-young": "Westfall-Young",
}

@dataclass
class _CorrectedResult:
    "Internal result record"
    nodeid: str
    p_value: float
    adjusted: Optional[float]
    passed: bool
    looks: int = 1
    k_looks: int = 1

@dataclass(eq=False)
class FamilywisePlugin:
    "Collects p-values, defers pass/fail, and applies the FWER correction."

    alpha: float
    power: float
    correction: str
    resamples: int
    groupsize: int = 0

    rng: "np.random.Generator" = field(init=False, default_factory=default_rng)
    _state: Dict[str, _LookState] = field(init=False, default_factory=dict)
    _corrected: List[_CorrectedResult] = field(init=False, default_factory=list)
    # Tests participating in the correction (i.e. using assertNotReject).
    _participating: Set[str] = field(init=False, default_factory=set)
    # Failures raised by a test during a later look, keyed by nodeid.
    _errors: Dict[str, object] = field(init=False, default_factory=dict)
    _nulls_open: bool = field(init=False, default=True)
    _need_backfill: bool = field(init=False, default=False)
    _backfill: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        # Outside sequential mode the null resampling is eager, as it always
        # was.  Under --groupsize it is deferred until it could change a
        # verdict; see _decide.
        self._nulls_open = self.groupsize <= 0

    # Record which tests participate in the correction
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items: List[pytest.Item]) -> None:
        self._participating = {
            item.nodeid
            for item in items
            if "assertNotReject" in getattr(item, "fixturenames", ())
        }

    def get_corrected_alpha(self, nodeid: str) -> float:
        if nodeid not in self._participating:
            return self.alpha
        return self.alpha / len(self._participating)

    def null_rng(self, look: int, b: int) -> "np.random.Generator":
        """Generator for Westfall-Young resample *b* at *look*.

        Keyed on (look, b) alone, never on the test, so that every test sees the
        same draw at the same index and their null columns stay aligned -- which
        is what lets the procedure measure their correlation.
        """
        if self.groupsize <= 0:
            return default_rng(_BASE_SEED + b)
        return default_rng(SeedSequence([_BASE_SEED, look, b]))

    def plan(
        self, nodeid: str, n_at: Callable[[float, float], int], alpha: float
    ) -> int:
        """Sample size for one test, and under --groupsize its number of looks.

        ``n_at(alpha, power)`` is one of the sample-size functions with its
        effect size bound.  In sequential mode the answer is always the group
        size -- the effect size moves the number of looks instead.
        """
        if self.groupsize <= 0:
            return n_at(alpha, self.power)
        mu = _group_drift(n_at, alpha, self.groupsize)
        if mu <= 0.0:
            raise ValueError(
                f"a single group of {self.groupsize} samples has no power to "
                f"detect this effect at α={alpha:g}, so no number of groups "
                f"will: combining groups is weaker than pooling them.  Raise "
                f"--groupsize (a fixed-sample run needs "
                f"n={n_at(alpha, self.power)})"
            )
        k = _n_looks(mu, alpha, self.power)
        state = self._state.setdefault(nodeid, _LookState())
        state.k_looks = max(state.k_looks, k)  # a test may size twice
        return self.groupsize

    # ---- group-sequential driver -------------------------------------------

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtestloop(self, session: pytest.Session):
        """Look 1 is the ordinary run; re-run survivors for the later looks.

        Re-running the protocol (rather than calling samplers at session end)
        is what keeps every sampler inside a live call phase, with its fixtures
        set up and its exceptions reported as ordinary test failures.
        """
        yield
        if self.groupsize <= 0 or session.shouldstop or session.shouldfail:
            return
        max_looks = max((st.k_looks for st in self._state.values()), default=1)
        look = 1
        while True:
            self._settle(session)
            if self._stop_for_maxfail(session) or look >= max_looks:
                break
            look += 1
            due = self._due(session, look)
            if not due:
                break
            self._rerun(due)
        self._settle(session)

    def _settle(self, session: pytest.Session) -> None:
        "Decide; if that opened the null gate, backfill the columns and redecide."
        self._decide()
        if self._need_backfill:
            self._need_backfill = False
            self._rerun(self._due(session, None), backfill=True)
            self._decide()

    def _due(self, session: pytest.Session, look: Optional[int]) -> List[pytest.Item]:
        "Items to run at *look*, or (look=None) every item with a p-value yet."
        out = []
        for item in session.items:
            state = self._state.get(item.nodeid)
            if state is None or state.value is None:
                continue
            if look is not None and not (state.active and state.k_looks >= look):
                continue
            out.append(item)
        return out

    def _rerun(self, items: List[pytest.Item], backfill: bool = False) -> None:
        from _pytest.runner import runtestprotocol

        self._backfill = backfill
        try:
            for i, item in enumerate(items):
                nextitem = items[i + 1] if i + 1 < len(items) else None
                # log=False: these reports never reach the terminal, so pytest's
                # own counts stay put and the verdict is applied to the look-1
                # report in pytest_terminal_summary as before.
                for report in runtestprotocol(item, nextitem=nextitem, log=False):
                    if report.failed:
                        # Log this one after all, so it gets the ordinary
                        # traceback and failure section.  Its stale look-1
                        # "passed" report is dropped in the terminal summary.
                        self._errors[item.nodeid] = report.longrepr
                        self._state[item.nodeid].active = False
                        item.ihook.pytest_runtest_logreport(report=report)
                        break
        finally:
            self._backfill = False

    def _stop_for_maxfail(self, session: pytest.Session) -> bool:
        "Honour -x / --maxfail, which pytest cannot see through unlogged reruns."
        maxfail = session.config.getvalue("maxfail")
        if not maxfail:
            return False
        rejected = sum(
            1 for st in self._state.values()
            if st.adjusted is not None and st.adjusted <= self.alpha
        )
        if rejected + len(self._errors) >= maxfail:
            session.shouldstop = "maxfail reached during group-sequential looks"
            return True
        return False

    def _decide(self) -> List[tuple]:
        "Apply the correction to every test with a p-value; deactivate rejected."
        # A test that raised on a later look never completed, so — like one that
        # raises before assertNotReject — it is not a member of the family.
        values = [
            (st.value, nodeid, st) for nodeid, st in self._state.items()
            if st.value is not None and nodeid not in self._errors
        ]
        if not values:
            return []
        values.sort(key=lambda triple: triple[0])
        ready = [(nodeid, st) for _, nodeid, st in values]

        if self.correction == "westfall-young" and not self._nulls_open:
            # A Westfall-Young adjusted p-value is never below the raw one, so
            # while every p* exceeds alpha nothing can be rejected and the
            # resampling is wasted work.  Open the gate only when it could
            # change a verdict, then rebuild the columns from look 1.
            if values[0][0] > self.alpha:
                return ready
            self._nulls_open = True
            self._need_backfill = True
            return ready

        adjusted = CORRECTIONS[self.correction](
            [p for p, _, _ in values],
            [st.nulls for _, _, st in values],
        )
        for (_, st), adj in zip(ready, adjusted):
            # Null hypothesis rejected — assertNotReject fails.  Monotonicity of
            # the adjusted values makes this the step-down stop condition, and
            # makes it safe to stop sampling this test now.
            st.adjusted = adj
            if adj <= self.alpha:
                st.active = False
        return ready

    def _apply_correction(self) -> None:
        for nodeid, state in self._decide():
            rejected = state.adjusted is not None and state.adjusted <= self.alpha
            self._corrected.append(_CorrectedResult(
                nodeid=nodeid,
                p_value=state.value,
                adjusted=state.adjusted,
                passed=not rejected and nodeid not in self._errors,
                looks=state.looks_done,
                k_looks=state.k_looks,
            ))

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._apply_correction()
        if any(not r.passed for r in self._corrected):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: pytest.Config) -> None:
        if not self._corrected and not self._errors:
            return

        stats = terminalreporter.stats
        label = CORRECTION_NAMES[self.correction]

        # Re-categorise deferred tests based on the corrected outcome so that
        # the "N passed, M failed" line printed by pytest reflects reality.
        # stats['passed'] holds exactly the call-phase passed reports (pytest
        # files passed setup/teardown under ''), which is the set we may move.
        failed = {r.nodeid: r for r in self._corrected if not r.passed}
        messages = {
            nodeid: (
                f"{label}: p={result.p_value:.6f}, "
                f"adjusted p={result.adjusted:.6f}"
                f" <= α={self.alpha} (null hypothesis rejected)"
            )
            for nodeid, result in failed.items()
            if result.adjusted is not None
        }

        passed_list = stats.get("passed", [])
        for report in [r for r in passed_list if r.nodeid in messages]:
            passed_list.remove(report)
            report.outcome = "failed"
            report.longrepr = messages[report.nodeid]
            stats.setdefault("failed", []).append(report)
        # Tests that raised on a later look already logged a real failure
        # report; drop the stale look-1 "passed" one so they are not counted
        # twice.
        for report in [r for r in passed_list if r.nodeid in self._errors]:
            passed_list.remove(report)

        if not self._corrected:
            return  # every test errored out; nothing was corrected

        n_total = len(self._corrected)
        n_failed = len(failed)
        n_passed = n_total - n_failed

        sequential = f"  groupsize={self.groupsize}" if self.groupsize > 0 else ""
        terminalreporter.write_sep(
            "=",
            f"{label} correction  α={self.alpha}  n={n_total}{sequential}",
        )
        for result in self._corrected:  # already sorted by p-value
            status = "PASSED" if result.passed else "FAILED"
            # Only tests whose p* could not have been rejected skip resampling,
            # so there is no adjusted value to show for them.
            adjusted = (
                f"adjusted p={result.adjusted:.6f}"
                if result.adjusted is not None else "adjusted p>α"
            )
            looks = (
                f"  looks={result.looks}/{result.k_looks}"
                if self.groupsize > 0 else ""
            )
            terminalreporter.write_line(
                f"  {status}  p={result.p_value:.6f}  "
                f"{adjusted}{looks}  {result.nodeid}"
            )
        terminalreporter.write_line(
            f"\n  {n_passed} passed, {n_failed} failed "
            f"after {label} correction"
        )

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Family-wise error rate for the correction procedure (default: 0.05)",
    )
    parser.addoption(
        "--correction",
        choices=sorted(CORRECTIONS),
        default="holm",
        help=(
            "Multiple-comparison procedure (default: holm).  'westfall-young' "
            "requires each test to pass a null_sample callable to assertNotReject."
        ),
    )
    parser.addoption(
        "--resamples",
        type=int,
        default=1000,
        metavar="B",
        help=(
            "Number of null resamples per test for --correction=westfall-young "
            "(default: 1000).  Must exceed 1/alpha for any rejection to be "
            "possible.  Ignored by holm."
        ),
    )
    parser.addoption(
        "--power",
        type=float,
        default=0.8,
        metavar="POWER",
        help=(
            "Per-test true positive rate (power) used by the sample-size fixtures "
            "(default: 0.8).  This is per-test, not family-wise."
        ),
    )
    parser.addoption(
        "--groupsize",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Samples per group for group-sequential testing (default: 0, "
            "disabled).  Tests draw one group at a time and stop as soon as "
            "their null is rejected; the number of groups follows from the "
            "sample-size fixtures.  Requires assertNotReject to be given a "
            "sampler and the test to use a sample-size fixture."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    groupsize = config.getoption("--groupsize")
    if groupsize < 0:
        raise pytest.UsageError("--groupsize must be >= 0")
    if groupsize > 0 and config.getoption("dist", "no") != "no":
        # Later looks re-run the test protocol from pytest_runtestloop, which
        # xdist replaces outright.
        raise pytest.UsageError("--groupsize is not supported under pytest-xdist")

    plugin = FamilywisePlugin(
        alpha=config.getoption("--alpha"),
        power=config.getoption("--power"),
        correction=config.getoption("--correction"),
        resamples=config.getoption("--resamples"),
        groupsize=groupsize,
    )
    config.pluginmanager.register(plugin, "familywise-correction")
    config._familywise_plugin = plugin  # type: ignore[attr-defined]


@pytest.fixture
def assertNotReject(request: pytest.FixtureRequest) -> PValueReporter:
    """Assert that the null hypothesis is not rejected after FWER correction.

    Call the returned object once inside your test with a **sampler**: a
    callable that draws one group of data from the generator it is handed and
    returns the p-value for that group.  The test passes if the null hypothesis
    survives correction; it fails if H0 is rejected.

    ```python
    def test_mean_zero(ztest_sample_size, assertNotReject):
        n = ztest_sample_size(effect_size=0.3)

        def sample(rng):
            return scipy.stats.ttest_1samp(rng.standard_normal(n), 0.0).pvalue

        assertNotReject(sample)
    ```

    Without ``--groupsize`` the sampler is called exactly once and its p-value
    is used directly.  Under ``--groupsize`` it is called once per look, on
    fresh data each time, until the null is rejected or the planned number of
    groups is exhausted; the group p-values are combined into a single
    sequential p-value against a Pocock boundary.  **Draw your data from the
    ``rng`` argument.**  A sampler that reads its data from a module- or
    session-scoped fixture would see the same observations at every look, and
    the groups would not be independent.

    Under ``--correction=westfall-young`` a second argument is required: a
    callable that recomputes *this test's* p-value under H0.  It receives a
    seeded ``numpy`` generator and must transform the **observed** data so as to
    impose H0 — permute labels, sign-flip, center-then-bootstrap — rather than
    drawing fresh data from an assumed-true model.

    ```python
    def test_mean_zero(assertNotReject, data):
        def sample(rng):
            return scipy.stats.ttest_1samp(data, 0.0).pvalue

        def under_h0(rng):
            centered = data - data.mean()          # impose H0
            boot = rng.choice(centered, size=len(data))
            return scipy.stats.ttest_1samp(boot, 0.0)[1]

        assertNotReject(sample, null_sample=under_h0)
    ```

    The ``rng`` for resample *b* is seeded identically in every test, so two
    tests that apply the same transform to the same data see the same draw and
    their null columns stay aligned — which is what lets the procedure exploit
    their correlation.
    """
    plugin: FamilywisePlugin = request.config._familywise_plugin  # type: ignore[attr-defined]
    return PValueReporter(plugin, request.node.nodeid)


@pytest.fixture
def ztest_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a z-test at the session's power and corrected alpha.

    The significance level used is ``alpha / m``, where *m* is the total number
    of ``assertNotReject`` tests.  That is the strictest threshold the step-down
    procedure can apply, and the one a test with a real effect actually faces:
    its p-value is small, so it sorts to rank 1.  Tests that do not use
    ``assertNotReject`` are never corrected and size against the raw alpha.

    Usage:

    ```python
    def test_mean(ztest_sample_size, assertNotReject):
        n = ztest_sample_size(effect_size=0.5)          # two-sided
        n = ztest_sample_size(effect_size=0.5, two_sided=False)
        data = generate(n)
        _, p = scipy.stats.ttest_1samp(data, 0)
        assertNotReject(p)
    ```

    ``effect_size`` is Cohen's d (mean difference / pooled SD).
    Returns per-group n for a two-sample test.
    """
    plugin: FamilywisePlugin = request.config._familywise_plugin  # type: ignore[attr-defined]
    nodeid = request.node.nodeid
    alpha = plugin.get_corrected_alpha(nodeid)

    def compute(effect_size: float, two_sided: bool = True) -> int:
        return plugin.plan(
            nodeid,
            lambda a, power: _ztest_n(a, power, effect_size, two_sided),
            alpha,
        )

    return compute


@pytest.fixture
def chisquare_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a chi-square goodness-of-fit test.

    Uses the same corrected alpha as ``ztest_sample_size``
    (see its docstring for details).

    Usage:

    ```python
    def test_distribution(chisquare_sample_size, assertNotReject):
        n = chisquare_sample_size(effect_size=0.3, df=4)
        counts = generate(n)
        _, p = scipy.stats.chisquare(counts, expected)
        assertNotReject(p)
    ```

    ``effect_size`` is Cohen's w; ``df`` is the degrees of freedom
    (number of categories − 1 for goodness-of-fit).
    """
    plugin: FamilywisePlugin = request.config._familywise_plugin  # type: ignore[attr-defined]
    nodeid = request.node.nodeid
    alpha = plugin.get_corrected_alpha(nodeid)

    def compute(effect_size: float, df: int) -> int:
        return plugin.plan(
            nodeid,
            lambda a, power: _chisquare_n(a, power, effect_size, df),
            alpha,
        )

    return compute


@pytest.fixture
def ks_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a KS test, sized via the DKW inequality.

    Uses the same corrected alpha as ``ztest_sample_size``
    (see its docstring for details).

    Usage:

    ```python
    def test_uniform(ks_sample_size, assertNotReject):
        n = ks_sample_size(effect_size=0.1)              # one-sample
        n = ks_sample_size(effect_size=0.1, two_sample=True)  # per-group
        data = generate(n)
        p = scipy.stats.kstest(data, 'uniform').pvalue
        assertNotReject(p)
    ```

    ``effect_size`` is the maximum absolute CDF difference ||F − G||_∞.
    When ``two_sample=True`` the returned value is the required per-group n
    (assuming equal group sizes).
    """
    plugin: FamilywisePlugin = request.config._familywise_plugin  # type: ignore[attr-defined]
    nodeid = request.node.nodeid
    alpha = plugin.get_corrected_alpha(nodeid)

    def compute(effect_size: float, two_sample: bool = False) -> int:
        return plugin.plan(
            nodeid,
            lambda a, power: _ks_n(a, power, effect_size, two_sample),
            alpha,
        )

    return compute
