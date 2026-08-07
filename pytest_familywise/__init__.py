"""Pytest plugin for family-wise error rate control of randomized tests.

Tests register a p-value via the ``assertNotReject`` fixture.  After all tests
run, the plugin applies a step-down multiple-comparison procedure to control the
family-wise error rate (FWER).  A test "passes" when its p-value is too large to
reject the null hypothesis after correction; it "fails" when the null hypothesis
is rejected.

Two procedures are available via ``--correction``:

* ``holm`` (default) – Holm-Bonferroni.  Assumes nothing about the dependence
  between tests, and is correspondingly conservative when they are correlated.
* ``westfall-young`` – resampling step-down (minP).  Estimates the *joint* null
  distribution of the p-value vector rather than bounding it, recovering the
  power Holm gives up.  Requires each test to supply a ``null_sample`` callable
  (see ``assertNotReject``).

Both are expressed as adjusted p-values: each test's raw p-value is mapped to an
adjusted value, and the null hypothesis is rejected when ``adjusted <= alpha``.

Three fixtures expose required-sample-size calculations so tests can be
sized for the desired per-test power before running:

* ``ztest_sample_size(effect_size, two_sided=True)``  – Cohen's d
* ``chisquare_sample_size(effect_size, df)``           – Cohen's w
* ``ks_sample_size(effect_size, two_sample=False)``    – max |F−G|, via DKW

The sample-size fixtures automatically use corrected significance levels rather
than the raw familywise alpha.  The plugin records which tests use
``assertNotReject`` at collection time; the *k*-th of those *m* tests, in
collection order, sizes against ``alpha / (m - k + 1)``.  This ensures each test
is properly powered for a threshold it may face during the step-down procedure.
Tests outside that set are never corrected, so they size against the raw alpha.

Calibration
-----------
Holm's ``alpha / (m - k + 1)`` is conservative under ``westfall-young``, whose
real critical values depend on the null distribution the tests generate and so
are unknown before they run.  ``--calibration=PATH`` breaks that circularity: a
westfall-young run with no calibration file records every test's null column to
one, and later runs load it and size against the *measured* ladder

    c_k = the alpha-quantile of the per-resample minimum null p-value
          over the tests at collection positions k..m

which is the threshold the rank-k hypothesis faces under the step-down
procedure.  Theory brackets each rung as ``alpha/(m-k+1) <= c_k <= alpha``, and
the estimates are clamped into that range, so calibrated sizing can only shrink
sample sizes -- rung by rung, against Holm's own rung.

This only affects **power**.  FWER control always comes from the current run's
own null draws, so a missing, stale or corrupt calibration costs sample size and
nothing else -- every failure path warns and falls back to Holm.

Two things worth knowing before enabling it:

* The gain comes entirely from *positive dependence between tests*.  Independent
  tests give ``c_k ~ alpha/(m-k+1)``, i.e. Holm, i.e. no gain at all -- the win
  needs tests that share a dataset or an RNG.
* Resolving a gain takes resamples.  The estimate carries roughly
  ``1/sqrt(B*alpha)`` relative error and is deliberately biased low, so a modest
  real gain can be swallowed at small *B*.  When that happens the run says
  ``no dependence resolved at B=...`` rather than silently reporting Holm's own
  value; raise ``--resamples``.

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
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2
import pytest

# Base seed for Westfall-Young resampling.  Fixed rather than configurable so
# that runs are reproducible by default; draw b uses default_rng(_BASE_SEED + b)
# identically in every test, which is what aligns the null columns.
_BASE_SEED = 0x5EED


# ---------------------------------------------------------------------------
# Sample-size helpers (pure functions, no pytest state)
# ---------------------------------------------------------------------------

def _ztest_n(alpha: float, power: float, effect_size: float, two_sided: bool = True) -> int:
    """Minimum n for a one-sample z-test (or two-sample with equal group sizes).

    Parameters
    ----------
    effect_size:
        Cohen's d – the expected difference in means divided by the
        pooled standard deviation.
    two_sided:
        Whether to use a two-sided test (default True).

    Returns
    -------
    int
        Required sample size (per group for a two-sample test).
    """
    z_alpha = norm.ppf(1 - alpha / 2) if two_sided else norm.ppf(1 - alpha)
    z_beta = norm.ppf(power)
    return math.ceil(((z_alpha + z_beta) / effect_size) ** 2)


def _chisquare_n(alpha: float, power: float, effect_size: float, df: int) -> int:
    """Minimum n for a chi-square goodness-of-fit test.

    Parameters
    ----------
    effect_size:
        Cohen's w = sqrt(sum((p_i - p0_i)^2 / p0_i)).
    df:
        Degrees of freedom (number of categories minus 1 for goodness-of-fit).

    Returns
    -------
    int
        Required total sample size.
    """

    critical = chi2.ppf(1 - alpha, df)

    def shortfall(n: float) -> float:
        return ncx2.sf(critical, df, n * effect_size ** 2) - power

    # Double upper bound until the power is achievable.
    hi = 4.0
    while shortfall(hi) < 0:
        hi *= 2
    return math.ceil(brentq(shortfall, 1.0, hi))


def _ks_n(alpha: float, power: float, effect_size: float, two_sample: bool = False) -> int:
    """Minimum n for a Kolmogorov-Smirnov test, via the DKW inequality.

    Uses the bound::

        n >= (sqrt(ln(2/alpha)) + sqrt(ln(2/beta)))^2 / (2 * delta^2)

    where ``beta = 1 - power`` and ``delta = effect_size``.

    Parameters
    ----------
    effect_size:
        Maximum absolute CDF difference under H1 (i.e. ||F − G||_∞ ∈ (0, 1]).
    two_sample:
        If True, return the per-group sample size for a two-sample test
        (assuming equal group sizes).  The effective n for the two-sample KS
        statistic is n1*n2/(n1+n2) = n_each/2 when groups are equal.

    Returns
    -------
    int
        Required sample size (per group when ``two_sample=True``).
    """
    beta = 1.0 - power
    delta = effect_size
    n = (math.sqrt(math.log(2.0 / alpha)) + math.sqrt(math.log(2.0 / beta))) ** 2 / (
        2.0 * delta ** 2
    )
    n_ceil = math.ceil(n)
    # For two equal groups the effective n is n_each/2, so double.
    return n_ceil * 2 if two_sample else n_ceil


# ---------------------------------------------------------------------------
# Public fixture object
# ---------------------------------------------------------------------------

def _validated(p: float) -> float:
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p-value must be in [0, 1], got {p!r}")
    return float(p)


class PValueReporter:
    """Callable returned by the ``assertNotReject`` fixture.

    The test calls it once with its computed p-value:

    ```python
    def test_foo(assertNotReject):
        p = run_experiment()
        assertNotReject(p)
    ```

    Under ``--correction=westfall-young`` a ``null_sample`` callable is also
    required; see the ``assertNotReject`` fixture docstring.
    """

    def __init__(self, needs_null: bool = False, resamples: int = 0) -> None:
        self.value: Optional[float] = None
        self.nulls: Optional["np.ndarray"] = None  # (B,) float64
        self._needs_null = needs_null
        self._resamples = resamples

    def __call__(
        self,
        p: float,
        null_sample: Optional[Callable[..., float]] = None,
    ) -> None:
        if self._needs_null and null_sample is None:
            raise ValueError(
                "the westfall-young correction requires "
                "assertNotReject(p, null_sample=fn), where fn(rng) recomputes "
                "this test's p-value on an H0-imposing transform of the "
                "observed data (permute, sign-flip, center-then-bootstrap)"
            )
        self.value = _validated(p)
        if null_sample is not None:
            # Draw b uses an rng seeded identically across every test, so tests
            # that apply the same transform to the same data produce aligned
            # (correlated) null columns -- which is where WY's power comes from.
            # float64 array rather than a list: 4x less memory held until
            # sessionfinish (32 MB vs 131 MB at m=100, B=40000) and no conversion
            # on the way into the correction.  Not float32 -- rounding there could
            # flip a `null <= p` comparison and change which tests are rejected.
            self.nulls = np.fromiter(
                (
                    _validated(null_sample(default_rng(_BASE_SEED + b)))
                    for b in range(self._resamples)
                ),
                dtype=float,
                count=self._resamples,
            )


# ---------------------------------------------------------------------------
# Correction procedures: raw p-values -> adjusted p-values
#
# Both are step-down procedures.  Expressing them as adjusted p-values (reject
# iff adjusted <= alpha) makes the step-down stop condition and monotonicity
# fall out of the values themselves, so all the pytest plumbing is shared.
# ---------------------------------------------------------------------------

def _holm_adjusted(
    pvalues: Sequence[float],
    nulls: Sequence[Optional[Sequence[float]]] = (),  # unused; shared with WY
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


# ---------------------------------------------------------------------------
# Calibration: reuse a recorded null matrix to size samples against the
# Westfall-Young critical value instead of Holm's conservative bound.
#
# This only ever affects POWER.  FWER control comes from _apply_correction,
# which recomputes the correction from the current run's own null draws, so a
# stale or missing calibration yields an underpowered test, never an invalid
# one.  Every failure path below is therefore "warn and fall back to Holm".
# ---------------------------------------------------------------------------

CALIBRATION_VERSION = 2

DEFAULT_CALIBRATION = ".familywise-calibration.npz"

# The alpha-quantile of the minima is estimated from about B*alpha order
# statistics; relative error is ~1/sqrt(B*alpha).  Below 100 the estimate is too
# noisy to size against (32% at B=1000, alpha=0.01).
MIN_ORDER_STATS = 100


class _CalibrationError(Exception):
    """Unusable calibration file.  Carries a human-readable reason."""


@dataclass
class _Calibration:
    resamples: int
    columns: Dict[str, "np.ndarray"]  # nodeid -> (B,) null p-values


def _conservative_quantile(minima: "np.ndarray", alpha: float) -> "np.ndarray":
    """Lower confidence bounds on the alpha-quantile of each row of ``minima``.

    The alpha-quantile sits at order statistic ``B*alpha``, whose standard error
    is ``sqrt(B*alpha*(1-alpha))``.  Stepping two of those down biases the
    estimate small, and for sizing "small" is the safe direction: a smaller
    critical value means a larger sample.  Without this an unlucky pilot run
    silently under-sizes the whole suite.

    ``minima`` is ``(rows, B)``; the order statistic index depends only on B and
    alpha, so every row is selected in one partition.
    """
    b = minima.shape[1]
    target = b * alpha
    # -1 converts the 1-based order statistic to a 0-based index.
    index = int(math.floor(target - 2.0 * math.sqrt(target * (1.0 - alpha)))) - 1
    index = max(0, min(b - 1, index))
    return np.partition(minima, index, axis=1)[:, index].astype(float)


def _calibrated_alphas(
    calibration: _Calibration,
    nodeids: Sequence[str],
    alpha: float,
) -> List[float]:
    """The Westfall-Young critical value ladder for a family of ``nodeids``.

    The step-down procedure compares the rank-k p-value against the
    alpha-quantile of the minimum null p-value over the hypotheses still
    standing at step k.  So there is a whole ladder of critical values, not one:

        c_k = alpha-quantile of  min over {k, ..., m} of the null columns

    with ``nodeids`` in collection order, exactly the convention
    ``get_corrected_alpha`` already uses for Holm.  ``c_k`` is the calibrated
    counterpart of Holm's ``alpha / (m - k + 1)``, and is monotone increasing in
    k for free: the minimum over a superset is never larger.

    Recomputed over the nodeids actually present in this run, so selecting a
    subset (``-k``, ``--lf``, one file) still uses the cache rather than falling
    back.

    Raises ``_CalibrationError`` if any nodeid is missing or B*alpha is too small
    for the quantile to mean anything.
    """
    missing = [n for n in nodeids if n not in calibration.columns]
    if missing:
        raise _CalibrationError(
            f"{len(missing)} test(s) absent from the calibration, "
            f"first: {missing[0]}"
        )
    b = calibration.resamples
    if b * alpha < MIN_ORDER_STATS:
        raise _CalibrationError(
            f"--resamples={b} too small at alpha={alpha}: needs "
            f"{math.ceil(MIN_ORDER_STATS / alpha)} for a stable quantile"
        )

    matrix = np.stack([calibration.columns[n] for n in nodeids])  # (m, B)
    # Row k of tail_min is the per-resample minimum over the tail {k, ..., m}:
    # reverse, running minimum, reverse back -- the same trick the correction
    # itself uses, applied to the recorded matrix instead of the live one.
    tail_min = np.minimum.accumulate(matrix[::-1])[::-1]
    c = _conservative_quantile(tail_min, alpha)

    # Theory brackets each rung as alpha/(m-k+1) <= c_k <= alpha, whatever the
    # dependence.  For a tail set S: Bonferroni's union bound gives
    # P(min_S <= alpha/|S|) <= alpha for the lower end, and
    # P(min_S <= alpha) >= P(p_1 <= alpha) = alpha for the upper.  Clamping into
    # that bracket removes estimation error wherever the answer is already
    # known, and guarantees each rung is never stricter than Holm's own rung --
    # so calibrated sizing can only shrink n, test by test.
    m = len(nodeids)
    holm = alpha / (m - np.arange(m))  # alpha / (m - k + 1) for k = 1..m
    # 1/B floor: the procedure cannot produce an adjusted p-value below 1/B, so
    # sizing for anything smaller targets a threshold it can never attain.
    return np.minimum(alpha, np.maximum(np.maximum(c, holm), 1.0 / b)).tolist()


def _calibration_mode(
    path: Optional[str],
    correction: str,
    is_xdist_worker: bool,
) -> str:
    """Whether this run records a calibration, loads one, or ignores them.

    Only westfall-young generates null draws, so only it can record -- and only
    its critical values may be used for sizing.  Loading a stale file under
    ``holm`` would size against c1 >= alpha/m while the run actually applies
    Holm's stricter ladder, which is the one direction sizing must never go.

    xdist workers each collect a subset, so every worker would write a partial
    matrix and the last would win; they neither record nor load.
    """
    if not path or correction != "westfall-young" or is_xdist_worker:
        return "off"
    return "load" if os.path.exists(path) else "record"


def _load_calibration(path: str) -> _Calibration:
    """Read a calibration file, raising ``_CalibrationError`` if unusable."""
    keys = ("version", "resamples", "base_seed", "nodeids", "nulls")
    try:
        with np.load(path) as data:
            arrays = {k: data[k] for k in keys}
    except FileNotFoundError:
        raise _CalibrationError("not found")
    except Exception as exc:  # unreadable, truncated, not an npz, missing key
        raise _CalibrationError(f"unreadable ({type(exc).__name__}: {exc})")

    version, seed = int(arrays["version"]), int(arrays["base_seed"])
    if version != CALIBRATION_VERSION:
        raise _CalibrationError(
            f"format version {version}, expected {CALIBRATION_VERSION}"
        )
    if seed != _BASE_SEED:
        raise _CalibrationError(
            f"recorded with base seed {seed}, this build uses {_BASE_SEED}"
        )

    resamples, nulls = int(arrays["resamples"]), arrays["nulls"]
    nodeids = [str(n) for n in arrays["nodeids"]]
    if nulls.shape != (len(nodeids), resamples):
        raise _CalibrationError(
            f"expected a {(len(nodeids), resamples)} matrix, got {nulls.shape}"
        )
    return _Calibration(resamples=resamples, columns=dict(zip(nodeids, nulls)))


def _write_calibration(
    path: str,
    resamples: int,
    columns: Mapping[str, Sequence[float]],
) -> None:
    nodeids = sorted(columns)
    # Write through a file object: np.savez_compressed silently appends ".npz"
    # when handed a path, which would not match what we later try to load.
    with open(path, "wb") as handle:
        np.savez_compressed(
            handle,
            version=CALIBRATION_VERSION,
            resamples=resamples,
            base_seed=_BASE_SEED,
            nodeids=np.array(nodeids),
            nulls=np.array([columns[n] for n in nodeids], dtype=np.float32),
        )


# ---------------------------------------------------------------------------
# Internal result record
# ---------------------------------------------------------------------------

@dataclass
class _CorrectedResult:
    nodeid: str
    p_value: float
    adjusted: float
    passed: bool


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@dataclass(eq=False)
class FamilywisePlugin:
    """Collects p-values, defers pass/fail, and applies the FWER correction.

    No field carries a default: every value comes from the command line, whose
    defaults live in ``pytest_addoption`` and nowhere else.
    """

    alpha: float
    power: float
    correction: str
    resamples: int
    calibration_path: Optional[str]
    is_xdist_worker: bool

    mode: str = field(init=False)
    # The c_k ladder from a loaded calibration, indexed by collection position;
    # None means size with Holm's ladder.
    _sizing_alphas: Optional[List[float]] = field(init=False, default=None)
    # Terminal-summary lines, in the order they were decided.
    _notes: List[str] = field(init=False, default_factory=list)
    _reporters: Dict[str, PValueReporter] = field(init=False, default_factory=dict)
    _corrected: List[_CorrectedResult] = field(init=False, default_factory=list)
    # Tests participating in the correction (i.e. using assertNotReject).
    _participating: List[str] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.mode = _calibration_mode(
            self.calibration_path, self.correction, self.is_xdist_worker
        )
        if (
            self.calibration_path
            and self.correction == "westfall-young"
            and self.is_xdist_worker
        ):
            self._notes.append(
                "cannot record under xdist; use -p no:xdist to record one"
            )

    # ------------------------------------------------------------------
    # Hook: record which tests participate in the correction
    # ------------------------------------------------------------------

    # trylast: pytest's own -k / -m deselection happens in this same hook, so
    # running first would count tests that are about to be deselected and
    # inflate m for the whole family.
    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, items: List[pytest.Item]) -> None:
        self._participating = [
            item.nodeid
            for item in items
            if "assertNotReject" in getattr(item, "fixturenames", ())
        ]
        self._load_sizing_alpha()

    def _load_sizing_alpha(self) -> None:
        """Settle the sizing threshold once, now that the family is known."""
        if self.mode == "record":
            self._notes.append("recording a new calibration; sizing with Holm")
        if self.mode != "load" or not self._participating:
            return

        m = len(self._participating)
        try:
            calibration = _load_calibration(self.calibration_path or "")
            ladder = _calibrated_alphas(
                calibration, self._participating, self.alpha
            )
        except _CalibrationError as exc:
            self._notes.append(f"{exc}; sizing with Holm")
            return
        self._sizing_alphas = ladder
        holm = self.alpha / m
        self._notes.append(
            f"sizing alpha={ladder[0]:.6f}..{ladder[-1]:.6f} for {m} tests "
            f"(Holm would use {holm:.6f}..{self.alpha:.6f})"
        )
        # Rung 1 only: the top rung is the alpha-quantile of a single column, so
        # it sits at alpha under any dependence and would always read as
        # "collapsed".  Rung 1 is where a real gain shows up.
        if m > 1 and math.isclose(ladder[0], holm):
            # The estimate fell to the Bonferroni clamp, so either the tests
            # really are independent or B is too small to resolve the gain.
            # Say which knob to turn rather than reporting a silent no-op.
            self._notes.append(
                f"no dependence resolved at B={self.resamples}; "
                f"raise --resamples if these tests share data"
            )

    # ------------------------------------------------------------------
    # Corrected alpha for sample-size fixtures
    # ------------------------------------------------------------------

    def get_corrected_alpha(self, nodeid: str) -> float:
        """Return the corrected alpha for a test's sample-size calculation.

        Both paths are the same ladder, indexed by the test's 1-based position k
        among the m participating tests in collection order.  Without a
        calibration that ladder is Holm's ``alpha / (m - k + 1)``; with one it is
        the measured ``c_k`` (see ``_calibrated_alphas``), which brackets above
        Holm's rung k -- so loading a calibration can only shrink n, test by
        test.

        Keying k off collection order rather than the order in which tests happen
        to *ask* for a size matters twice over: tests outside the family cannot
        consume a rung of the ladder, and the answer no longer depends on which
        tests ran, so it is stable under -k, --lf and pytest-randomly.  Since
        collection order is execution order by default, ordering expensive tests
        last still earns them the smaller samples.

        The convention is an approximation in both cases: a test is sized for the
        rung matching its collection position, but at run time it faces the rung
        matching its observed p-value *rank*.  A test with a real effect tends to
        land at rank 1 and face the harshest rung whatever its position.
        """
        if nodeid not in self._participating:
            # Not part of the correction family -- this test's p-value is never
            # adjusted, so no correction applies to its sample size either.
            return self.alpha

        m = len(self._participating)
        k = self._participating.index(nodeid) + 1
        if self._sizing_alphas is not None:
            return self._sizing_alphas[k - 1]
        return self.alpha / (m - k + 1)

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------

    def _apply_correction(self) -> None:
        pvalue_items = [
            (nodeid, r.value, r.nulls)
            for nodeid, r in self._reporters.items()
            if r.value is not None
        ]
        if not pvalue_items:
            return

        sorted_items = sorted(pvalue_items, key=lambda x: x[1])
        adjust = CORRECTIONS[self.correction]
        adjusted = adjust(
            [p for _, p, _ in sorted_items],
            [nulls for _, _, nulls in sorted_items],
        )

        for (nodeid, p, _), adj in zip(sorted_items, adjusted):
            # Null hypothesis rejected — assertNotReject fails.  Monotonicity of
            # the adjusted values makes this the step-down stop condition.
            self._corrected.append(_CorrectedResult(
                nodeid=nodeid,
                p_value=p,
                adjusted=adj,
                passed=adj > self.alpha,
            ))

    # ------------------------------------------------------------------
    # Hook: run correction and update session exit status
    # ------------------------------------------------------------------

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._apply_correction()
        self._maybe_write_calibration()
        if any(not r.passed for r in self._corrected):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def _maybe_write_calibration(self) -> None:
        """Record the null matrix, if this is a recording run that collected one."""
        if self.mode != "record":
            return
        columns = {
            nodeid: r.nulls
            for nodeid, r in self._reporters.items()
            if r.nulls is not None
        }
        if not columns:
            return
        _write_calibration(
            self.calibration_path or "", self.resamples, columns
        )
        self._notes.append(
            f"wrote {self.calibration_path} ({len(columns)} tests, "
            f"B={self.resamples})"
        )

    # ------------------------------------------------------------------
    # Hook: update terminal stats and print summary
    # ------------------------------------------------------------------

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: pytest.Config) -> None:
        for note in self._notes:
            terminalreporter.write_line(f"calibration: {note}")
        if not self._corrected:
            return

        stats = terminalreporter.stats
        label = CORRECTION_NAMES[self.correction]

        # Re-categorise deferred tests based on the corrected outcome so that
        # the "N passed, M failed" line printed by pytest reflects reality.
        # stats['passed'] holds exactly the call-phase passed reports (pytest
        # files passed setup/teardown under ''), which is the set we may move.
        failed = {r.nodeid: r for r in self._corrected if not r.passed}
        passed_list = stats.get("passed", [])
        for report in [r for r in passed_list if r.nodeid in failed]:
            result = failed[report.nodeid]
            passed_list.remove(report)
            report.outcome = "failed"
            report.longrepr = (
                f"{label}: p={result.p_value:.6f}, "
                f"adjusted p={result.adjusted:.6f}"
                f" <= α={self.alpha} (null hypothesis rejected)"
            )
            stats.setdefault("failed", []).append(report)

        n_total = len(self._corrected)
        n_failed = len(failed)
        n_passed = n_total - n_failed

        terminalreporter.write_sep(
            "=",
            f"{label} correction  α={self.alpha}  n={n_total}",
        )
        for result in self._corrected:  # already sorted by p-value
            status = "PASSED" if result.passed else "FAILED"
            terminalreporter.write_line(
                f"  {status}  p={result.p_value:.6f}  "
                f"adjusted p={result.adjusted:.6f}  {result.nodeid}"
            )
        terminalreporter.write_line(
            f"\n  {n_passed} passed, {n_failed} failed "
            f"after {label} correction"
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

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
        "--calibration",
        default=DEFAULT_CALIBRATION,
        metavar="PATH",
        help=(
            f"Calibration file of recorded null p-values (default: "
            f"{DEFAULT_CALIBRATION}).  When present, the sample-size "
            "fixtures size against the measured Westfall-Young critical value "
            "instead of Holm's conservative bound; when absent, a "
            "westfall-young run records one -- so delete the file to "
            "recalibrate.  Pass an empty string to disable."
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


def pytest_configure(config: pytest.Config) -> None:
    plugin = FamilywisePlugin(
        alpha=config.getoption("--alpha"),
        power=config.getoption("--power"),
        correction=config.getoption("--correction"),
        resamples=config.getoption("--resamples"),
        calibration_path=config.getoption("--calibration"),
        # Set by pytest-xdist on worker processes only.
        is_xdist_worker=hasattr(config, "workerinput"),
    )
    config.pluginmanager.register(plugin, "familywise-correction")
    config._familywise_plugin = plugin  # type: ignore[attr-defined]


@pytest.fixture
def assertNotReject(request: pytest.FixtureRequest) -> PValueReporter:
    """Assert that the null hypothesis is not rejected after FWER correction.

    Call the returned object once inside your test with the computed p-value.
    The test passes if the p-value is too large to reject H0; it fails if
    H0 is rejected.

    Example:

    ```python
    def test_chi_squared(assertNotReject):
        stat, p = scipy.stats.chisquare(observed, expected)
        assertNotReject(p)
    ```

    Under ``--correction=westfall-young`` a second argument is required: a
    callable that recomputes *this test's* p-value under H0.  It receives a
    seeded ``numpy`` generator and must transform the **observed** data so as to
    impose H0 — permute labels, sign-flip, center-then-bootstrap — rather than
    drawing fresh data from an assumed-true model (which is only valid when H0
    actually holds, and throws away the dependence structure Westfall-Young
    exists to exploit):

    ```python
    def test_mean_zero(assertNotReject, data):
        _, p = scipy.stats.ttest_1samp(data, 0.0)

        def under_h0(rng):
            centered = data - data.mean()          # impose H0
            boot = rng.choice(centered, size=len(data))
            return scipy.stats.ttest_1samp(boot, 0.0)[1]

        assertNotReject(p, null_sample=under_h0)
    ```

    The ``rng`` for resample *b* is seeded identically in every test, so two
    tests that apply the same transform to the same data see the same draw and
    their null columns stay aligned — which is what lets the procedure exploit
    their correlation.  Tests whose transforms differ still get valid results,
    just with less of a power gain over Holm.
    """
    plugin: FamilywisePlugin = request.config._familywise_plugin  # type: ignore[attr-defined]
    reporter = PValueReporter(
        needs_null=plugin.correction == "westfall-young",
        resamples=plugin.resamples,
    )
    plugin._reporters[request.node.nodeid] = reporter
    return reporter


@pytest.fixture
def ztest_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a z-test at the session's power and corrected alpha.

    The significance level used is ``alpha / (m - k + 1)`` where *m* is the
    total number of ``assertNotReject`` tests and *k* is the position (in
    execution order) at which this test requests a sample size.  This matches
    the Holm-Bonferroni step-down thresholds so each test is properly powered.

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
        return _ztest_n(alpha, plugin.power, effect_size, two_sided)

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
        return _chisquare_n(alpha, plugin.power, effect_size, df)

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
        return _ks_n(alpha, plugin.power, effect_size, two_sample)

    return compute
