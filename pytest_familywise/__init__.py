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
from typing import Callable, Dict, List, Optional, Sequence, Set

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

class PValueReporter:
    "Callable returned by the ``assertNotReject`` fixture."

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
            self.nulls = np.fromiter(
                (
                    _validated(null_sample(default_rng(_BASE_SEED + b)))
                    for b in range(self._resamples)
                ),
                dtype=float,
                count=self._resamples,
            )


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
    adjusted: float
    passed: bool

@dataclass(eq=False)
class FamilywisePlugin:
    "Collects p-values, defers pass/fail, and applies the FWER correction."

    alpha: float
    power: float
    correction: str
    resamples: int

    _reporters: Dict[str, PValueReporter] = field(init=False, default_factory=dict)
    _corrected: List[_CorrectedResult] = field(init=False, default_factory=list)
    # Tests participating in the correction (i.e. using assertNotReject).
    _participating: Set[str] = field(init=False, default_factory=set)

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

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._apply_correction()
        if any(not r.passed for r in self._corrected):
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: pytest.Config) -> None:
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


def pytest_configure(config: pytest.Config) -> None:
    plugin = FamilywisePlugin(
        alpha=config.getoption("--alpha"),
        power=config.getoption("--power"),
        correction=config.getoption("--correction"),
        resamples=config.getoption("--resamples"),
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
    drawing fresh data from an assumed-true model.

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
    their correlation.
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
