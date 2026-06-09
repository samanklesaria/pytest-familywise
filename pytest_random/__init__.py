"""Pytest plugin for Holm-Bonferroni correction of randomized tests.

Tests register a p-value via the ``pvalue`` fixture.  After all tests run,
the plugin sorts the p-values and applies the Holm-Bonferroni step-down
procedure to control the family-wise error rate (FWER).  A test "passes" when
its p-value is small enough to reject the null hypothesis after correction;
it "fails" otherwise.

Three fixtures expose required-sample-size calculations so tests can be
sized for the desired per-test power before running:

* ``ztest_sample_size(effect_size, two_sided=True)``  – Cohen's d
* ``chisquare_sample_size(effect_size, df)``           – Cohen's w
* ``ks_sample_size(effect_size, two_sample=False)``    – max |F−G|, via DKW

Loading
-------
The package registers itself with pytest via the ``pytest11`` entry point in
``pyproject.toml``::

    [project.entry-points."pytest11"]
    random = "pytest_random"

Installing the package is sufficient — pytest discovers the entry point at
startup and loads the plugin automatically.  No ``conftest.py`` import is
required in the project under test.

If you are working from a source checkout without installing the package,
add the following to your project's ``conftest.py`` instead::

    pytest_plugins = ["pytest_random"]
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set

import pytest

__all__ = [
    "PValueReporter",
    "pvalue",
    "ztest_sample_size",
    "chisquare_sample_size",
    "ks_sample_size",
]


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
    from scipy.stats import norm
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
    from scipy.optimize import brentq
    from scipy.stats import chi2, ncx2

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

class PValueReporter:
    """Callable returned by the ``pvalue`` fixture.

    The test calls it once with its computed p-value::

        def test_foo(pvalue):
            p = run_experiment()
            pvalue(p)
    """

    def __init__(self) -> None:
        self.value: Optional[float] = None

    def __call__(self, p: float) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-value must be in [0, 1], got {p!r}")
        self.value = p


# ---------------------------------------------------------------------------
# Internal result record
# ---------------------------------------------------------------------------

@dataclass
class _CorrectedResult:
    nodeid: str
    p_value: float
    threshold: float
    passed: bool


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class HolmBonferroniPlugin:
    """Collects p-values, defers pass/fail, and applies Holm-Bonferroni."""

    def __init__(self, alpha: float, power: float) -> None:
        self.alpha = alpha
        self.power = power
        self._reporters: Dict[str, PValueReporter] = {}
        # Stores the call-phase TestReport for deferred tests so we can move
        # it between stats['passed'] / stats['failed'] in terminal summary.
        self._deferred_reports: Dict[str, pytest.TestReport] = {}
        self._corrected: List[_CorrectedResult] = []
        self._failed_nodeids: Set[str] = set()

    # ------------------------------------------------------------------
    # Hook: capture call-phase reports for deferred tests
    # ------------------------------------------------------------------

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.when != "call":
            return
        reporter = self._reporters.get(report.nodeid)
        if reporter is not None and reporter.value is not None and report.outcome == "passed":
            self._deferred_reports[report.nodeid] = report

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------

    def _apply_correction(self) -> None:
        pvalue_items = [
            (nodeid, r.value)
            for nodeid, r in self._reporters.items()
            if r.value is not None
        ]
        if not pvalue_items:
            return

        n = len(pvalue_items)
        sorted_items = sorted(pvalue_items, key=lambda x: x[1])

        stop_rejecting = False
        for k, (nodeid, p) in enumerate(sorted_items, 1):
            threshold = self.alpha / (n - k + 1)
            if not stop_rejecting and p <= threshold:
                passed = True
            else:
                stop_rejecting = True
                passed = False
                self._failed_nodeids.add(nodeid)

            self._corrected.append(_CorrectedResult(
                nodeid=nodeid,
                p_value=p,
                threshold=threshold,
                passed=passed,
            ))

    # ------------------------------------------------------------------
    # Hook: run correction and update session exit status
    # ------------------------------------------------------------------

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        self._apply_correction()
        if self._failed_nodeids:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    # ------------------------------------------------------------------
    # Hook: update terminal stats and print summary
    # ------------------------------------------------------------------

    def pytest_terminal_summary(self, terminalreporter, exitstatus: int, config: pytest.Config) -> None:
        if not self._corrected:
            return

        stats = terminalreporter.stats

        # Re-categorise deferred tests based on the corrected outcome so that
        # the "N passed, M failed" line printed by pytest reflects reality.
        threshold_map = {r.nodeid: r.threshold for r in self._corrected}
        for nodeid in self._failed_nodeids:
            report = self._deferred_reports.get(nodeid)
            if report is None:
                continue
            passed_list = stats.get("passed", [])
            if report in passed_list:
                passed_list.remove(report)
            p_value = self._reporters[nodeid].value
            threshold = threshold_map[nodeid]
            report.outcome = "failed"
            report.longrepr = (
                f"Holm-Bonferroni: p={p_value:.6f} > threshold={threshold:.6f}"
            )
            stats.setdefault("failed", []).append(report)

        # Print the correction table.
        n_total = len(self._corrected)
        n_failed = len(self._failed_nodeids)
        n_passed = n_total - n_failed

        terminalreporter.write_sep(
            "=",
            f"Holm-Bonferroni correction  α={self.alpha}  n={n_total}",
        )
        for result in self._corrected:  # already sorted by p-value
            status = "PASSED" if result.passed else "FAILED"
            terminalreporter.write_line(
                f"  {status}  p={result.p_value:.6f}  "
                f"threshold={result.threshold:.6f}  {result.nodeid}"
            )
        terminalreporter.write_line(
            f"\n  {n_passed} passed, {n_failed} failed "
            f"after Holm-Bonferroni correction"
        )


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--holm-alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Family-wise error rate for Holm-Bonferroni correction (default: 0.05)",
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
    try:
        alpha = config.getoption("--holm-alpha")
        power = config.getoption("--power")
    except (ValueError, AttributeError):
        alpha = 0.05
        power = 0.8
    plugin = HolmBonferroniPlugin(alpha=alpha, power=power)
    config.pluginmanager.register(plugin, "holm_bonferroni")
    config._holm_plugin = plugin  # type: ignore[attr-defined]


@pytest.fixture
def pvalue(request: pytest.FixtureRequest) -> PValueReporter:
    """Register a p-value for Holm-Bonferroni correction.

    Call the returned object once inside your test with the computed p-value.
    The plugin will determine pass/fail after all tests finish.

    Example::

        def test_chi_squared(pvalue):
            stat, p = scipy.stats.chisquare(observed, expected)
            pvalue(p)
    """
    plugin: HolmBonferroniPlugin = request.config._holm_plugin  # type: ignore[attr-defined]
    reporter = PValueReporter()
    plugin._reporters[request.node.nodeid] = reporter
    return reporter


@pytest.fixture
def ztest_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a z-test at the session's alpha and power.

    Usage::

        def test_mean(ztest_sample_size, pvalue):
            n = ztest_sample_size(effect_size=0.5)          # two-sided
            n = ztest_sample_size(effect_size=0.5, two_sided=False)
            data = generate(n)
            _, p = scipy.stats.ttest_1samp(data, 0)
            pvalue(p)

    ``effect_size`` is Cohen's d (mean difference / pooled SD).
    Returns per-group n for a two-sample test.
    """
    plugin: HolmBonferroniPlugin = request.config._holm_plugin  # type: ignore[attr-defined]

    def compute(effect_size: float, two_sided: bool = True) -> int:
        return _ztest_n(plugin.alpha, plugin.power, effect_size, two_sided)

    return compute


@pytest.fixture
def chisquare_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a chi-square goodness-of-fit test.

    Usage::

        def test_distribution(chisquare_sample_size, pvalue):
            n = chisquare_sample_size(effect_size=0.3, df=4)
            counts = generate(n)
            _, p = scipy.stats.chisquare(counts, expected)
            pvalue(p)

    ``effect_size`` is Cohen's w; ``df`` is the degrees of freedom
    (number of categories − 1 for goodness-of-fit).
    """
    plugin: HolmBonferroniPlugin = request.config._holm_plugin  # type: ignore[attr-defined]

    def compute(effect_size: float, df: int) -> int:
        return _chisquare_n(plugin.alpha, plugin.power, effect_size, df)

    return compute


@pytest.fixture
def ks_sample_size(request: pytest.FixtureRequest) -> Callable[..., int]:
    """Return required n for a KS test, sized via the DKW inequality.

    Usage::

        def test_uniform(ks_sample_size, pvalue):
            n = ks_sample_size(effect_size=0.1)              # one-sample
            n = ks_sample_size(effect_size=0.1, two_sample=True)  # per-group
            data = generate(n)
            p = scipy.stats.kstest(data, 'uniform').pvalue
            pvalue(p)

    ``effect_size`` is the maximum absolute CDF difference ||F − G||_∞.
    When ``two_sample=True`` the returned value is the required per-group n
    (assuming equal group sizes).
    """
    plugin: HolmBonferroniPlugin = request.config._holm_plugin  # type: ignore[attr-defined]

    def compute(effect_size: float, two_sample: bool = False) -> int:
        return _ks_n(plugin.alpha, plugin.power, effect_size, two_sample)

    return compute
