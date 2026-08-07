"""Tests for the familywise-correction pytest plugin."""
import json
import math
import numpy as np
import pytest
from numpy.random import default_rng
from pytest_familywise import (
    CALIBRATION_VERSION,
    _ztest_n,
    _chisquare_n,
    _ks_n,
    _holm_adjusted,
    _westfall_young_adjusted,
    _calibrated_alphas,
    _conservative_quantile,
    _Calibration,
    _CalibrationError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(pytester, src: str, alpha: float = 0.05):
    pytester.makepyfile(src)
    return pytester.runpytest(f"--alpha={alpha}", "-v")


# ---------------------------------------------------------------------------
# Single-test cases
# ---------------------------------------------------------------------------

def test_passes_when_data_consistent_with_h0(pytester):
    """A large p-value means data is consistent with H0 — test passes."""
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.9)
    """)
    result.assert_outcomes(passed=1)


def test_fails_when_h0_rejected(pytester):
    """A very small p-value means H0 is rejected — test fails."""
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.001)
    """)
    result.assert_outcomes(failed=1)


def test_rejected_at_boundary(pytester):
    # n=1: threshold = alpha = 0.05; p = 0.05 <= threshold -> rejected -> fail
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.05)
    """, alpha=0.05)
    result.assert_outcomes(failed=1)


def test_not_rejected_just_above_alpha(pytester):
    # n=1: threshold = alpha = 0.05; p = 0.051 > threshold -> not rejected -> pass
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.051)
    """, alpha=0.05)
    result.assert_outcomes(passed=1)


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down logic
# ---------------------------------------------------------------------------

def test_all_pass_when_all_pvalues_large(pytester):
    """When all p-values are large (data consistent with H0), all tests pass."""
    result = run(pytester, """
        def test_a(assertNotReject): assertNotReject(0.5)
        def test_b(assertNotReject): assertNotReject(0.7)
        def test_c(assertNotReject): assertNotReject(0.3)
    """)
    result.assert_outcomes(passed=3)


def test_all_fail_when_all_pvalues_tiny(pytester):
    """When all p-values are tiny, every null hypothesis is rejected."""
    # n=4, sorted: 0.001, 0.002, 0.003, 0.004
    # k=1: threshold=0.05/4=0.0125; 0.001<=0.0125 -> REJECT
    # k=2: threshold=0.05/3=0.0167; 0.002<=0.0167 -> REJECT
    # k=3: threshold=0.05/2=0.025;  0.003<=0.025  -> REJECT
    # k=4: threshold=0.05/1=0.05;   0.004<=0.05   -> REJECT
    result = run(pytester, """
        def test_a(assertNotReject): assertNotReject(0.001)
        def test_b(assertNotReject): assertNotReject(0.002)
        def test_c(assertNotReject): assertNotReject(0.003)
        def test_d(assertNotReject): assertNotReject(0.004)
    """)
    result.assert_outcomes(failed=4)


def test_correction_protects_marginal_pvalues(pytester):
    """P-values that would be rejected alone are protected by the correction.

    With n=2, the first threshold is alpha/2 = 0.025.  A p-value of 0.04
    would be rejected at alpha=0.05 in a single test, but Holm-Bonferroni
    tightens the threshold so 0.04 > 0.025 -> not rejected -> pass.
    """
    result = run(pytester, """
        def test_a(assertNotReject): assertNotReject(0.04)
        def test_b(assertNotReject): assertNotReject(0.08)
    """)
    result.assert_outcomes(passed=2)


def test_step_down_rejects_only_smallest(pytester):
    """Only the smallest p-value is rejected; once the step-down stops, the
    rest pass even though some are below the uncorrected alpha.

    sorted: 0.01, 0.03, 0.07
    k=1: threshold=0.05/3=0.0167; 0.01<=0.0167 -> REJECT (fail)
    k=2: threshold=0.05/2=0.025;  0.03>0.025   -> stop   (pass)
    k=3: stop                                             (pass)
    """
    result = run(pytester, """
        def test_a(assertNotReject): assertNotReject(0.01)
        def test_b(assertNotReject): assertNotReject(0.03)
        def test_c(assertNotReject): assertNotReject(0.07)
    """)
    result.assert_outcomes(passed=2, failed=1)


def test_step_down_rejects_all_when_all_below_thresholds(pytester):
    """When every p-value falls below its Holm-Bonferroni threshold, all are
    rejected.

    sorted: 0.007, 0.01, 0.04
    k=1: threshold=0.05/3=0.0167; 0.007<=0.0167 -> REJECT
    k=2: threshold=0.05/2=0.025;  0.01 <=0.025  -> REJECT
    k=3: threshold=0.05/1=0.05;   0.04 <=0.05   -> REJECT
    """
    result = run(pytester, """
        def test_a(assertNotReject): assertNotReject(0.01)
        def test_b(assertNotReject): assertNotReject(0.04)
        def test_c(assertNotReject): assertNotReject(0.007)
    """)
    result.assert_outcomes(failed=3)


# ---------------------------------------------------------------------------
# Non-assertNotReject tests are unaffected
# ---------------------------------------------------------------------------

def test_ordinary_passing_test_unaffected(pytester):
    """A plain assertion test coexists with an assertNotReject test."""
    result = run(pytester, """
        def test_ordinary():
            assert 1 + 1 == 2

        def test_stat(assertNotReject):
            assertNotReject(0.9)
    """)
    result.assert_outcomes(passed=2)


def test_ordinary_failing_test_unaffected(pytester):
    """A plain assertion failure is independent of assertNotReject results."""
    result = run(pytester, """
        def test_ordinary():
            assert False

        def test_stat(assertNotReject):
            assertNotReject(0.9)
    """)
    result.assert_outcomes(passed=1, failed=1)


# ---------------------------------------------------------------------------
# Exceptions in assertNotReject tests fail normally
# ---------------------------------------------------------------------------

def test_exception_before_assertNotReject_fails_normally(pytester):
    """An exception before assertNotReject is called fails the test normally,
    without entering the Holm-Bonferroni set."""
    result = run(pytester, """
        def test_raises(assertNotReject):
            raise RuntimeError("boom")
            assertNotReject(0.9)
    """)
    result.assert_outcomes(failed=1)


def test_exception_after_assertNotReject_still_fails(pytester):
    """An exception after assertNotReject is called still fails the test
    normally — the plugin does not override it to passed."""
    result = run(pytester, """
        def test_raises(assertNotReject):
            assertNotReject(0.9)
            raise RuntimeError("boom after assertNotReject")
    """)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# Custom alpha
# ---------------------------------------------------------------------------

def test_stricter_alpha_protects_more(pytester):
    """A stricter alpha=0.01 does not reject p=0.02 (which alpha=0.05 would)."""
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.02)
    """, alpha=0.01)
    result.assert_outcomes(passed=1)


def test_stricter_alpha_still_rejects_very_small(pytester):
    """Even at alpha=0.01, a very small p-value is still rejected."""
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.005)
    """, alpha=0.01)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# p-value validation
# ---------------------------------------------------------------------------

def test_invalid_pvalue_raises(pytester):
    result = run(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(1.5)
    """)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

def test_parametrized_all_consistent_with_h0(pytester):
    """Each parametrized variant gets its own p-value; all large -> all pass."""
    result = run(pytester, """
        import pytest

        @pytest.mark.parametrize("p", [0.5, 0.6, 0.7])
        def test_param(assertNotReject, p):
            assertNotReject(p)
    """)
    result.assert_outcomes(passed=3)


def test_parametrized_mixed(pytester):
    """Parametrized tests with mixed results: small p rejected, large p pass.

    sorted: 0.001, 0.5, 0.9
    k=1: threshold=0.05/3=0.0167; 0.001<=0.0167 -> REJECT (fail)
    k=2: threshold=0.05/2=0.025;  0.5>0.025     -> stop   (pass)
    k=3: stop                                             (pass)
    """
    result = run(pytester, """
        import pytest

        @pytest.mark.parametrize("p", [0.001, 0.5, 0.9])
        def test_param(assertNotReject, p):
            assertNotReject(p)
    """)
    result.assert_outcomes(passed=2, failed=1)


# ---------------------------------------------------------------------------
# Adjusted-p-value unit tests (pure functions, no pytester, no randomness)
# ---------------------------------------------------------------------------

class TestHolmAdjusted:
    def test_matches_threshold_form(self):
        # sorted p: 0.01, 0.03, 0.07 with m=3
        # (3-k+1)*p_k = 0.03, 0.06, 0.07 -> already monotone
        adj = _holm_adjusted([0.01, 0.03, 0.07], [None] * 3)
        assert adj == pytest.approx([0.03, 0.06, 0.07])
        # Rejecting where adjusted <= 0.05 reproduces "only the smallest fails".
        assert [a <= 0.05 for a in adj] == [True, False, False]

    def test_monotonicity_enforced(self):
        # (2-k+1)*p_k = 0.08, 0.06 -> second must be pulled up to 0.08
        adj = _holm_adjusted([0.04, 0.06], [None] * 2)
        assert adj == pytest.approx([0.08, 0.08])

    def test_clipped_at_one(self):
        assert _holm_adjusted([0.9, 0.95], [None] * 2) == pytest.approx([1.0, 1.0])


class TestWestfallYoungAdjusted:
    def test_hand_computed_counts(self):
        # m=2, B=4.  Columns are the null p-values per test.
        #   test0 nulls: 0.10 0.60 0.02 0.80
        #   test1 nulls: 0.50 0.05 0.90 0.70
        # per-resample minima over {0,1}: 0.10 0.05 0.02 0.70
        # observed p = [0.04, 0.30]
        #   k=0: #{min_over_tail <= 0.04} = {0.02} -> 1/4
        #   k=1: tail is test1 alone: 0.50 0.05 0.90 0.70 <= 0.30 -> {0.05} -> 1/4
        #        monotonicity: max(0.25, 0.25) = 0.25
        nulls = [[0.10, 0.60, 0.02, 0.80], [0.50, 0.05, 0.90, 0.70]]
        adj = _westfall_young_adjusted([0.04, 0.30], nulls)
        assert adj == pytest.approx([0.25, 0.25])

    def test_perfectly_correlated_nulls_barely_correct(self):
        """Identical null columns => effectively one test => adjusted ≈ raw p.

        This is the property Holm cannot express, and the whole point of the
        procedure: two tests that are the same test should not cost each other
        any power.
        """
        column = [i / 1000 for i in range(1000)]
        adj = _westfall_young_adjusted([0.30, 0.30], [column, column])
        # min over identical columns is the column itself, so the adjusted value
        # is just the empirical CDF at 0.30.
        assert adj[0] == pytest.approx(0.30, abs=0.01)
        # Holm, by contrast, doubles it.
        assert _holm_adjusted([0.30, 0.30], [None] * 2)[0] == pytest.approx(0.60)

    def test_independent_nulls_approach_one_minus_survival(self):
        """Independent uniform nulls: adjusted p for m=2 -> 1 - (1-p)^2.

        This is the other end of the range from the correlated case: with no
        dependence to exploit, WY reproduces the exact Sidak-style correction
        (and Holm's 2*p is its conservative bound).
        """
        rng = default_rng(0)
        b = 20000
        col_a = rng.random(b).tolist()
        col_b = rng.random(b).tolist()
        adj = _westfall_young_adjusted([0.20, 0.90], [col_a, col_b])
        assert adj[0] == pytest.approx(1 - (1 - 0.20) ** 2, abs=0.01)

    def test_missing_nulls_rejected(self):
        with pytest.raises(ValueError, match="null samples for every test"):
            _westfall_young_adjusted([0.1, 0.2], [[0.5], None])

    def test_ragged_nulls_rejected(self):
        with pytest.raises(ValueError, match="same number of null samples"):
            _westfall_young_adjusted([0.1, 0.2], [[0.5], [0.5, 0.6]])

    def test_granularity_floor(self):
        """The smallest attainable adjusted p is 1/B, so B must exceed 1/alpha."""
        # B=10 -> nothing can ever be adjusted below 0.1, so alpha=0.05 can
        # never reject no matter how extreme the observed p-value.
        nulls = [[0.5] * 10]
        adj = _westfall_young_adjusted([1e-9], nulls)
        assert adj[0] == 0.0  # no null draw is below 1e-9 here
        nulls = [[1e-12] + [0.5] * 9]
        assert _westfall_young_adjusted([1e-9], nulls)[0] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Calibration: critical value from a recorded null matrix
# ---------------------------------------------------------------------------

def make_calibration(columns, resamples=None):
    """Build a _Calibration from {nodeid: sequence of null p-values}."""
    b = resamples if resamples is not None else len(next(iter(columns.values())))
    return _Calibration(
        resamples=b,
        columns={k: np.asarray(v, dtype=np.float32) for k, v in columns.items()},
    )


class TestCalibratedAlpha:
    def test_perfect_correlation_approaches_alpha(self):
        """Identical columns => one effective test => c_1 approaches alpha.

        This is the payoff: Holm would use alpha/m = 0.01 at rung 1 here.
        """
        grid = [i / 20000 for i in range(20000)]
        cal = make_calibration({f"t{j}": grid for j in range(5)})
        c = _calibrated_alphas(cal, [f"t{j}" for j in range(5)], 0.05)
        assert 0.045 < c[0] <= 0.05
        assert c[0] > 0.05 / 5 * 4  # far looser than Holm's alpha/m

    def test_independence_lands_near_bonferroni(self):
        """Independent columns => c_1 barely above alpha/m.

        The feature correctly finding nothing is as important as it finding
        something: this is the regime where calibration should not help.
        """
        rng = default_rng(0)
        b = 40000
        cal = make_calibration({f"t{j}": rng.random(b).tolist() for j in range(2)})
        c = _calibrated_alphas(cal, ["t0", "t1"], 0.05)
        # exact value is 1 - sqrt(0.95) = 0.02532, vs Bonferroni's 0.025
        assert c[0] == pytest.approx(0.0253, abs=0.0015)

    def test_ladder_brackets_holm_rung_by_rung(self):
        """c_k >= alpha / (m - k + 1) for every k, so n only ever shrinks.

        This is the property that lets the recording run size with Holm's ladder
        instead of paying uniform alpha/m: each rung is loosened against the
        matching rung, never against a collapsed single value.
        """
        rng = default_rng(1)
        alpha = 0.05
        for m in (2, 5, 20):
            cal = make_calibration({f"t{j}": rng.random(4000).tolist() for j in range(m)})
            c = _calibrated_alphas(cal, [f"t{j}" for j in range(m)], alpha)
            assert len(c) == m
            for k in range(1, m + 1):
                assert alpha / (m - k + 1) <= c[k - 1] <= alpha

    def test_ladder_is_monotone(self):
        """A minimum over a superset is never larger, so c_k increases in k.

        Later tests therefore get looser thresholds and smaller samples, which is
        what makes ordering expensive tests last pay off.
        """
        rng = default_rng(7)
        cols = {f"t{j}": (rng.random(8000) ** (1 + j)).tolist() for j in range(6)}
        c = _calibrated_alphas(make_calibration(cols), sorted(cols), 0.05)
        assert c == sorted(c)
        assert c[-1] == pytest.approx(0.05)  # top rung: one column, quantile = alpha

    def test_never_looser_than_alpha(self):
        """c_k <= alpha always — no correction can beat doing no correction."""
        cal = make_calibration({f"t{j}": [0.99] * 4000 for j in range(3)})
        assert _calibrated_alphas(cal, ["t0", "t1", "t2"], 0.05) == [0.05] * 3

    def test_quantile_estimate_is_conservative(self):
        """The estimate sits below the median-unbiased quantile."""
        grid = np.array([[i / 20000 for i in range(20000)]], dtype=np.float32)
        conservative = _conservative_quantile(grid, 0.05)[0]
        assert conservative < 0.05
        # ...but converges toward it as B grows.
        small = _conservative_quantile(grid[:, :2000] * 10, 0.05)[0]
        assert small / 0.05 < conservative / 0.05

    def test_granularity_floor_applies_for_large_m(self):
        """With m > B*alpha, alpha/m falls under 1/B and the floor takes over."""
        rng = default_rng(3)
        b, m = 4000, 500
        cal = make_calibration({f"t{j}": rng.random(b).tolist() for j in range(m)})
        assert 0.05 / m < 1.0 / b  # premise: Bonferroni is below the floor
        c = _calibrated_alphas(cal, [f"t{j}" for j in range(m)], 0.05)
        assert c[0] == pytest.approx(1.0 / b)

    def test_subsetting_matches_a_standalone_matrix(self):
        """The ladder over a subset equals that of a calibration built from it.

        This is what lets `-k` / `--lf` / single-file runs still use the cache.
        """
        rng = default_rng(2)
        cols = {f"t{j}": rng.random(4000).tolist() for j in range(5)}
        full = _calibrated_alphas(make_calibration(cols), ["t1", "t3"], 0.05)
        subset = _calibrated_alphas(
            make_calibration({k: cols[k] for k in ("t1", "t3")}), ["t1", "t3"], 0.05
        )
        assert full == subset

    def test_missing_nodeid_raises(self):
        cal = make_calibration({"t0": [0.5] * 4000})
        with pytest.raises(_CalibrationError, match="absent from the calibration"):
            _calibrated_alphas(cal, ["t0", "t_new"], 0.05)

    def test_too_few_order_statistics_raises(self):
        cal = make_calibration({"t0": [0.5] * 100})
        with pytest.raises(_CalibrationError, match="too small at alpha"):
            _calibrated_alphas(cal, ["t0"], 0.05)


# ---------------------------------------------------------------------------
# Calibration: file round trip and fallback behaviour
# ---------------------------------------------------------------------------

CORRELATED_SUITE = """
    def sample(rng):
        return float(rng.random())

    def test_a(assertNotReject, ztest_sample_size):
        n = ztest_sample_size(effect_size=0.3)
        with open("n_a.txt", "w") as f:
            f.write(str(n))
        assertNotReject(0.5, null_sample=sample)

    def test_b(assertNotReject, ztest_sample_size):
        n = ztest_sample_size(effect_size=0.3)
        with open("n_b.txt", "w") as f:
            f.write(str(n))
        assertNotReject(0.5, null_sample=sample)
"""


def wy_args(*extra, resamples=4000):
    return ("--correction=westfall-young", f"--resamples={resamples}", *extra)


def test_calibration_round_trip_shrinks_sample_size(pytester):
    """Calibrate, then re-run: the recorded correlation loosens the threshold.

    Both tests draw their nulls from the same rng stream, so they are perfectly
    correlated and c1 should approach alpha rather than alpha/2.
    """
    pytester.makepyfile(CORRELATED_SUITE)

    first = pytester.runpytest(*wy_args())
    first.assert_outcomes(passed=2)
    n_uncalibrated = int((pytester.path / "n_a.txt").read_text())

    path = pytester.path / ".familywise-calibration.npz"
    assert path.exists()
    with np.load(path) as data:
        assert int(data["version"]) == CALIBRATION_VERSION
        assert int(data["resamples"]) == 4000
        assert "pilot_n" not in data  # dropped in format version 2
        assert data["nulls"].shape == (2, 4000)
        assert data["nulls"].dtype == np.float32
        assert sorted(str(n) for n in data["nodeids"]) == [
            "test_calibration_round_trip_shrinks_sample_size.py::test_a",
            "test_calibration_round_trip_shrinks_sample_size.py::test_b",
        ]

    second = pytester.runpytest(*wy_args())
    second.assert_outcomes(passed=2)
    n_calibrated = int((pytester.path / "n_a.txt").read_text())

    second.stdout.fnmatch_lines(["calibration: sizing alpha=*for 2 tests*"])
    assert n_calibrated < n_uncalibrated


def test_calibration_reports_and_falls_back_on_unknown_test(pytester):
    """A test absent from the cache => warn, and size exactly as Holm would."""
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args())
    n_two_tests = int((pytester.path / "n_a.txt").read_text())

    # Add a third test that the calibration has never seen.
    pytester.makepyfile(CORRELATED_SUITE + """
    def test_c(assertNotReject):
        assertNotReject(0.5, null_sample=sample)
    """)
    result = pytester.runpytest(*wy_args())
    result.stdout.fnmatch_lines(["calibration: *absent from the calibration*Holm*"])
    # Holm's ladder for the first requester of m=3 is alpha/3.
    assert int((pytester.path / "n_a.txt").read_text()) != n_two_tests


def test_calibration_survives_subset_run(pytester):
    """Selecting a subset still uses the cache — no fallback warning."""
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args())

    result = pytester.runpytest(*wy_args(), "-k", "test_a")
    result.assert_outcomes(passed=1)
    result.stdout.fnmatch_lines(["calibration: sizing alpha=*for 1 tests*"])
    result.stdout.no_fnmatch_line("*absent*")


def test_calibration_reused_at_a_different_alpha(pytester):
    """The stored matrix is alpha-free, so changing alpha needs no re-run.

    Raising alpha, not lowering it: a smaller alpha needs a larger B to keep
    B*alpha order statistics, which is a separate (and correct) fallback.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args())
    stamp = (pytester.path / ".familywise-calibration.npz").stat().st_mtime_ns

    result = pytester.runpytest(*wy_args("--alpha=0.10"))
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["calibration: sizing alpha=*"])
    result.stdout.no_fnmatch_line("*absent*")
    # Not rewritten.
    assert (pytester.path / ".familywise-calibration.npz").stat().st_mtime_ns == stamp


def test_too_few_resamples_falls_back(pytester):
    """B*alpha < 100 is too noisy to size against."""
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args(resamples=200))
    result = pytester.runpytest(*wy_args(resamples=200))
    result.stdout.fnmatch_lines(["calibration: --resamples=200 too small*2000*Holm*"])


def test_corrupt_calibration_falls_back_without_crashing(pytester):
    """A damaged cache may only cost power, never break the suite."""
    pytester.makepyfile(CORRELATED_SUITE)
    (pytester.path / ".familywise-calibration.npz").write_bytes(b"not an npz file")

    result = pytester.runpytest(*wy_args())
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["calibration: unreadable*Holm*"])


def test_deleting_the_file_recalibrates(pytester):
    """Deleting the calibration is how you re-record it.

    There is no --recalibrate flag: absence of the file *is* the record signal,
    so `rm` is the whole interface.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    path = pytester.path / ".familywise-calibration.npz"

    pytester.runpytest(*wy_args())                       # no cache -> Holm sizing
    n_holm = int((pytester.path / "n_a.txt").read_text())

    pytester.runpytest(*wy_args())                       # cache hit -> smaller
    assert int((pytester.path / "n_a.txt").read_text()) < n_holm

    path.unlink()
    result = pytester.runpytest(*wy_args())
    result.stdout.fnmatch_lines([
        "calibration: recording a new calibration*",
        "calibration: wrote *",
    ])
    assert path.exists()
    assert int((pytester.path / "n_a.txt").read_text()) == n_holm


def test_recording_run_sizes_with_holms_ladder(pytester):
    """A recording run sizes with Holm's ladder, same as an uncalibrated run.

    Each rung of the calibrated ladder brackets above Holm's matching rung, so
    the pilot's n is an upper bound test by test — no need to charge every test
    the most stringent rung just to keep the recorded matrix comparable.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    result = pytester.runpytest(*wy_args())
    result.assert_outcomes(passed=2)
    n_a = int((pytester.path / "n_a.txt").read_text())
    n_b = int((pytester.path / "n_b.txt").read_text())
    # m=2: rung 1 is alpha/2, rung 2 is alpha.
    assert n_a == _ztest_n(0.05 / 2, 0.8, 0.3)
    assert n_b == _ztest_n(0.05, 0.8, 0.3)
    assert n_b < n_a


def test_calibrated_ladder_shrinks_every_rung(pytester):
    """Loading a calibration may never enlarge any test's n.

    The pilot sizes at Holm's rung k, the calibrated run at c_k >= that rung, so
    every test shrinks or holds — the guarantee the module docstring makes.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args())  # pilot: records, sizes with Holm
    pilot = [int((pytester.path / f"n_{t}.txt").read_text()) for t in ("a", "b")]

    pytester.runpytest(*wy_args())  # cache hit: sizes with the c_k ladder
    loaded = [int((pytester.path / f"n_{t}.txt").read_text()) for t in ("a", "b")]

    assert all(new <= old for new, old in zip(loaded, pilot))
    assert loaded[0] < pilot[0]  # rung 1 is where a correlated suite pays off


def test_unresolved_dependence_says_to_raise_resamples(pytester):
    """Falling back to the Bonferroni clamp must not look like a success.

    Two genuinely independent tests, so the estimate lands on the clamp; the
    report has to name the knob instead of printing alpha/m as if it measured
    something.
    """
    pytester.makepyfile("""
        def test_a(assertNotReject, ztest_sample_size):
            ztest_sample_size(effect_size=0.3)
            assertNotReject(0.5, null_sample=lambda rng: float(rng.random()))

        def test_b(assertNotReject, ztest_sample_size):
            ztest_sample_size(effect_size=0.3)
            # Consumes the stream differently, so the columns are independent.
            def sample(rng):
                rng.random(7)
                return float(rng.random())
            assertNotReject(0.5, null_sample=sample)
    """)
    pytester.runpytest(*wy_args())
    result = pytester.runpytest(*wy_args())
    result.stdout.fnmatch_lines(["calibration: no dependence resolved at B=4000*"])


def test_holm_run_ignores_a_leftover_calibration(pytester):
    """A calibration is only valid for the procedure that produced it.

    Loading c1 under --correction=holm would size against c1 >= alpha/m while the
    run actually applies Holm's stricter ladder -- under-sizing every test, the
    one direction sizing must never go.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args())          # records a calibration
    assert (pytester.path / ".familywise-calibration.npz").exists()

    result = pytester.runpytest("--alpha=0.05")   # holm, cache present
    result.assert_outcomes(passed=2)
    result.stdout.no_fnmatch_line("*calibration: sizing alpha*")
    # Holm's own ladder, rung 1 of 2.
    assert int((pytester.path / "n_a.txt").read_text()) == _ztest_n(
        0.05 / 2, 0.8, 0.3
    )


def test_outsider_gets_raw_alpha_even_with_a_calibration_loaded(pytester):
    """c1 applies to the family; a test outside it is never corrected."""
    pytester.makepyfile(CORRELATED_SUITE + """
    def test_outsider(ztest_sample_size):
        with open("n_out.txt", "w") as f:
            f.write(str(ztest_sample_size(effect_size=0.3)))
    """)
    pytester.runpytest(*wy_args())
    result = pytester.runpytest(*wy_args())
    result.stdout.fnmatch_lines(["calibration: sizing alpha=*"])
    assert int((pytester.path / "n_out.txt").read_text()) == _ztest_n(
        0.05, 0.8, 0.3
    )


def test_calibration_survives_a_changed_power(pytester):
    """Sizing knobs may move between runs without invalidating the cache.

    The recorded columns are null draws; --power enters only the sample-size
    formula, not the null distribution, so the cache still applies.
    """
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args(), "--power=0.8")

    result = pytester.runpytest(*wy_args(), "--power=0.99")
    result.stdout.fnmatch_lines(["calibration: sizing alpha=*"])
    result.assert_outcomes(passed=2)


def test_holm_run_writes_no_calibration(pytester):
    """Only westfall-young collects nulls, so only it can calibrate."""
    pytester.makepyfile(CORRELATED_SUITE.replace(", null_sample=sample", ""))
    pytester.runpytest("--alpha=0.05")
    assert not (pytester.path / ".familywise-calibration.npz").exists()


def test_calibration_can_be_disabled(pytester):
    pytester.makepyfile(CORRELATED_SUITE)
    pytester.runpytest(*wy_args("--calibration="))
    assert not (pytester.path / ".familywise-calibration.npz").exists()


def test_calibration_does_not_change_the_correction(pytester):
    """Sizing may only affect power — never which hypotheses are rejected.

    Same suite, with and without a calibration file: identical verdicts and
    identical adjusted p-values.
    """
    src = """
        def sample(rng):
            return float(rng.random())

        def test_a(assertNotReject): assertNotReject(0.03, null_sample=sample)
        def test_b(assertNotReject): assertNotReject(0.60, null_sample=sample)
    """
    pytester.makepyfile(src)
    first = pytester.runpytest(*wy_args())
    assert (pytester.path / ".familywise-calibration.npz").exists()
    second = pytester.runpytest(*wy_args())

    def verdicts(result):
        return [
            line.strip()
            for line in result.stdout.lines
            if "adjusted p=" in line
        ]

    assert verdicts(first) == verdicts(second)
    assert verdicts(first)  # non-empty, or the assertion above is vacuous


# ---------------------------------------------------------------------------
# Westfall-Young end-to-end
# ---------------------------------------------------------------------------

def run_wy(pytester, src: str, alpha: float = 0.05, resamples: int = 200):
    pytester.makepyfile(src)
    return pytester.runpytest(
        f"--alpha={alpha}",
        "--correction=westfall-young",
        f"--resamples={resamples}",
        "-v",
    )


def test_wy_passes_when_observed_p_is_typical(pytester):
    """An observed p in the bulk of its own null distribution is not rejected."""
    result = run_wy(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.5, null_sample=lambda rng: rng.random())
    """)
    result.assert_outcomes(passed=1)


def test_wy_fails_when_observed_p_is_extreme(pytester):
    """An observed p far below every null draw is rejected."""
    result = run_wy(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(1e-6, null_sample=lambda rng: rng.random())
    """)
    result.assert_outcomes(failed=1)


def test_wy_requires_null_sample(pytester):
    """Omitting null_sample under WY fails the test with a clear message."""
    result = run_wy(pytester, """
        def test_foo(assertNotReject):
            assertNotReject(0.5)
    """)
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(["*requires*null_sample*"])


def test_wy_null_columns_are_aligned(pytester):
    """Two tests applying the same transform must see the same null draws.

    This is the mechanism the whole dependence story rests on: draw b is seeded
    identically in every test, so aligned transforms give aligned (correlated)
    null columns.  If the shared-seed wiring regresses, this fails.
    """
    pytester.makepyfile("""
        import json, pathlib

        def sample(rng):
            return float(rng.random())

        def test_a(assertNotReject, request):
            assertNotReject(0.5, null_sample=sample)
            nulls = request.config._familywise_plugin._reporters[request.node.nodeid].nulls
            pathlib.Path("a.json").write_text(json.dumps(nulls.tolist()))

        def test_b(assertNotReject, request):
            assertNotReject(0.5, null_sample=sample)
            nulls = request.config._familywise_plugin._reporters[request.node.nodeid].nulls
            pathlib.Path("b.json").write_text(json.dumps(nulls.tolist()))

        def test_aligned():
            a = json.loads(pathlib.Path("a.json").read_text())
            b = json.loads(pathlib.Path("b.json").read_text())
            assert len(a) == 50
            assert a == b
    """)
    result = pytester.runpytest(
        "--correction=westfall-young", "--resamples=50", "-v"
    )
    result.assert_outcomes(passed=3)


def test_readme_wy_example_runs(pytester):
    """The Westfall-Young example in README.md, verbatim.

    Guards against the docs drifting from the API.
    """
    pytester.makepyfile("""
        import numpy as np
        import pytest
        import scipy.stats


        @pytest.fixture(scope="module")
        def samples():
            return np.random.default_rng(0).standard_normal(2000)


        def under_h0(rng, data):
            centered = data - data.mean()
            return rng.choice(centered, size=len(data), replace=True)


        def test_mean_zero(assertNotReject, samples):
            p = scipy.stats.ttest_1samp(samples, 0.0).pvalue
            assertNotReject(p, null_sample=lambda rng:
                scipy.stats.ttest_1samp(under_h0(rng, samples), 0.0).pvalue)


        def test_median_zero(assertNotReject, samples):
            def pvalue(data):
                return scipy.stats.binomtest(int((data > 0).sum()), len(data), 0.5).pvalue

            assertNotReject(pvalue(samples), null_sample=lambda rng:
                pvalue(under_h0(rng, samples)))
    """)
    result = pytester.runpytest("--correction=westfall-young", "--resamples=100")
    result.assert_outcomes(passed=2)
    result.stdout.fnmatch_lines(["*Westfall-Young correction*"])


def test_readme_permutation_example_runs(pytester):
    """The permutation variant from README.md."""
    pytester.makepyfile("""
        import numpy as np
        import pytest
        import scipy.stats

        @pytest.fixture
        def group_a(): return np.random.default_rng(1).standard_normal(200)

        @pytest.fixture
        def group_b(): return np.random.default_rng(2).standard_normal(200)

        def test_groups_are_identical(assertNotReject, group_a, group_b):
            pooled = np.concatenate([group_a, group_b])
            n = len(group_a)

            def pvalue(x):
                return scipy.stats.ks_2samp(x[:n], x[n:]).pvalue

            assertNotReject(pvalue(pooled), null_sample=lambda rng:
                pvalue(rng.permutation(pooled)))
    """)
    result = pytester.runpytest("--correction=westfall-young", "--resamples=100")
    result.assert_outcomes(passed=1)


def test_wy_beats_holm_on_correlated_tests(pytester):
    """The payoff: identical correlated tests that Holm rejects, WY does not.

    Two perfectly correlated tests each with p=0.03.  Holm doubles it to 0.06
    (> 0.05, so both pass) -- but at p=0.02 Holm gives 0.04 and rejects both,
    while WY sees one effective test and adjusts to ~0.02... which also rejects.
    So compare the other way: p=0.03 under WY adjusts to ~0.03, rejecting, where
    Holm's 0.06 does not.  WY is strictly more powerful here.
    """
    src = """
        def sample(rng):
            return float(rng.random())

        def test_a(assertNotReject): assertNotReject(0.03, null_sample=sample)
        def test_b(assertNotReject): assertNotReject(0.03, null_sample=sample)
    """
    wy = run_wy(pytester, src, resamples=2000)
    wy.assert_outcomes(failed=2)   # WY: adjusted ~0.03 <= 0.05 -> rejected

    pytester.makepyfile(src)
    holm = pytester.runpytest("--alpha=0.05", "-v")
    holm.assert_outcomes(passed=2)  # Holm: adjusted 0.06 > 0.05 -> not rejected


# ---------------------------------------------------------------------------
# Sample-size helper unit tests (no pytester needed)
# ---------------------------------------------------------------------------

class TestZtestN:
    def test_known_result_two_sided(self):
        # Cohen's d=0.5, alpha=0.05, power=0.8, two-sided -> 32 (standard result)
        assert _ztest_n(0.05, 0.8, 0.5, two_sided=True) == 32

    def test_one_sided_smaller_than_two_sided(self):
        n_two = _ztest_n(0.05, 0.8, 0.5, two_sided=True)
        n_one = _ztest_n(0.05, 0.8, 0.5, two_sided=False)
        assert n_one < n_two

    def test_larger_effect_needs_fewer_samples(self):
        assert _ztest_n(0.05, 0.8, 1.0) < _ztest_n(0.05, 0.8, 0.5)

    def test_higher_power_needs_more_samples(self):
        assert _ztest_n(0.05, 0.9, 0.5) > _ztest_n(0.05, 0.8, 0.5)

    def test_lower_alpha_needs_more_samples(self):
        assert _ztest_n(0.01, 0.8, 0.5) > _ztest_n(0.05, 0.8, 0.5)

    def test_returns_int(self):
        assert isinstance(_ztest_n(0.05, 0.8, 0.5), int)


class TestChisquareN:
    def test_known_result(self):
        # w=0.3, df=3, alpha=0.05, power=0.8 -> ~121 (standard result)
        n = _chisquare_n(0.05, 0.8, 0.3, df=3)
        assert 115 <= n <= 130

    def test_larger_effect_fewer_samples(self):
        assert _chisquare_n(0.05, 0.8, 0.5, 3) < _chisquare_n(0.05, 0.8, 0.3, 3)

    def test_more_df_more_samples(self):
        assert _chisquare_n(0.05, 0.8, 0.3, 6) > _chisquare_n(0.05, 0.8, 0.3, 3)

    def test_higher_power_more_samples(self):
        assert _chisquare_n(0.05, 0.9, 0.3, 3) > _chisquare_n(0.05, 0.8, 0.3, 3)

    def test_returns_int(self):
        assert isinstance(_chisquare_n(0.05, 0.8, 0.3, 3), int)

    def test_achieved_power_at_n(self):
        from scipy.stats import chi2, ncx2
        w, df = 0.3, 3
        n = _chisquare_n(0.05, 0.8, w, df)
        crit = chi2.ppf(0.95, df)
        achieved = ncx2.sf(crit, df, n * w ** 2)
        assert achieved >= 0.8
        # One fewer sample should fall below the target.
        achieved_minus1 = ncx2.sf(crit, df, (n - 1) * w ** 2)
        assert achieved_minus1 < 0.8


class TestKsN:
    def test_known_result_one_sample(self):
        # delta=0.1, alpha=0.05, power=0.8
        # n = (sqrt(ln(40)) + sqrt(ln(10)))^2 / (2 * 0.01) ≈ 591.1 -> 592
        n = _ks_n(0.05, 0.8, 0.1, two_sample=False)
        assert n == 592

    def test_two_sample_is_double_one_sample(self):
        n_one = _ks_n(0.05, 0.8, 0.1, two_sample=False)
        n_two = _ks_n(0.05, 0.8, 0.1, two_sample=True)
        assert n_two == n_one * 2

    def test_larger_effect_fewer_samples(self):
        assert _ks_n(0.05, 0.8, 0.2) < _ks_n(0.05, 0.8, 0.1)

    def test_higher_power_more_samples(self):
        assert _ks_n(0.05, 0.9, 0.1) > _ks_n(0.05, 0.8, 0.1)

    def test_lower_alpha_more_samples(self):
        assert _ks_n(0.01, 0.8, 0.1) > _ks_n(0.05, 0.8, 0.1)

    def test_returns_int(self):
        assert isinstance(_ks_n(0.05, 0.8, 0.1), int)

    def test_dkw_bound_satisfied_at_n(self):
        # Verify the DKW-derived n actually satisfies both the alpha and power bounds.
        alpha, power, delta = 0.05, 0.8, 0.1
        n = _ks_n(alpha, power, delta)
        # Critical value from DKW: c = sqrt(ln(2/alpha) / (2n))
        c_alpha = math.sqrt(math.log(2 / alpha) / (2 * n))
        # Lower-bound on power from DKW at the true effect delta:
        power_lb = 1 - 2 * math.exp(-2 * n * (delta - c_alpha) ** 2)
        assert power_lb >= power - 1e-9  # allow tiny float rounding


# ---------------------------------------------------------------------------
# Fixture integration tests (via pytester)
# ---------------------------------------------------------------------------

def test_ztest_sample_size_fixture(pytester):
    pytester.makepyfile("""
        def test_uses_fixture(ztest_sample_size):
            n = ztest_sample_size(effect_size=0.5)
            assert n == 32  # known result for alpha=0.05, power=0.8
    """)
    result = pytester.runpytest("--alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


def test_chisquare_sample_size_fixture(pytester):
    pytester.makepyfile("""
        def test_uses_fixture(chisquare_sample_size):
            n = chisquare_sample_size(effect_size=0.3, df=3)
            assert 115 <= n <= 130
    """)
    result = pytester.runpytest("--alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


def test_ks_sample_size_fixture(pytester):
    pytester.makepyfile("""
        def test_uses_fixture(ks_sample_size):
            n = ks_sample_size(effect_size=0.1)
            assert n == 592  # known result for alpha=0.05, power=0.8
    """)
    result = pytester.runpytest("--alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


# ---------------------------------------------------------------------------
# The Holm sizing ladder: who consumes a rung, and in what order
# ---------------------------------------------------------------------------

def test_non_participating_test_does_not_consume_a_ladder_rung(pytester):
    """A test that sizes but never calls assertNotReject is not in the family.

    It used to bump the shared counter, pushing every later test onto a looser
    rung (and some past the end of the ladder onto the full alpha), which
    silently under-sized them.
    """
    pytester.makepyfile("""
        import json, pathlib

        def record(name, n):
            pathlib.Path(name).write_text(json.dumps(n))

        def test_outsider(ztest_sample_size):
            # No assertNotReject -- not part of the correction family.
            record("outsider.json", ztest_sample_size(effect_size=0.3))

        def test_first(assertNotReject, ztest_sample_size):
            record("first.json", ztest_sample_size(effect_size=0.3))
            assertNotReject(0.5)

        def test_second(assertNotReject, ztest_sample_size):
            record("second.json", ztest_sample_size(effect_size=0.3))
            assertNotReject(0.5)
    """)
    result = pytester.runpytest("--alpha=0.05")
    result.assert_outcomes(passed=3)

    def n_of(name):
        return json.loads((pytester.path / f"{name}.json").read_text())

    # m=2 participating tests, so the ladder is alpha/2 then alpha.
    assert n_of("first") == _ztest_n(0.05 / 2, 0.8, 0.3)
    assert n_of("second") == _ztest_n(0.05, 0.8, 0.3)
    # The outsider is uncorrected: its p-value is never adjusted.
    assert n_of("outsider") == _ztest_n(0.05, 0.8, 0.3)


def test_rung_follows_collection_position_not_request_order(pytester):
    """Each participating test occupies its own rung, whether or not it sizes.

    The middle test is in the family but never asks for a sample size.  Keying k
    off collection position leaves it holding rung 2, so the third test gets
    rung 3 (full alpha).  Request-order numbering would instead compact the
    ladder and hand the third test rung 2.
    """
    pytester.makepyfile("""
        import json, pathlib

        def test_a(assertNotReject, ztest_sample_size):
            pathlib.Path("a.json").write_text(
                json.dumps(ztest_sample_size(effect_size=0.3)))
            assertNotReject(0.5)

        def test_b(assertNotReject):
            # In the family, but never sizes.
            assertNotReject(0.5)

        def test_c(assertNotReject, ztest_sample_size):
            pathlib.Path("c.json").write_text(
                json.dumps(ztest_sample_size(effect_size=0.3)))
            assertNotReject(0.5)
    """)
    pytester.runpytest("--alpha=0.05").assert_outcomes(passed=3)

    assert json.loads((pytester.path / "a.json").read_text()) == _ztest_n(
        0.05 / 3, 0.8, 0.3
    )
    assert json.loads((pytester.path / "c.json").read_text()) == _ztest_n(
        0.05, 0.8, 0.3
    )


def test_every_sizing_fixture_in_a_test_sees_the_same_alpha(pytester):
    """One test, several sizing fixtures, one rung between them."""
    pytester.makepyfile("""
        import json, pathlib

        def test_both(assertNotReject, ztest_sample_size, ks_sample_size):
            pathlib.Path("both.json").write_text(json.dumps([
                ztest_sample_size(effect_size=0.3),
                ks_sample_size(effect_size=0.1),
            ]))
            assertNotReject(0.5)

        def test_other(assertNotReject, ztest_sample_size):
            ztest_sample_size(effect_size=0.3)
            assertNotReject(0.5)
    """)
    pytester.runpytest("--alpha=0.05").assert_outcomes(passed=2)

    z_n, ks_n = json.loads((pytester.path / "both.json").read_text())
    # Both fixtures must have used rung 1 of 2, i.e. alpha/2.
    assert z_n == _ztest_n(0.05 / 2, 0.8, 0.3)
    assert ks_n == _ks_n(0.05 / 2, 0.8, 0.1)


def test_power_option_affects_sample_size(pytester):
    pytester.makepyfile("""
        def test_n_90(ztest_sample_size):
            n = ztest_sample_size(effect_size=0.5)
            assert n > 32  # must be larger than at power=0.8
    """)
    result = pytester.runpytest("--power=0.9")
    result.assert_outcomes(passed=1)
