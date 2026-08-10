"""Tests for the familywise-correction pytest plugin."""
import json
import math
import pytest
from numpy.random import default_rng
from pytest_familywise import (
    _ztest_n,
    _chisquare_n,
    _ks_n,
    _holm_adjusted,
    _westfall_young_adjusted,
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
# Sizing: every member of the family gets alpha/m, and only members count
# ---------------------------------------------------------------------------

def test_non_participating_test_does_not_inflate_m(pytester):
    """A test that sizes but never calls assertNotReject is not in the family.

    Counting it would raise m for everyone else, over-sizing the whole suite for
    a test whose p-value is never adjusted at all.
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

    # m=2 participating tests, so both size against alpha/2 -- position in the
    # file buys nothing, since either could be the one that lands at rank 1.
    assert n_of("first") == _ztest_n(0.05 / 2, 0.8, 0.3)
    assert n_of("second") == _ztest_n(0.05 / 2, 0.8, 0.3)
    # The outsider is uncorrected: its p-value is never adjusted.
    assert n_of("outsider") == _ztest_n(0.05, 0.8, 0.3)


def test_family_members_count_toward_m_even_if_they_never_size(pytester):
    """m is family membership, not a count of who asked for a sample size.

    The middle test uses assertNotReject but never sizes.  It still competes for
    rank 1 against the other two, so it still costs them a factor in m.
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

    for name in ("a", "c"):
        assert json.loads((pytester.path / f"{name}.json").read_text()) == _ztest_n(
            0.05 / 3, 0.8, 0.3
        )


def test_every_sizing_fixture_in_a_test_sees_the_same_alpha(pytester):
    """One test, several sizing fixtures, one threshold between them."""
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
    # Both fixtures must have used the family threshold, alpha/2.
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
