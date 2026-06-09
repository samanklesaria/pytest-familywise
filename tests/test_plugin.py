"""Tests for the Holm-Bonferroni pytest plugin."""
import math
import pytest
from pytest_random import _ztest_n, _chisquare_n, _ks_n


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(pytester, src: str, alpha: float = 0.05):
    pytester.makepyfile(src)
    return pytester.runpytest(f"--holm-alpha={alpha}", "-v")


# ---------------------------------------------------------------------------
# Single-test cases
# ---------------------------------------------------------------------------

def test_single_low_pvalue_passes(pytester):
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(0.001)
    """)
    result.assert_outcomes(passed=1)


def test_single_high_pvalue_fails(pytester):
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(0.9)
    """)
    result.assert_outcomes(failed=1)


def test_single_pvalue_exactly_at_alpha_passes(pytester):
    # With n=1: threshold = alpha/1 = alpha; p <= threshold -> pass
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(0.05)
    """, alpha=0.05)
    result.assert_outcomes(passed=1)


def test_single_pvalue_just_above_alpha_fails(pytester):
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(0.051)
    """, alpha=0.05)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# Holm-Bonferroni step-down logic
# ---------------------------------------------------------------------------

def test_holm_bonferroni_stops_at_first_failure(pytester):
    # n=3, alpha=0.05
    # sorted p-values: 0.01, 0.04, 0.007  ->  0.007, 0.01, 0.04
    # k=1: threshold=0.05/3=0.0167; 0.007<=0.0167 -> PASS
    # k=2: threshold=0.05/2=0.025;  0.01 <=0.025  -> PASS
    # k=3: threshold=0.05/1=0.05;   0.04 <=0.05   -> PASS
    result = run(pytester, """
        def test_a(pvalue): pvalue(0.01)
        def test_b(pvalue): pvalue(0.04)
        def test_c(pvalue): pvalue(0.007)
    """)
    result.assert_outcomes(passed=3)


def test_holm_bonferroni_second_fails_rest_also_fail(pytester):
    # n=3, alpha=0.05
    # sorted: 0.01, 0.03, 0.07
    # k=1: threshold=0.0167; 0.01<=0.0167 -> PASS
    # k=2: threshold=0.025;  0.03>0.025   -> FAIL (stop)
    # k=3: stop -> FAIL
    result = run(pytester, """
        def test_a(pvalue): pvalue(0.01)
        def test_b(pvalue): pvalue(0.03)
        def test_c(pvalue): pvalue(0.07)
    """)
    result.assert_outcomes(passed=1, failed=2)


def test_all_fail_when_first_exceeds_threshold(pytester):
    # n=2, alpha=0.05
    # sorted: 0.04, 0.08
    # k=1: threshold=0.05/2=0.025; 0.04>0.025 -> FAIL (stop)
    # k=2: stop -> FAIL
    result = run(pytester, """
        def test_a(pvalue): pvalue(0.04)
        def test_b(pvalue): pvalue(0.08)
    """)
    result.assert_outcomes(failed=2)


def test_all_pass_when_all_small(pytester):
    result = run(pytester, """
        def test_a(pvalue): pvalue(0.001)
        def test_b(pvalue): pvalue(0.002)
        def test_c(pvalue): pvalue(0.003)
        def test_d(pvalue): pvalue(0.004)
    """)
    result.assert_outcomes(passed=4)


# ---------------------------------------------------------------------------
# Non-pvalue tests are unaffected
# ---------------------------------------------------------------------------

def test_normal_passing_test_unaffected(pytester):
    result = run(pytester, """
        def test_ordinary():
            assert 1 + 1 == 2

        def test_stat(pvalue):
            pvalue(0.001)
    """)
    result.assert_outcomes(passed=2)


def test_normal_failing_test_unaffected(pytester):
    result = run(pytester, """
        def test_ordinary():
            assert False

        def test_stat(pvalue):
            pvalue(0.001)
    """)
    result.assert_outcomes(passed=1, failed=1)


# ---------------------------------------------------------------------------
# Exception in a pvalue test => fails as an error, not statistically
# ---------------------------------------------------------------------------

def test_exception_before_pvalue_call_fails_normally(pytester):
    result = run(pytester, """
        def test_raises(pvalue):
            raise RuntimeError("boom")
            pvalue(0.001)
    """)
    # Uncaught exceptions in the test body are reported as 'failed', not 'errors'
    result.assert_outcomes(failed=1)


def test_exception_after_pvalue_call(pytester):
    # p-value was set, but test also raised -> test fails normally;
    # the plugin should not override it to passed.
    result = run(pytester, """
        def test_raises(pvalue):
            pvalue(0.001)
            raise RuntimeError("boom after pvalue")
    """)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# custom alpha
# ---------------------------------------------------------------------------

def test_custom_alpha(pytester):
    # alpha=0.01, n=1: threshold=0.01; p=0.05>0.01 -> FAIL
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(0.05)
    """, alpha=0.01)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# pvalue validation
# ---------------------------------------------------------------------------

def test_invalid_pvalue_raises(pytester):
    result = run(pytester, """
        def test_foo(pvalue):
            pvalue(1.5)
    """)
    result.assert_outcomes(failed=1)


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

def test_parametrized(pytester):
    # Each param gets its own p-value; treated as independent tests.
    # n=3, alpha=0.05; p-values: 0.001, 0.002, 0.003 -> all pass
    result = run(pytester, """
        import pytest

        @pytest.mark.parametrize("p", [0.001, 0.002, 0.003])
        def test_param(pvalue, p):
            pvalue(p)
    """)
    result.assert_outcomes(passed=3)


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
    result = pytester.runpytest("--holm-alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


def test_chisquare_sample_size_fixture(pytester):
    pytester.makepyfile("""
        def test_uses_fixture(chisquare_sample_size):
            n = chisquare_sample_size(effect_size=0.3, df=3)
            assert 115 <= n <= 130
    """)
    result = pytester.runpytest("--holm-alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


def test_ks_sample_size_fixture(pytester):
    pytester.makepyfile("""
        def test_uses_fixture(ks_sample_size):
            n = ks_sample_size(effect_size=0.1)
            assert n == 592  # known result for alpha=0.05, power=0.8
    """)
    result = pytester.runpytest("--holm-alpha=0.05", "--power=0.8")
    result.assert_outcomes(passed=1)


def test_power_option_affects_sample_size(pytester):
    pytester.makepyfile("""
        def test_n_90(ztest_sample_size):
            n = ztest_sample_size(effect_size=0.5)
            assert n > 32  # must be larger than at power=0.8
    """)
    result = pytester.runpytest("--power=0.9")
    result.assert_outcomes(passed=1)
