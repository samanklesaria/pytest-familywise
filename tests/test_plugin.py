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
