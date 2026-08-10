# pytest-familywise

A pytest plugin for running multiple randomized tests while controlling the
family-wise error rate (FWER) via a step-down multiple-comparison procedure —
Holm-Bonferroni or Westfall-Young. [![][docs-dev-img]][docs-dev-url]

## Motivation

A test suite that contains several independent statistical tests will, under the
null hypothesis, produce at least one false positive with probability greater
than the nominal level α.  For $m$ independent tests each at level $\alpha$ the FWER
is $1 - (1-\alpha)^m$.  Holm-Bonferroni corrects for this without being as
conservative as a plain Bonferroni adjustment.

The complication is that a step-down procedure must process p-values from
smallest to largest — the threshold for rank *k* depends on the total count *m*
and all smaller p-values before it.  This plugin defers pass/fail decisions:
every test runs to completion first, p-values are collected, and then the
procedure is applied once to the full set.

Two procedures are available via `--correction`:

| | Assumes | Needs from you | Power |
|---|---|---|---|
| `holm` (default) | nothing about dependence | just the p-value | conservative when tests are correlated |
| `westfall-young` | you can resample under H0 | a `null_sample` callable per test | recovers what Holm gives up |

Holm bounds the joint null distribution of the p-value vector; Westfall-Young
*estimates* it by resampling.  If your tests are correlated — several tests over
the same generated dataset or the same RNG, which is the common case — Holm
charges you for $m$ independent tests you don't have.  Westfall-Young does not.

## Installation and loading

Add the package as a dev dependency:

```
pip add --dev pytest-familywise
```

That is all that is needed.  The package declares a `pytest11` entry point:

```toml
# pyproject.toml
[project.entry-points."pytest11"]
random = "pytest_familywise"
```

pytest scans installed `pytest11` entry points at startup and loads matching
modules automatically.  The fixtures (`assertNotReject`, `ztest_sample_size`, etc.) are
defined at module level in `pytest_familywise`, so they become available in every
test file without any import or `conftest.py` change.


## Quick example

```python
import numpy as np
import scipy.stats

def test_uniform_marginals(ks_sample_size, assertNotReject):
    """Each output coordinate of our RNG should be marginally uniform."""
    n = ks_sample_size(effect_size=0.05)   # detect CDF deviation >= 5 pp
    samples = np.random.rand(n)
    result = scipy.stats.kstest(samples, "uniform")
    assertNotReject(result.pvalue)


def test_normal_mean_zero(ztest_sample_size, assertNotReject):
    """Standardised output should have mean zero"""
    n = ztest_sample_size(effect_size=0.3)   # Cohen's d = 0.3
    samples = np.random.randn(n)
    _, p = scipy.stats.ttest_1samp(samples, 0.0)
    assertNotReject(p)


def test_discrete_distribution(chisquare_sample_size, assertNotReject):
    """A categorical sampler should match its target probabilities."""
    n = chisquare_sample_size(effect_size=0.2, df=4)   # Cohen's w = 0.2
    observed = np.random.multinomial(n, [0.2] * 5)
    _, p = scipy.stats.chisquare(observed)
    assertNotReject(p)
```

Run with:

```
pytest --alpha=0.05 --power=0.8
```

After all three tests complete, the plugin applies the correction and appends
a summary to the terminal output:

```
============ Holm-Bonferroni correction  α=0.05  n=3 =============
  PASSED  p=0.312541  adjusted p=0.937623  test_rng.py::test_uniform_marginals
  PASSED  p=0.487302  adjusted p=0.974604  test_rng.py::test_normal_mean_zero
  PASSED  p=0.621088  adjusted p=0.974604  test_rng.py::test_discrete_distribution

  3 passed, 0 failed after Holm-Bonferroni correction
```

The exit code is non-zero if any null hypothesis is rejected.

## Quick example: Westfall-Young

Holm treats those three tests as three independent chances to be wrong.  When
your tests instead share a dataset, they are correlated, and Westfall-Young can
measure that instead of assuming the worst.  Each test supplies a `null_sample`
callable that recomputes its own p-value under H0:

```python
import numpy as np
import pytest
import scipy.stats


@pytest.fixture(scope="module")
def samples():
    """One dataset, exercised by several tests — so their p-values are correlated."""
    return np.random.default_rng(0).standard_normal(2000)


def under_h0(rng, data):
    """Impose H0 (mean zero) on the OBSERVED data, then bootstrap."""
    centered = data - data.mean()
    return rng.choice(centered, size=len(data), replace=True)


def test_mean_zero(assertNotReject, samples):
    p = scipy.stats.ttest_1samp(samples, 0.0).pvalue
    assertNotReject(p, null_sample=lambda rng:
        scipy.stats.ttest_1samp(under_h0(rng, samples), 0.0).pvalue)


def test_median_zero(assertNotReject, samples):
    """A different statistic on the same data — strongly correlated with the above."""
    def pvalue(data):
        return scipy.stats.binomtest(int((data > 0).sum()), len(data), 0.5).pvalue

    assertNotReject(pvalue(samples), null_sample=lambda rng:
        pvalue(under_h0(rng, samples)))
```

Run with:

```
pytest --correction=westfall-young --resamples=2000
```

```
==================== Westfall-Young correction  α=0.05  n=2 ====================
  PASSED  p=0.134070  adjusted p=0.218500  examples/westfall_young.py::test_median_zero
  PASSED  p=0.210432  adjusted p=0.218500  examples/westfall_young.py::test_mean_zero

  2 passed, 0 failed after Westfall-Young correction
```

Both tests route their randomness through the same `under_h0` helper, so at
resample *b* they see the same bootstrap draw — their null columns stay aligned
and the procedure can see how correlated they are.  Drop `--correction` and Holm
charges rather more for the same evidence:

```
=================== Holm-Bonferroni correction  α=0.05  n=2 ====================
  PASSED  p=0.134070  adjusted p=0.268140  examples/westfall_young.py::test_median_zero
  PASSED  p=0.210432  adjusted p=0.268140  examples/westfall_young.py::test_mean_zero
```

0.2185 versus 0.2681 — Holm doubles the smallest p-value because it must assume
the two tests are independent evidence.  They are not, and Westfall-Young
measures how much they overlap.  Neither rejects here, but the gap is the margin
you get back, and it is what decides borderline cases.

### Permutation instead of bootstrap

When H0 says two groups are exchangeable, the transform is a shuffle of the
pooled data:

```python
def test_groups_are_identical(assertNotReject, group_a, group_b):
    pooled = np.concatenate([group_a, group_b])
    n = len(group_a)

    def pvalue(x):
        return scipy.stats.ks_2samp(x[:n], x[n:]).pvalue

    assertNotReject(pvalue(pooled), null_sample=lambda rng:
        pvalue(rng.permutation(pooled)))
```

Sign-flipping (`data * rng.choice([-1, 1], size=len(data))`) is the third common
idiom, for H0s that assert symmetry about zero.

---

## The Step Down Procedures

Both procedures are expressed as **adjusted p-values**: each raw p-value $p_i$ is
mapped to an adjusted value $\tilde{p}_i$, and null hypothesis $i$ is rejected when
$\tilde{p}_i \le \alpha$.  The p-values are adjusted to be monotonically increasing, so the first time we fail to reject a null hypothesis, we'll fail to reject the rest of the null hypotheses as well. To be specific, say we have $m$ tests with p-values sorted ascending as $p_1 \le p_2 \le \cdots \le p_m$. 

For Holm Bonferroni, $\tilde{p}_k = \max\left(\tilde{p}_{k-1},\ \min(1,\ (m - k + 1)\, p_k)\right)$. Let $m_0$ be the number of true null hypotheses. Say our first true rejected null hypothesis was \(L\). The number of previously considered hypotheses \(L-1\) can't be more than the number of false hypotheses \(m - m_0\). If we had previously considered a true hypothesis and accepted it, monotonicity would require us to accept hypothesis $L$ too. So \(L - 1 \leq m - m_0\). For us to reject \(L\), we'd have to have \(p_L \leq \frac{\alpha}{m - L + 1} \leq \frac{\alpha}{m_0}\). Taking a union bound over all $m_0$ true null hypotheses bounds the family-wise error rate by $\alpha$. 

For Westfall Young, $\tilde{p}_k = \max\left(\tilde{p}_{k-1},P( c(S_k) < p_k) \right)$ where $S_k = \{k, k+1, \dotsc, m\}$, the random variable $c(S) = \min_{j \in S} P_j$ , and $P$ is distributed according to the joint null distribution.  Say $L$ is the first true null hypothesis in our ordering. By monotonicity, the event that we reject a true null hypothesis is the same as the event that we reject hypothesis $L$. We reject at $L$ when $P( c(S_L) \leq p_L) \leq \alpha$. Let $I_0$ be the set of true null hypotheses. Because \(L\) is the first true null, $I_0 \subseteq S_L$. So $c(S_L) \leq c(I_0)$ and $\{c(S_L) < p_L\} \subseteq \{c(I_0) \leq p_L\}$.  Let $F$ be the CDF of $c(I_0)$ so that our rejection event is contained in $\{F(p_L) \leq \alpha\}$. Let $p_L'$ be the smallest observed p-value for hypotheses in $I_0$. As $p_L$ was the smallest observed p-value for hypotheses in $S_k$, $p'_L \geq p_L$. So $\{F(p_L) \leq \alpha\} \subseteq \{F(p'_L) \leq \alpha\}$. As $p'_L$ also has cdf $F$, so we can write this as $\{F(F^{-1}(u)) \leq \alpha\}$ where $u$ is uniformly distributed. This occurs with probability $\alpha$. 



Note that to estimate $P(c(S_L) \leq p_L)$ on Westfall-Young, we take samples of $c(S_L)$ and count the number below $p_L$. This means the number of samples (given by the `--resamples` argument) must exceed $1/\alpha$ for any rejection to be possible.



## CLI options

| Option | Default | Description |
|---|---|---|
| `--alpha` | `0.05` | Family-wise error rate |
| `--correction` | `holm` | `holm` or `westfall-young` |
| `--resamples` | `1000` | Null resamples *B* per test (Westfall-Young only) |
| `--power` | `0.8` | Per-test power used by the sample-size fixtures |
| `--calibration` | `.familywise-calibration.npz` | Recorded null p-values, used to size samples against the measured threshold. Delete the file to re-record it; empty string disables |

`--power` is per-test, not family-wise.  The sample-size fixtures use
Holm-Bonferroni corrected significance levels rather than the raw alpha.  At
collection time the plugin records which tests use `assertNotReject` (*m* of
them); the *k*-th of those, **in collection order**, sizes against
`alpha / (m - k + 1)`.  The first receives the most stringent threshold
(`alpha / m`) and therefore the largest sample size; later tests receive
progressively relaxed thresholds and smaller samples.  Because of this, it is
worth ordering your test suite so that more computationally expensive tests come
later, where the required sample sizes are smaller.

Two consequences worth knowing:

- A test that uses a sample-size fixture but **not** `assertNotReject` is not in
  the family, so no correction applies to it and it sizes against the raw alpha.
  It does not consume a rung of the ladder.
- Because *k* is a collection position rather than a count of who asked first,
  sample sizes are reproducible: they do not shift under `-k`, `--lf`, or
  `pytest-randomly`.

By default the sample-size fixtures use Holm's threshold under **both**
procedures. Westfall-Young's adjusted p-values are always ≤ Holm's, so sizing
against Holm over-sizes safely — your tests come out at least as powered as you
asked for. To stop over-sizing, calibrate.

## Calibration: sizing against the measured threshold

Westfall-Young's real critical value depends on the null distribution the tests
generate, so it isn't known before they run. A calibration file breaks that
circularity — record the null columns once, reuse them to size later runs:

```
rm -f .familywise-calibration.npz
pytest --correction=westfall-young --resamples=40000   # records; sizes with Holm
pytest --correction=westfall-young --resamples=40000   # loads; sizes smaller
```

```
calibration: recording a new calibration; sizing with Holm
calibration: wrote .familywise-calibration.npz (2 tests, B=40000)
...
calibration: sizing alpha=0.032212 for 2 tests (Holm would use 0.025000)
```

In [`examples/calibration.py`](examples/calibration.py) — a t-test and a sign
test over one dataset, measured rank correlation 0.48 — that takes *n* from 238
to 223.

The threshold is

$$c_1 = \text{quantile}_\alpha\left(\min_j P_{b,j}\right)$$

the α-quantile of the per-resample minimum null p-value: the harshest threshold
any test can face under the step-down procedure, and so the right one to size
against, since a test with a real effect is the one likely to land at rank 1.
Theory brackets it as $\alpha/m \le c_1 \le \alpha$ and the estimate is clamped
into that range, so **calibrated sizing can only shrink sample sizes**.

### What it costs and when it helps

**It only affects power.** FWER control always comes from the current run's own
null draws, so a missing, stale, subsetted or corrupt calibration costs sample
size and nothing else. Every failure path warns and falls back to Holm; it never
degrades silently.

**The gain comes entirely from positive dependence between tests.** Independent
tests give $c_1 \approx \alpha/m$ — Bonferroni, no gain. Roughly, for the first
requester at α=0.05:

| m | dependence | n saving |
|---|---|---|
| any | independent | 0.3% |
| 10 | ρ=0.7 | 11% |
| 50 | ρ=0.9 | 31% |
| 50 | perfect | 54% |

So this is for suites where many tests hit one shared dataset or RNG. If your
tests are independent, it will correctly find nothing.

**Resolving a gain costs resamples.** The estimate carries about
$1/\sqrt{B\alpha}$ relative error and is deliberately biased low (an unlucky
pilot must not silently under-size your suite), so a modest real gain can be
swallowed at small *B*. At B=4000 the example above resolves nothing; at B=40000
it does. When the estimate lands on the Bonferroni clamp the run says so:

```
calibration: no dependence resolved at B=4000; raise --resamples if these tests share data
```

**A recording run sizes every test identically** (uniform `alpha/m` rather than
Holm's ladder). A per-test *n* means a per-test number of draws from the shared
rng stream, which decorrelates the very columns being recorded — the pilot would
destroy the dependence it exists to measure.

**The file is not portable across suites.** It is keyed by nodeid, so adding a
test falls back until you re-record; subsetting with `-k` or `--lf` still works,
since $c_1$ is recomputed over whichever tests are present. It holds random
draws and is rewritten wholesale, so gitignore it rather than committing it.

**To re-record, delete it.** Absence of the file *is* the record signal, so `rm`
is the whole interface — there is no `--recalibrate` flag.

### Where it does not transfer

The cached columns are reused at whatever *n* the loading run picks, which
assumes the cross-test dependence is stable in *n*. That is asymptotically true
for continuous, correctly specified tests, and degrades when statistics are
discrete (chi-square with small expected counts), when tests mix a fixed-size
fixture with n-sized draws, or when `null_sample` consumes the rng an
n-dependent number of times. The run warns when the *n* it hands out differs from
the pilot's by more than 2×; beyond that, delete the file and re-record rather
than trusting it.

## Fixtures

### `assertNotReject`

```python
def test_something(assertNotReject):
    p = run_statistical_test()
    assertNotReject(p)   # registers the p-value; plugin decides pass/fail
```

The test passes if the null hypothesis is *not* rejected after correction (i.e.
the p-value is large enough).  It fails if H0 is rejected.

Calling `assertNotReject(p)` with a value outside [0, 1] raises `ValueError`.
If a test raises an exception before `assertNotReject` is called, it fails
normally and is excluded from the correction set.

#### Under `--correction=westfall-young`

A second argument is required: a callable that recomputes *this test's* p-value
under H0.  It receives a seeded numpy generator.

```python
def test_mean_zero(assertNotReject, data):
    _, p = scipy.stats.ttest_1samp(data, 0.0)

    def under_h0(rng):
        centered = data - data.mean()            # impose H0 on the OBSERVED data
        boot = rng.choice(centered, size=len(data))
        return scipy.stats.ttest_1samp(boot, 0.0)[1]

    assertNotReject(p, null_sample=under_h0)
```

**Transform the observed data; do not draw fresh data from a model.** Permute
labels, sign-flip, center-then-bootstrap.  Sampling from an assumed-true model is
only valid when H0 actually holds, and it throws away the dependence structure
Westfall-Young exists to exploit.

**Alignment.** Resample *b* uses `default_rng(seed + b)`, constructed identically
in every test.  So two tests that apply the same transform to the same data see
the same draw at index *b*, and their null columns come out aligned — which is
what lets the procedure detect that they are correlated and stop charging them
for each other.

This holds only if the tests consume the rng compatibly (same transform, same
call order), and the plugin cannot verify that.  Tests whose transforms differ
still get **valid** results — just with less of a power gain over Holm.  The
seed is fixed rather than configurable, so runs are reproducible by default.

### `ztest_sample_size`

```python
n = ztest_sample_size(effect_size=0.5)               # two-sided (default)
n = ztest_sample_size(effect_size=0.5, two_sided=False)
```

`effect_size` is Cohen's d.  Uses the exact closed form:

$$n = \left\lceil \left(\frac{z_\alpha + z_\beta}{d}\right)^2 \right\rceil$$

Returns per-group *n* for a two-sample test.

### `chisquare_sample_size`

```python
n = chisquare_sample_size(effect_size=0.3, df=4)
```

`effect_size` is Cohen's $w = \sqrt{\sum (p_i - p_{0i})^2 / p_{0i}}$; `df` is the degrees of
freedom (number of categories − 1 for goodness-of-fit).  Solves numerically via
the non-central χ² survival function.

### `ks_sample_size`

```python
n = ks_sample_size(effect_size=0.1)                 # one-sample
n = ks_sample_size(effect_size=0.1, two_sample=True) # per-group
```

`effect_size` is the maximum absolute CDF difference $\|F - G\|_\infty \in (0, 1]$.
Uses the DKW-inequality bound:

$$n \ge \frac{\left(\sqrt{\ln(2/\alpha)} + \sqrt{\ln(2/\beta)}\right)^2}{2\Delta^2}$$

where $\beta = 1 - \text{power}$.  For `two_sample=True` the effective sample size for the
two-sample KS statistic is $n_1 n_2/(n_1+n_2) = n_\text{each}/2$ (equal groups), so the
returned per-group count is double the formula above.

[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://samanklesaria.github.io/pytest-familywise
