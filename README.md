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

You hand `assertNotReject` a **sampler**: a callable that draws a group of data
from the generator it is given and returns the p-value for that group.

```python
import scipy.stats

def test_uniform_marginals(ks_sample_size, assertNotReject):
    """Each output coordinate of our RNG should be marginally uniform."""
    n = ks_sample_size(effect_size=0.05)   # detect CDF deviation >= 5 pp
    assertNotReject(lambda rng: scipy.stats.kstest(rng.random(n), "uniform").pvalue)


def test_normal_mean_zero(ztest_sample_size, assertNotReject):
    """Standardised output should have mean zero"""
    n = ztest_sample_size(effect_size=0.3)   # Cohen's d = 0.3

    def sample(rng):
        return scipy.stats.ttest_1samp(rng.standard_normal(n), 0.0).pvalue

    assertNotReject(sample)


def test_discrete_distribution(chisquare_sample_size, assertNotReject):
    """A categorical sampler should match its target probabilities."""
    n = chisquare_sample_size(effect_size=0.2, df=4)   # Cohen's w = 0.2

    def sample(rng):
        return scipy.stats.chisquare(rng.multinomial(n, [0.2] * 5)).pvalue

    assertNotReject(sample)
```

Ordinarily the sampler is called exactly once.  Under `--groupsize` it is
called once per group, which is what makes [group-sequential
testing](#group-sequential-testing) possible without rewriting your tests.

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
    assertNotReject(
        lambda rng: scipy.stats.ttest_1samp(samples, 0.0).pvalue,
        null_sample=lambda rng:
        scipy.stats.ttest_1samp(under_h0(rng, samples), 0.0).pvalue)


def test_median_zero(assertNotReject, samples):
    """A different statistic on the same data — strongly correlated with the above."""
    def pvalue(data):
        return scipy.stats.binomtest(int((data > 0).sum()), len(data), 0.5).pvalue

    assertNotReject(
        lambda rng: pvalue(samples),
        null_sample=lambda rng: pvalue(under_h0(rng, samples)))
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

    assertNotReject(
        lambda rng: pvalue(pooled),
        null_sample=lambda rng: pvalue(rng.permutation(pooled)))
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



## Group-sequential testing

Sizing every test for the effect you are worried about means a healthy suite
draws its full sample every run, even though most tests would have been settled
by a fraction of it — and a badly broken test pays the same price as a healthy
one.  `--groupsize` makes the suite draw incrementally instead:

```
pytest --groupsize=40
```

Every test draws one group of 40, the plugin applies the step-down correction
to the whole family, tests whose null is rejected stop, and the survivors draw
another group. 

Each look draws a *fresh, independent* group, and the sampler returns the
p-value for that group alone.  So under $H_0$ the group p-values $p_1, \dotsc,
p_K$ are independent and uniform.  Probit-transform and accumulate them:

$$X_j = \Phi^{-1}(1 - p_j) \sim \mathcal{N}(0,1), \qquad S_k = \sum_{j \le k} X_j$$

Let $M_l = \max_{k \leq l} S_k$ and its survival function $G_l(b) = P_{H_0}(M_l \geq b)$. Our group-sequential design will reject at the first look crossing boundary $c_K(\alpha)$. For us to have family-wise error rate $\alpha$, we must have $G_K(c_K(\alpha)) = \alpha$. 

Rather than compare $Z_k$ to $c_K$ at each look, the plugin inverts the
boundary into a **sequential p-value** $p^{*}_L = G_K(M_L)$ where $M_L$ is observed after $L$ looks. As $G$ is monotonically decreasing, $p^*_L \leq \alpha \iff G_K(M_L) \leq G_K(c_K(\alpha)) \iff M_L \geq c_K(\alpha)$, so our group-sequential scheme is equivalent to rejecting when $p^*_L \leq \alpha$. 

To compute $p^*_L$, recurse on the density $f_k$ of $S_k$ given that no earlier look crossed:

$$f_1(s) = \varphi(s), \qquad f_{k+1}(s) = \int_{-\infty}^{\,b\sqrt{k}} f_k(u)\, \varphi(s - u)\, du,$$

$$G(b) = 1 - \int_{-\infty}^b f_K(s)\, ds .$$

Independent increments turn a $K$-dimensional orthant probability into $K$
one-dimensional convolutions.  

### Fitting everything together

During testing, we repeat the following $K$ times:

- Run all the remaining unit tests on the next group of observations, getting raw p-values for each test.
- Adjust these raw p-values into sequential p values $p^*$.
- Further adjust these sequential $p^*$ values into step-down $\tilde{p}$ values. 
- Reject any tests for which $\tilde{p} \le \alpha$

### Planning the number of groups

With per-group drift $\mu = \mathbb{E}[X_j]$ under the alternative, $K$ is the smallest number of looks with

$$P_{\mu}\left(M_K \ge c_K(\alpha_i)\right) \ \ge\ 1 - \beta$$

at the same corrected level $\alpha_i = \alpha/m$ used everywhere else.  To get
$\mu$ for a group of `--groupsize` samples, the sample-size functions are run
backwards: find the power $\pi$ one group achieves at level $\alpha_i$, then
$\mu = \Phi^{-1}(\pi) + \Phi^{-1}(1 - \alpha_i)$.  That identity is exact when
the probit-transformed p-value is normal with unit variance (the z-test) and an
approximation otherwise — it selects the number of looks, so a poor $\mu$ costs
power, never error-rate control.



## CLI options

| Option | Default | Description |
|---|---|---|
| `--alpha` | `0.05` | Family-wise error rate |
| `--correction` | `holm` | `holm` or `westfall-young` |
| `--resamples` | `1000` | Null resamples per test (Westfall-Young only) |
| `--power` | `0.8` | Per-test power used by the sample-size fixtures |
| `--groupsize` | `0` | Samples per group for group-sequential testing; `0` disables |

`--power` is per-test, not family-wise.  The sample-size fixtures use corrected
significance levels rather than the raw alpha.  At collection time the plugin
records which tests use `assertNotReject` (*m* of them), and every one of them
sizes against `alpha / m`.

A test that uses a sample-size fixture but **not** `assertNotReject` is not in
the family, so no correction applies to it and it sizes against the raw alpha.
It also does not raise *m* for anyone else.

### Why `alpha / m` for every test

The threshold a test actually faces is set by its p-value **rank**, not by where
it sits in the file — and the rank isn't known until the whole suite has run.
The step-down ladder gives rank *k* a threshold of `alpha / (m - k + 1)`, so
rank 1 faces the strictest rung, `alpha / m`.

A test with a real effect produces a small p-value.  It therefore sorts to (or
near) rank 1 and faces `alpha / m` regardless of its position.  Sizing any test
for a laxer rung amounts to betting that it is not the broken one, and there is
nothing available before the run to place that bet with.  So every test is sized
for the rung it would face if it were the one to fail.

The cost is milder than it looks, because the penalty grows like `log m`, not
`m`.  For a z-test at power 0.8, relative to sizing at the uncorrected alpha:

| m | 5 | 10 | 20 | 50 | 100 |
|---|---|---|---|---|---|
| n | 1.49× | 1.70× | 1.90× | 2.18× | 2.38× |

Doubling the size of the suite costs about 10% more samples.  If that still
matters, the lever with real leverage is *m* itself — split unrelated tests into
separate runs so they stop paying for each other.

Under `--correction=westfall-young` the true rank-1 threshold is somewhat above
`alpha / m`; that gap is exactly what the procedure recovers at decision time.
Sizing still uses `alpha / m` there, so Westfall-Young runs are over-sized.
Erring large costs samples, erring small costs the power you asked for.

## Fixtures

### `assertNotReject`

```python
def test_something(assertNotReject):
    # fn(rng) draws one group of data and returns its p-value
    assertNotReject(lambda rng: run_statistical_test(rng))
```

The test passes if the null hypothesis is *not* rejected after correction (i.e.
the p-value is large enough).  It fails if H0 is rejected.

The argument is a **sampler**, not a p-value: a callable taking a numpy
generator and returning a p-value.  Without `--groupsize` it is called exactly
once; with it, once per group.  Passing a bare float raises `TypeError`.

A sampler returning a value outside [0, 1] raises `ValueError`.  If a test
raises an exception before `assertNotReject` is called — or from inside the
sampler on any look — it fails normally and is excluded from the correction
set.

#### Under `--correction=westfall-young`

A second argument is required: a callable that recomputes *this test's* p-value
under H0.  It receives a seeded numpy generator.

```python
def test_mean_zero(assertNotReject, data):
    def observed(rng):
        return scipy.stats.ttest_1samp(data, 0.0).pvalue

    def under_h0(rng):
        centered = data - data.mean()            # impose H0 on the OBSERVED data
        boot = rng.choice(centered, size=len(data))
        return scipy.stats.ttest_1samp(boot, 0.0)[1]

    assertNotReject(observed, null_sample=under_h0)
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
