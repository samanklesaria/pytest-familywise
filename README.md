# pytest-familywise

A pytest plugin for running multiple randomized tests while controlling the
family-wise error rate (FWER) via step-down multiple-comparison procedures and group sequential tests. [![][docs-dev-img]][docs-dev-url]

## Installation and loading

Add the package as a dev dependency:

```
pip add --dev pytest-familywise
```

That's all you need to do! The package's fixtures (`assertNotReject, ztest_sample_size`, etc) become available in every test file without any import. 


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



## Using the Westfall-Young Procedure

Holm or Sidak corrections treats those three tests as three independent chances to be wrong.  When
your tests instead share a dataset, they are correlated: Westfall-Young can
measure that instead.  Each test supplies a `null_sample`
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

## Testing Step Down Procedures for Multiple Hypothesis Testing

Each raw p-value $p_k$ is
mapped to an adjusted value $\tilde{p}_k$, and null hypothesis $k$ is rejected when
$\tilde{p}_k \le \alpha$.  The p-values are adjusted to be monotonically increasing, so the first time we fail to reject a null hypothesis, we'll fail to reject the rest of the null hypotheses as well. To be specific, say we have $K$ tests with p-values sorted ascending as $p_1 \le p_2 \le \cdots \le p_K$. We set $\tilde{p}_k = \max\left(\tilde{p}_{k-1},P( c(S_k) < p_k) \right)$ where $S_k = \{k, k+1, \dotsc, K\}$, the random variable $c(S) = \min_{j \in S} P_j$ , and $P$ is distributed according to the joint null distribution. 

Let $I_0$ be the set of true null hypotheses. Say $L$ is the first true null hypothesis in our ordering. That is, $p_L$ is an observation of $c(I_0)$. By monotonicity, the event that we reject a true null hypothesis is the same as the event that we reject hypothesis $L$. We reject at $L$ when $P( c(S_L) \leq p_L) \leq \alpha$. Because \(L\) is the first true null, $I_0 \subseteq S_L$. So $c(S_L) \leq c(I_0)$ and $P(c(S_L) \leq p_L ) \geq P(c(I_0) \leq p_L)$. Let $F$ be the CDF of $c(I_0)$. This means that $F(p_L) \leq \alpha$ or $p_L \leq F^{-1}(\alpha)$ when we reject. As $p_L$ has CDF $F$, this occurs with probability $\alpha$. 

Where the procedures differ is how they estimate $P(c(S_k) \leq p_k)$. For Westfall-Young, we can simulate samples of $c(S_k)$ and count the number of samples below $p_k$. This means the number of samples must exceed $1/\alpha$ for any rejection to be possible.

For Holm Bonferroni, we use a union bound: the probability the smallest $p$ value is below a threshold is at most the probability that *any* of the p-values are below the threshold: $P(c(S_k) \leq p_k) \leq |S_k|p_k$. 

For the Sidak correction, we assume independence: $P(c(S_k) \leq p_k) = 1 - (1 - p_k)^{|S_k|}$. 



## Stepping Down with Independent Group-Sequential Testing

Instead of forcing all tests to draw the same number of samples, we'd like to be able to stop sampling if it becomes clear early on that a test is broken. Say we draw $T$ independent groups of samples in chunks of `groupsize` at a time, rejecting tests the first time their adjusted $p$ values go below a modified threshold $\alpha_K$ rather than $\alpha$. Let $\tilde{p}_{i,t}$ be the adjusted p-value for test $i$ and group $t$. 

We will choose $\alpha_K$ so that
$$
P_{H_0}(\min_{t \in [T]} (\min_{k \in I_0} \tilde{p}_{k,t}) \leq \alpha_K) \leq \alpha
$$
This is the same as the condition that 
$$
P(\min_{t \in [T]} \tilde{p}_{L_t,t} \leq \alpha_K) \leq \alpha \\
\sqrt[K]{1 - \alpha} \leq P(\tilde{p}_{L, t} > \alpha_K)
$$
But $\tilde{p}_{L,t}$ were transformed so that $P(\tilde{p}_{L, t} < \alpha_K) = \alpha_K$. So $P(\tilde{p}_{L, t} \geq \alpha_K) = 1 - \alpha_K$. Plugging this in gives $\alpha_K = 1 - \sqrt[K]{1 - \alpha}$.  



TODO: Can we avoid sampling groups independently and use the dependence structure like in standard group sequential testing? 



### Planning the number of groups

For each test $k$, we want to have probability $\beta$ of rejecting if the functionality we're testing is actually broken. That is, we want to achieve a specified power *per-test* even though we're bounding the *familywise* error rate. We want to choose $K$ to be the smallest number of looks for which
$$
\forall i. \, P_{H_1}(\min_t \tilde{p}_{i,t} < \alpha_K) \geq \beta
$$
This means
$$
1- \prod_t P_{H_1}(\tilde{p}_{i,t} > 1 - \sqrt[K]{1 - \alpha}) = \beta
$$
Now, we don't know the distribution of the $\tilde{p}$ under $H_1$. But we can appeal to the central limit theorem and assume that for large enough $K$, the underlying test statistic will be approximately normally distributed about some alternative mean $\mu$. Applying the delta method, this means the $p$ values will be approximately normal around $F_0(1-\mu)$ where $F_0$ is the test's null CDF. 



TODO: continue from here.



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
sizes against the conservative bound `alpha / m`.

A test that uses a sample-size fixture but **not** `assertNotReject` is not in
the family, so no correction applies to it and it sizes against the raw alpha.
It also does not raise *m* for anyone else.



[docs-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[docs-dev-url]: https://samanklesaria.github.io/pytest-familywise
