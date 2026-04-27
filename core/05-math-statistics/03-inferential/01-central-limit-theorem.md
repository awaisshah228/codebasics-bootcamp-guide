# Section 8 — Central Limit Theorem

## Lectures covered
- Random Sampling & Sample Bias
- The Law of Large Numbers
- Central Limit Theorem · Sampling Distribution
- Case Study: Solar Panels
- Standard Error · Quiz
- Z Score Table (Z-Table) · Quiz
- Confidence Interval · Confidence Interval: Estimate Car Miles · Exercise · Chapter Summary

---

## 1. Random sampling — what makes a sample valid

### Random sample
Every member of the population has an *equal* chance of being selected.

### Why it matters
A non-random sample produces biased estimates that don't generalize. Examples:
- Polling at a sports stadium → biased toward sports fans
- Survey at 9am Monday → biased toward office workers
- Online survey → biased toward people who use the internet

### Common biases
| Bias | What | Example |
|---|---|---|
| **Selection bias** | not everyone could be sampled | airport survey misses non-flyers |
| **Non-response bias** | who responds differs from who doesn't | political polls skew toward people with strong opinions |
| **Survivorship bias** | only looking at "winners" | "successful startups all have X" — but failed ones did too |
| **Confirmation bias** | analyst chooses cuts that confirm expectation | not strictly sampling, but worth flagging |

### Practical sampling techniques
- **Simple random** — uniformly at random
- **Stratified** — sample within subgroups, by their share (preserves balance)
- **Cluster** — sample whole groups (cities), then sample within
- **Systematic** — every k-th element (risky if data has a period)

For DS work: prefer stratified when key subgroups exist (e.g., gender split for a demographic study).

---

## 2. Law of Large Numbers (LLN)

> As sample size grows, the sample mean converges to the population mean.

```python
import numpy as np
true_mean = 5.0
xs = np.random.normal(loc=true_mean, scale=2, size=10_000)
running_avg = np.cumsum(xs) / np.arange(1, 10_001)
# running_avg approaches 5.0 as n grows
```

### What LLN tells us
With enough data, sample mean is a reliable estimate. **It does NOT guarantee small samples are reliable.**

---

## 3. Central Limit Theorem (CLT) — the load-bearing theorem

> The sampling distribution of the **sample mean** approaches a Normal distribution as sample size grows, *regardless* of the population's underlying distribution.

Let $\bar{X}_n$ = mean of a random sample of size $n$. Then:

$$\bar{X}_n \xrightarrow{d} N\!\left(\mu, \frac{\sigma^2}{n}\right)$$

- $\mu$ = population mean
- $\sigma$ = population std dev
- $\sigma / \sqrt{n}$ = **standard error of the mean**

### Why this is incredible
- The original data can be Poisson, log-normal, weird bimodal — doesn't matter
- The *averages* of large-enough samples are Normal
- We can use Normal-based formulas (Z-tables, confidence intervals, Z-tests) on almost anything

### Rule of thumb for "large enough n"
- $n \ge 30$ — usually fine
- Strongly skewed data → larger n needed (50+)
- Bimodal / weird → may need 100+

### Demonstration
```python
import numpy as np
import matplotlib.pyplot as plt

# population: very skewed exponential
pop = np.random.exponential(scale=1, size=100_000)

# 1000 samples, each of size 50
sample_means = [np.mean(np.random.choice(pop, size=50)) for _ in range(1000)]

plt.hist(sample_means, bins=40)
# Looks Normal — even though source is highly skewed.
```

---

## 4. Sampling distribution

The distribution of a sample statistic (e.g., sample mean) across all possible samples of a given size.

### Key properties of the sampling distribution of the mean
- Center: $\mu$ (population mean)
- Spread: $\sigma / \sqrt{n}$ — the **standard error** (SE)

### Why standard error shrinks with $\sqrt{n}$
- Doubling sample size doesn't halve SE — it divides by √2 ≈ 1.41
- To halve SE, you need 4× the data
- This is why "more data" has diminishing returns

---

## 5. Standard Error (SE)

The standard deviation of the sampling distribution.

For the mean:
$$SE = \frac{\sigma}{\sqrt{n}}$$

If $\sigma$ is unknown (almost always), use the sample std dev $s$:
$$SE \approx \frac{s}{\sqrt{n}}$$

### What it tells you
- Smaller SE → more precise estimate
- Used inside confidence intervals, hypothesis tests

```python
import numpy as np
from scipy import stats

sample = np.array([23, 27, 30, 35, 22, 31, 28])
se = sample.std(ddof=1) / np.sqrt(len(sample))
print(se)
# or: stats.sem(sample)
```

---

## 6. Z-table refresher

For Standard Normal Distribution:
- 1.645 → 90% one-sided
- 1.96 → 95% two-sided
- 2.576 → 99% two-sided

```python
from scipy.stats import norm
norm.ppf(0.975)         # 1.96
norm.cdf(1.96)          # 0.975
```

These constants appear in every confidence interval / Z-test formula.

---

## 7. Confidence Interval (CI)

A range of plausible values for a parameter, with a confidence level (typically 95%).

For the population mean (assuming CLT applies):
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{s}{\sqrt{n}}$$

For 95% CI:
$$\bar{x} \pm 1.96 \cdot SE$$

### Common interpretation mistakes (read twice)
- ❌ "95% probability the true mean is in this interval"
- ✅ "If we repeated this sampling process many times, ~95% of the constructed intervals would contain the true mean"

The true mean either is or isn't in the interval — once we've computed it, there's no probability over it. We just trust the *procedure*.

### Example — "Estimate average car miles"
Sample of 50 cars: $\bar{x} = 12,500$ miles, $s = 2,000$ miles.

```python
import numpy as np
mean = 12500
s = 2000
n = 50
se = s / np.sqrt(n)
z = 1.96
ci = (mean - z * se, mean + z * se)
# (~11946, ~13054)
```

We're 95% confident the true average miles per car (across the population) is between ~11,946 and ~13,054 miles.

### When n is small + σ unknown → t-distribution

For n < 30 or for tiny samples in general, use the **t-distribution** instead of Z. It has heavier tails (accounts for more uncertainty when σ is estimated from the sample).

```python
from scipy import stats
ci = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
```

---

## 8. CI shortcut with scipy

```python
import numpy as np
from scipy import stats

sample = np.array([12300, 11800, 13200, 12500, 12900, 11400, 13100])
mean = sample.mean()
se = stats.sem(sample)

# 95% CI using t-distribution (correct when σ unknown)
ci = stats.t.interval(0.95, df=len(sample)-1, loc=mean, scale=se)
print(ci)
```

---

## 9. Solar panels case study (Codebasics example)

Typical setup: a solar panel manufacturer claims average daily output of 5 kWh per panel (population claim). You install 100 panels, measure for a week, get $\bar{x} = 4.7$ kWh, $s = 0.6$ kWh.

```python
import scipy.stats as stats
mean = 4.7
s = 0.6
n = 100
se = s / np.sqrt(n)         # 0.06

ci = (mean - 1.96 * se, mean + 1.96 * se)        # (4.58, 4.82)
```

**The claim of 5 is *outside* the 95% CI.** That's evidence (not yet a hypothesis test) the claim might be false. Hypothesis testing — next chapter — formalizes this.

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Using SE = σ instead of σ/√n | underestimates uncertainty | always divide by √n |
| Using Z-formula on n=10 | tails too narrow | use t-distribution |
| "95% probability the mean is in [a, b]" | classical mis-interpretation | "the procedure works 95% of the time" |
| CLT on highly skewed + small n | sampling distribution still skewed | take larger n or use bootstrap |
| Sampling without randomness | biased estimate, garbage CI | enforce randomization |

## Self-check

- [ ] State the CLT in your own words.
- [ ] Why does standard error scale with $1/\sqrt{n}$?
- [ ] What's the right confidence interval interpretation?
- [ ] When use t-distribution over Z?
- [ ] If you double your sample size, what happens to your CI width?
- [ ] Compute a 95% CI for $\bar{x}=10$, $s=2$, $n=25$. (Use t.)
- [ ] Why do skewed distributions need larger n for CLT to "kick in"?
- [ ] What's stratified sampling and when do you use it?
