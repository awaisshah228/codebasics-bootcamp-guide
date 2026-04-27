# Section 5 — Distributions

## Lectures covered
- What Is a Distribution?
- Skewness · Normal Distribution
- Detect Outliers Using Normal Distribution
- Z Score · Standard Normal Distribution (SND)
- Quizzes & Exercises · Chapter Summary

---

## 1. What is a distribution?

A **distribution** describes how the values of a variable are spread.

- **Discrete** distribution: each possible value has a probability mass (PMF). Σ probs = 1.
- **Continuous** distribution: probabilities are densities (PDF). ∫ density = 1; probability of any single point is 0; probability is over *ranges*.

Visually: a histogram (sample) approximates the underlying PDF (population).

---

## 2. Common distributions you'll see in DS

### Bernoulli
One trial, two outcomes (success/failure). $P(X=1) = p$.
- Example: "did the user click? (1 = yes)"

### Binomial
Number of successes in $n$ Bernoulli trials.
- Example: "how many of 100 visitors clicked?"

### Poisson
Number of events in a fixed interval, when events are rare and independent.
- Example: customers arriving per hour; goals per match

### Uniform
All values equally likely between $a$ and $b$.
- Example: random number between 0 and 1

### Normal (Gaussian)
The bell curve. Defined by mean $\mu$ and standard deviation $\sigma$.

### Exponential
Time until a Poisson event.
- Example: time between API requests

### Log-normal
$\log(X)$ is Normal. Heavy right tail.
- Example: incomes, file sizes, latencies

> The bootcamp focuses on Normal because it's the foundation for hypothesis testing — but real-world data is often skewed (log-normal, Poisson, etc.). Knowing the family changes which test you'd use.

---

## 3. Skewness

A measure of asymmetry of a distribution.

```
Right-skewed                  Symmetric                   Left-skewed
    ████▆▄▂▁                    ▁▂▄▆██▆▄▂▁                       ▁▂▄▆████
   skew > 0                    skew ≈ 0                          skew < 0
   mean > median               mean ≈ median                     mean < median
```

| Skewness range | Severity |
|---|---|
| −0.5 to 0.5 | approximately symmetric |
| −1 to −0.5 / 0.5 to 1 | moderately skewed |
| < −1 / > 1 | heavily skewed |

```python
df["income"].skew()
```

### What to do when skewed
- Consider **log-transform** for right-skewed positive data: `np.log1p(x)`
- Use **median + IQR** rather than mean + std dev for summary
- Some ML models (linear regression, neural nets pre-batch-norm) prefer roughly Normal features — transform first

---

## 4. Normal distribution

PDF:
$$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

Defined by:
- $\mu$ — mean (center)
- $\sigma$ — standard deviation (spread)

### The 68–95–99.7 rule (empirical rule)
For Normal data:
- ~68% of values within $\mu \pm 1\sigma$
- ~95% within $\mu \pm 2\sigma$
- ~99.7% within $\mu \pm 3\sigma$

This is the foundation of "3σ outlier" detection.

### Why Normal shows up everywhere
Two reasons:
1. **Many natural quantities** are sums of small effects → CLT (next chapter) → Normal
2. **Statistical tests** assume Normal residuals → so we *try* to engineer Normal features

---

## 5. Detect outliers using Normal distribution (3σ rule)

If your data is approximately Normal:
```python
mean = df["age"].mean()
std = df["age"].std()
mask = (df["age"] < mean - 3 * std) | (df["age"] > mean + 3 * std)
outliers = df[mask]
```

### When 3σ rule is appropriate
- Data is approximately Normal (check histogram + skewness)
- No bimodality
- Reasonably large sample (≥30 ideally)

### When 3σ rule fails
- Heavy-tailed data (incomes, latencies) — IQR is better
- Bimodal data — split first
- Very small samples (<10) — neither method is reliable

---

## 6. Z-score (standardization)

The Z-score of a value $x$:
$$z = \frac{x - \mu}{\sigma}$$

It says: *how many standard deviations is x from the mean?*

```python
from scipy import stats
df["age_z"] = stats.zscore(df["age"])
# alternatively:
df["age_z"] = (df["age"] - df["age"].mean()) / df["age"].std()

# typical outlier filter
df_clean = df[df["age_z"].abs() < 3]
```

### Why standardize
- Make features on the **same scale** for distance-based ML (k-NN, k-means, SVM)
- Compute "how unusual is this value?" interpretably
- Bring features into roughly Normal centered at 0

### sklearn equivalent
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)        # mean 0, std 1
```

---

## 7. Standard Normal Distribution (SND)

The Normal distribution with $\mu = 0$ and $\sigma = 1$.

If $X \sim N(\mu, \sigma^2)$, then $Z = (X - \mu)/\sigma \sim N(0, 1)$.

That's why we standardize: to use universal Z-tables / Z-tests regardless of original units.

### Reading a Z-table (for the curious)
- Find the row matching the integer + first decimal of Z
- Find the column matching the second decimal
- Cell gives $P(Z \le z)$, the cumulative probability up to that z

Modern code:
```python
from scipy.stats import norm
norm.cdf(1.96)         # 0.975  (95% one-sided)
norm.ppf(0.975)        # 1.96   (the inverse)
```

### Why 1.96 is famous
- 95% of values fall within ±1.96 std devs of the mean in a Normal distribution
- → most "95% confidence intervals" are mean ± 1.96 × SE

---

## 8. Distribution-checking workflow on real data

```python
import seaborn as sns
import scipy.stats as stats

# 1. Histogram + KDE
sns.histplot(df["age"], kde=True)

# 2. Skewness + kurtosis
print("skew:", df["age"].skew(), "kurtosis:", df["age"].kurtosis())

# 3. Q-Q plot — visual normality check
stats.probplot(df["age"], dist="norm", plot=plt)

# 4. Statistical normality test (only meaningful for n < 5000)
stat, p = stats.shapiro(df["age"].dropna().sample(min(5000, len(df))))
print(f"Shapiro p-value: {p:.4f}")    # p > 0.05 → can't reject normality
```

> **Don't blindly trust normality tests** on huge samples — they detect tiny deviations as "non-normal" even when the data is normal *enough* for downstream methods.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Assuming any bell-ish histogram is Normal | wrong test choice | check Q-Q + skew |
| Using 3σ on heavy-tailed data | drops too many real points | use IQR or domain rule |
| Standardizing with stats from train + test combined | data leakage | fit `StandardScaler` on train only |
| Confusing PMF and PDF | one's discrete, one continuous | discrete: prob *of* a value; continuous: density *at* a value |
| Forgetting Z-score is unitless | mixing with raw values | always scale to z, then back |

## Self-check

- [ ] State the 68-95-99.7 rule.
- [ ] What's the Z-score and what does it mean intuitively?
- [ ] Why does standardization help k-NN but not tree-based models?
- [ ] When use IQR-based outlier detection vs 3σ?
- [ ] State the difference between PDF and PMF.
- [ ] Why is the Normal distribution so common (preview of CLT)?
- [ ] If a value has Z = 2.5, is it an outlier? Justify.
- [ ] What's a Q-Q plot used for?
