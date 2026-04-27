# Section 9 — Hypothesis Testing

## Lectures covered
- Null vs Alternate Hypothesis
- Z Test, Rejection Region · p-Value
- Statistical Power & Effect Size
- A/B Testing · A/B Testing Using Z Test
- T-test
- Chi-squared Distribution · Chi-squared Test of Goodness of Fit · Chi-squared Test of Independence

---

## 1. The framework — what hypothesis testing actually does

You have a claim ("the new ad performs better than the old"). You collect data. The data shows *some* difference. **Is the difference real, or could it have happened by chance?**

Hypothesis testing turns "is this real?" into a quantified probability — the **p-value** — and a decision rule.

---

## 2. H₀ and H₁

- **H₀** (null hypothesis): the boring default. "There's no effect / no difference."
- **H₁** (alternative hypothesis): what you'd like to prove. "There IS an effect."

You either **reject H₀** or **fail to reject H₀**. You never "accept" H₀ — absence of evidence ≠ evidence of absence.

### Examples

| Question | H₀ | H₁ |
|---|---|---|
| New drug works better than placebo? | drug = placebo | drug > placebo |
| New page CTR = old page CTR? | CTR_new = CTR_old | CTR_new ≠ CTR_old |
| Coin is fair? | p_heads = 0.5 | p_heads ≠ 0.5 |

### One-sided vs two-sided
- **Two-sided**: "different" (in either direction)
- **One-sided**: "better than" or "less than" (direction specified upfront)

Default to two-sided unless you have a strong directional reason. Don't pick one-sided after looking at the data.

---

## 3. Type I and Type II errors

|  | H₀ true | H₀ false |
|---|---|---|
| **Reject H₀** | Type I (false positive) | ✅ correct |
| **Fail to reject** | ✅ correct | Type II (false negative) |

- α (alpha) = P(Type I) — typically 0.05 (5%)
- β (beta) = P(Type II) — depends on effect size + sample size
- **Power** = 1 − β = P(detecting a real effect)

### α and the threshold
α = 0.05 means: "I'm willing to tolerate a 5% chance of falsely flagging a non-effect as an effect."

### Power
Targeted power: 80% (industry standard) or 90%. Higher power needs larger sample.

---

## 4. p-value

The probability of observing data **as extreme as ours, assuming H₀ is true**.

- Small p (< 0.05): "if H₀ were true, our data would be weird → reject H₀"
- Large p (≥ 0.05): "our data is consistent with H₀ → fail to reject"

### Common misinterpretations (every one of these is WRONG)
- ❌ "p = 0.03 means there's a 3% chance H₀ is true"
- ❌ "p = 0.03 means the effect is small / large"
- ❌ "p < 0.05 means the result is important"
- ❌ "p > 0.05 means H₀ is true"

### Right interpretation
> "Assuming H₀ is true, the probability of seeing data this extreme is p."

p-value answers a *very specific* question. It does NOT measure the magnitude of effect (use **effect size**) nor the *probability* of H₁.

---

## 5. Z-test (large samples, σ known or large n)

Test statistic:
$$Z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

Where:
- $\bar{x}$ = sample mean
- $\mu_0$ = the H₀ value
- $\sigma / \sqrt{n}$ = standard error

### Decision
Compare |Z| to critical value (e.g., 1.96 for two-sided α = 0.05). Or compute p-value from Z.

```python
from scipy import stats
z = (sample_mean - mu0) / (sigma / np.sqrt(n))
p = 2 * (1 - stats.norm.cdf(abs(z)))         # two-sided
# decide: if p < 0.05, reject H₀
```

### Solar panel example (continuing from CLT chapter)
$\bar{x} = 4.7$, $s = 0.6$, $n = 100$, claim $\mu_0 = 5$.

```python
z = (4.7 - 5) / (0.6 / np.sqrt(100))
# z = -5.0
# |z| = 5.0 → way beyond 1.96 → reject H₀
# p-value ≈ 5e-7 (essentially 0)
```

The manufacturer's 5 kWh claim is statistically rejected.

---

## 6. T-test

Same shape as Z-test, but uses t-distribution. Use when:
- Sample size is small (< 30) **and**
- Population σ is unknown (almost always)

### One-sample t-test
"Does my sample mean differ from a known value?"
```python
from scipy import stats
t_stat, p = stats.ttest_1samp(sample, popmean=5)
```

### Two-sample t-test (independent samples)
"Do these two groups have different means?"
```python
t_stat, p = stats.ttest_ind(group_a, group_b, equal_var=False)   # Welch's t-test
```

### Paired t-test
"Did the same subjects change?" (before/after)
```python
t_stat, p = stats.ttest_rel(before, after)
```

### Assumptions
- Approximately Normal samples (CLT helps for n ≥ 30)
- For independent two-sample: ideally similar variance (Welch's relaxes this)
- Paired version requires: paired observations, differences approximately Normal

---

## 7. Chi-squared tests

For **categorical** data.

### Test of goodness-of-fit
"Does my categorical data match a hypothesized distribution?"

Example: A die rolled 60 times. Expected 10 of each face. Observed: [12, 9, 11, 8, 13, 7]. Is the die fair?

```python
from scipy.stats import chisquare
observed = [12, 9, 11, 8, 13, 7]
expected = [10] * 6
stat, p = chisquare(observed, expected)
# p-value tells us if observed differs from expected significantly
```

### Test of independence
"Are two categorical variables related?"

Example: Is hire decision (hire/no-hire) independent of gender? Build a contingency table.

```python
import numpy as np
from scipy.stats import chi2_contingency

table = np.array([[40, 60],          # hired (men, women)
                  [10, 30]])         # not hired

chi2, p, dof, expected = chi2_contingency(table)
```

H₀: independent. p < 0.05 → reject; the variables are related.

### Assumptions
- Each expected count ≥ 5 (else use Fisher's exact)
- Independent observations

---

## 8. Effect size — what p-value won't tell you

p-value tells you "is there an effect?". **Effect size** tells you "how big is it?". Both matter.

### Cohen's d (for two-group means)
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$$

| |d| | Interpretation |
|---|---|
| 0.2 | small |
| 0.5 | medium |
| 0.8 | large |

```python
import numpy as np
def cohens_d(a, b):
    pooled = np.sqrt((a.std(ddof=1)**2 + b.std(ddof=1)**2) / 2)
    return (a.mean() - b.mean()) / pooled
```

### Why this matters
With huge samples (millions of users), tiny differences become "statistically significant" (p < 0.05) but are **practically meaningless**. Always pair p with effect size.

---

## 9. A/B testing — putting it together

### Setup
- Randomly split users into A (control) and B (treatment)
- Both see the version, traffic identical otherwise
- Measure the outcome (CTR, conversion, revenue per user)

### Pre-launch checklist
- [ ] Define metric upfront (don't peek and decide later)
- [ ] Pre-register direction (one-sided vs two-sided)
- [ ] Pick α (usually 0.05) and power (usually 0.80)
- [ ] Compute required sample size

### Sample size calculation
For two-proportion z-test:
```python
import statsmodels.stats.power as smp

# we want to detect 2pp lift on a 10% baseline; α=.05, power=.80
n = smp.zt_ind_solve_power(effect_size=0.06,    # standardized effect (small lift)
                           alpha=0.05,
                           power=0.80,
                           ratio=1.0,
                           alternative="two-sided")
```

For two-sample t-test:
```python
n = smp.tt_ind_solve_power(effect_size=0.5,      # Cohen's d = 0.5 (medium)
                           alpha=0.05,
                           power=0.80)
```

> Don't run an A/B test without first computing required n. Otherwise you risk under-powered tests (waste time) or over-powered tests (catch trivial effects).

### A/B test using Z-test for proportions
H₀: p_a = p_b. H₁: p_a ≠ p_b.

```python
import numpy as np
from scipy.stats import norm

n_a, k_a = 5000, 510    # control: 5000 users, 510 conversions
n_b, k_b = 5000, 600    # treatment: 5000 users, 600 conversions

p_a = k_a / n_a
p_b = k_b / n_b
p_pooled = (k_a + k_b) / (n_a + n_b)
se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n_a + 1/n_b))

z = (p_b - p_a) / se
p_value = 2 * (1 - norm.cdf(abs(z)))
print(f"lift: {p_b-p_a:.4f}, z={z:.2f}, p={p_value:.4f}")
```

### Common A/B testing mistakes
- **Peeking**: stopping the test early when p < 0.05 → inflates Type I error
- **Multiple metrics without correction**: 20 metrics × α=0.05 → expect 1 false positive
- **Selection effects**: treatment differs from control in ways unrelated to the change
- **Novelty effect**: short tests catch the "new shiny" lift that fades

---

## 10. Translating to AtliQo Bank Phase 2

The Phase-2 goal is to validate the chosen target segment via a controlled experiment:
- A/B: Variant A = standard offer; Variant B = tailored credit-card pitch to chosen segment
- Metric: application rate (or, lower-funnel: approval + activation)
- Sample size: power-analyzed up front
- Test: two-proportion Z-test
- Decision rule: H₀ rejected at α=0.05 + meaningful effect size (e.g., ≥ 1pp lift)

This is the bridge from this chapter to a real-world deliverable.

---

## 11. Common pitfalls (the cheat sheet)

| Mistake | Why it's wrong |
|---|---|
| Reading p < 0.05 as "result is true" | only "data unlikely under H₀" |
| Reading p > 0.05 as "no effect exists" | "we lacked evidence" |
| Multiple comparisons without correction | inflated Type I rate |
| Peeking + stopping early | also inflated Type I |
| Pure p-value reporting | always pair with effect size |
| Using Z when σ unknown + n small | use t |
| Chi-squared with cells < 5 | use Fisher's exact |
| Choosing one-sided after data | post-hoc cherry-pick |

## Self-check

- [ ] State H₀ and H₁ for "the new email subject line increases CTR."
- [ ] Define Type I and Type II errors.
- [ ] What does p = 0.04 actually mean?
- [ ] When use Z-test vs t-test?
- [ ] When use chi-squared goodness-of-fit vs chi-squared independence?
- [ ] What's Cohen's d and why is it important?
- [ ] Walk through the steps of designing an A/B test.
- [ ] Why is "peeking" at A/B test results a problem?
- [ ] Compute required n for an A/B test with baseline 10% conversion, MDE +2pp, 80% power, α 0.05.
- [ ] How does this connect back to AtliQo Phase 2?
