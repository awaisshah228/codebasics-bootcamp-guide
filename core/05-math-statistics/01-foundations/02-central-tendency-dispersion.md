# Section 3 — Measures of Central Tendency and Dispersion

## Lectures covered
- Descriptive vs. Inferential Statistics
- Measures of Central Tendency: Mean, Median, Mode · Percentile · Shoe Sales analysis
- Measures of Dispersion: Range, IQR · Box plot · Outlier treatment using IQR + Box plot
- Measures of Dispersion: Variance, Std Dev · Stock Returns volatility analysis
- Correlation · Correlation vs Causation
- Quizzes & Exercises · Chapter Summary

---

## 1. Descriptive vs Inferential statistics

| | Descriptive | Inferential |
|---|---|---|
| **What** | summarize *the sample you have* | infer about *a larger population from a sample* |
| **Tools** | mean, median, std dev, charts | hypothesis tests, confidence intervals |
| **Example** | "the mean salary in this dataset is $52k" | "I'm 95% confident the population mean is between $48k and $56k" |

This chapter (and most of EDA) = descriptive. The CLT/hypothesis-testing chapters = inferential.

---

## 2. Central tendency — "what's a typical value?"

### Mean (arithmetic average)
$$\text{mean} = \frac{1}{n}\sum x_i$$

```python
df["age"].mean()
```

- Influenced strongly by outliers (one billionaire shifts the mean a lot)
- Best on roughly symmetric data with no extreme outliers

### Median (middle value when sorted)
```python
df["age"].median()
```

- Robust to outliers
- Best on skewed data (income, house prices, durations)

### Mode (most frequent value)
```python
df["age"].mode()
```

- Useful for categorical data
- Continuous data may have no clear mode (or many)

### When mean lies — the classic example
> "The average salary at our company is $250k!"
> — true; but the CEO makes $20M and 99 others make $50k. **Median** ($50k) tells the truth.

### Quick rule
- Symmetric data → mean ≈ median ≈ mode → use any
- Skewed data → use **median** (and report mode for context)
- Categorical data → use **mode**

### "Shoe sales" analysis (Codebasics example)
Shoe-size data is often bimodal (men's vs women's distributions). Mean / median can both be misleading; **mode** (most-sold size in each segment) drives stocking decisions.

---

## 3. Percentile

The **k-th percentile** is the value below which k% of the data falls.

- **25th percentile (Q1)** — first quartile
- **50th percentile** — median
- **75th percentile (Q3)** — third quartile
- **90th, 95th, 99th** — common in latency / SLO reporting

```python
df["latency_ms"].quantile(0.95)        # P95 latency
df["age"].describe()                    # auto includes 25/50/75
```

> In production engineering, the "p99" of latency matters more than the average.

---

## 4. Range and IQR

### Range
$$\text{range} = \max - \min$$
Crude. Sensitive to outliers (one bad row blows it up).

### IQR (Interquartile Range)
$$\text{IQR} = Q_3 - Q_1$$
The width of the middle 50% of the data. Robust to outliers.

```python
q1, q3 = df["age"].quantile([0.25, 0.75])
iqr = q3 - q1
```

---

## 5. Box plot

Visualizes Q1, Q3, median, and outlier candidates.

```
        |  whisker (≤ 1.5×IQR below Q1)
   ──┬──
   ┌─┴─┐
   │   │
   │ • │ median
   │   │
   └─┬─┘
   ──┴──
        |  whisker (≤ 1.5×IQR above Q3)

   °  outliers (beyond 1.5×IQR)
```

The "1.5 × IQR" rule is convention; tighter (1.0) or looser (3.0) is sometimes used.

```python
sns.boxplot(data=df, x="city", y="income")
```

Compare distributions across categories at a glance.

---

## 6. Outlier treatment — IQR method

```python
def iqr_bounds(s, k=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

low, high = iqr_bounds(df["income"])
df_clean = df[(df["income"] >= low) & (df["income"] <= high)]
```

### Three options when you find outliers
1. **Drop** them — easy, but loses real data
2. **Cap (Winsorize)** — replace beyond-bound values with the bound
3. **Investigate** — many "outliers" are *real* signals (high-value customers, fraud, etc.)

> Always look at outlier rows before deciding. A typo (`age = 999`) gets dropped; a real billionaire gets kept.

---

## 7. Variance and standard deviation

### Variance
Average squared distance from the mean:

$$\text{Var}(x) = \frac{1}{n}\sum_i (x_i - \bar{x})^2$$

(For a *sample*, use $n - 1$ in the denominator — Bessel's correction.)

### Standard deviation
$$\sigma = \sqrt{\text{Var}(x)}$$

In the original units of the variable. Easier to interpret than variance.

```python
df["age"].std()           # sample std dev (n-1)
df["age"].std(ddof=0)     # population std dev (n)
df["age"].var()
```

### Quick interpretation
- **Low std dev**: values cluster near the mean
- **High std dev**: values spread out

### "Stock returns volatility" (Codebasics example)
Two stocks both average +10%/year. Stock A's daily returns std dev is 0.5%; Stock B's is 4%. Stock B is much riskier despite the same average. Volatility = std dev of returns. This is exactly how risk is measured in finance.

---

## 8. Correlation

### Pearson correlation
Measures the **linear** relationship between two numeric variables.

$$r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$$

Range: −1 (perfect negative) to +1 (perfect positive). 0 = no linear relationship.

```python
df[["age", "income"]].corr()
df.corr(numeric_only=True)
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
```

### Spearman correlation
Same idea, but on **ranks**. Captures any monotonic relationship (linear or not). Robust to outliers.

```python
df.corr(method="spearman")
```

### Rules of thumb (Pearson)
| |r| | Interpretation |
|---|---|
| 0.0–0.2 | very weak |
| 0.2–0.4 | weak |
| 0.4–0.6 | moderate |
| 0.6–0.8 | strong |
| 0.8–1.0 | very strong |

These are domain-dependent — in physics 0.95 is mediocre; in psychology 0.4 is great.

---

## 9. Correlation ≠ causation

| Statement | OK? |
|---|---|
| "Ice cream sales correlate with drowning deaths." | ✅ true |
| "Eating ice cream causes drowning." | ❌ no — both caused by *summer* (lurking variable) |

### Why this matters
Almost every "data finding" needs to ask: **could a third variable explain this?**

### How to (start to) move toward causation
- Randomized controlled trial / A/B test (gold standard)
- Difference-in-differences
- Instrumental variables
- Causal inference frameworks (DoWhy, EconML)

**The bootcamp's later A/B testing chapter is the closest practical tool here.**

---

## 10. Concrete pandas reference

```python
df.describe()                            # mean, std, min, percentiles, max
df["age"].agg(["mean", "median", "std", "var", "min", "max"])
df["age"].quantile([0.25, 0.5, 0.75])
df.corr(numeric_only=True)
df.cov()
```

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Reporting only mean for skewed data | misleading | also report median |
| Treating all outliers as errors | loses real data | investigate first |
| Reading correlation as causation | bad decisions | always consider lurking variables |
| Using `.var()` without checking `ddof` | sample vs population mismatch | be explicit |
| Heatmap colors not centered at 0 | hides positive vs negative | use `cmap="coolwarm", center=0` |

## Self-check

- [ ] When does mean lie and median tell the truth?
- [ ] What's IQR and how does it define outliers?
- [ ] Difference between variance and standard deviation — which is in the variable's units?
- [ ] Why use Spearman over Pearson?
- [ ] Walk through interpreting a correlation matrix on a 5-variable dataset.
- [ ] Give 2 examples of correlation that's NOT causation.
- [ ] How would you outlier-treat a `salary` column with both real high earners and obvious typos?
