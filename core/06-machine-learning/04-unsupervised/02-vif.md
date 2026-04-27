# Unsupervised 2 — VIF (Variance Inflation Factor)

## Lectures covered
- Variance Inflation Factor (VIF)

---

## 1. The problem — multicollinearity

If two features are highly correlated, the model can't tell which one is "really" responsible for changes in y. Linear regression coefficients become unstable: small data perturbations produce wildly different coefficients.

Effects:
- Coefficient signs flip
- Standard errors balloon
- Interpretation becomes meaningless
- Predictions can still be OK (the model finds a working combo) — but you can't *trust* individual coefficients

---

## 2. What VIF measures

For each feature $x_j$:
1. Fit a regression of $x_j$ on **all other features**
2. Get $R_j^2$
3. $\text{VIF}_j = \frac{1}{1 - R_j^2}$

Interpretation:
- VIF = 1 → feature uncorrelated with others (great)
- VIF = 5 → feature 80% explained by others (concerning)
- VIF > 10 → high multicollinearity (act on it)

---

## 3. Computing VIF in Python

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from statsmodels.tools.tools import add_constant

X_const = add_constant(X)                        # add intercept column
vifs = pd.Series(
    [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])],
    index=X_const.columns,
)
print(vifs.sort_values(ascending=False))
```

`add_constant` is required because VIF is computed in a regression that needs an intercept.

---

## 4. Treatment options

### 1. Drop one of the correlated features
The simplest fix. Often two features are essentially the same (e.g., `price` and `price_in_dollars`).

### 2. Combine them into one feature
E.g., `BMI = weight / height²` collapses two correlated features into one.

### 3. PCA
Project onto principal components — guaranteed orthogonal.

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=0.95)         # keep 95% of variance
X_pca = pca.fit_transform(X_scaled)
```

### 4. Use regularization (Ridge / Lasso)
Doesn't fix interpretability, but stabilizes coefficients and prediction performance.

### 5. Tree models / gradient boosting
Don't really care about multicollinearity — they pick whichever feature splits best.

---

## 5. Iterative VIF reduction

The classic workflow:

```python
def reduce_vif(X, threshold=5):
    X = X.copy()
    while True:
        Xc = add_constant(X)
        vifs = pd.Series(
            [variance_inflation_factor(Xc.values, i) for i in range(Xc.shape[1])],
            index=Xc.columns,
        ).drop("const")
        if vifs.max() <= threshold:
            return X
        worst = vifs.idxmax()
        print(f"Dropping {worst} (VIF={vifs.max():.1f})")
        X = X.drop(columns=[worst])

X_reduced = reduce_vif(X)
```

Drop the highest-VIF feature, recompute, repeat until all are below threshold.

---

## 6. When VIF doesn't matter

- **Tree-based models** — robust to multicollinearity
- **Regularized linear models** (Ridge / Lasso) — stable in the presence of multicollinearity
- **Pure prediction tasks** — if you don't need to interpret coefficients, multicollinearity hurting interpretability isn't a problem

When VIF *does* matter:
- Plain OLS with focus on coefficient interpretation
- Statistical inference (CIs, p-values on coefficients)
- Healthcare / finance / regulated domains where each feature's effect is reported separately

---

## 7. Real example — the credit-risk project

In Codebasics' credit-risk project (Module 6 project 2), VIF analysis is part of the feature-engineering step. Often:
- `outstanding_debt` and `credit_utilization` are highly correlated → drop one
- Multiple credit-history-length features encode similar info → keep one
- Income and net-worth features → consolidate

This produces a cleaner, more interpretable model that satisfies regulators (banking models often need explainability for compliance).

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| VIF on un-standardized data | mostly fine, but check if results match expectation | scale before if comparing magnitudes |
| Forgetting `add_constant` | wrong VIFs | always add intercept |
| Dropping all high-VIF features at once | might drop *all* useful info | drop one at a time, recompute |
| Worrying about VIF for tree models | not a real concern there | only worry for linear models |
| Confusing correlation with VIF | correlation is pairwise, VIF is multivariate | VIF catches "this feature is explained by *combinations* of others" |

## Self-check

- [ ] What does VIF measure?
- [ ] Threshold: when do I act on a VIF value?
- [ ] Why does VIF require `add_constant`?
- [ ] Three ways to fix high VIF.
- [ ] When does VIF not matter for your model choice?
- [ ] How is VIF different from a pairwise correlation matrix?
- [ ] Walk through iterative VIF reduction on a 10-feature dataset.
