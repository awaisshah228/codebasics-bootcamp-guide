# Foundations 7 — Regularization (L1 / L2)

## Lectures covered
- L1 and L2 Regularization

---

## 1. The idea

Add a penalty for **large weights** to the loss function. The model can no longer freely assign huge coefficients to fit noise — it pays a tax.

$$J_{\text{reg}}(\beta) = \underbrace{\frac{1}{n}\sum (y_i - \hat{y}_i)^2}_{\text{fit term}} + \underbrace{\lambda \cdot \text{penalty}(\beta)}_{\text{regularization}}$$

Higher $\lambda$ → simpler model (more shrinkage). Lower $\lambda$ → closer to plain OLS.

---

## 2. L2 — Ridge regression

Penalty: sum of squared weights.

$$J = \text{MSE} + \lambda \sum \beta_j^2$$

- Shrinks coefficients toward 0 but rarely to *exactly* 0
- Reduces variance, increases (a bit of) bias
- Stable on multicollinear features

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0)             # alpha == λ
ridge.fit(X_train, y_train)
```

---

## 3. L1 — Lasso regression

Penalty: sum of absolute weights.

$$J = \text{MSE} + \lambda \sum |\beta_j|$$

- Shrinks AND **zeros out** some coefficients → automatic feature selection
- Useful when you suspect many irrelevant features
- Can be unstable when features are highly correlated (picks one, drops the other)

```python
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

---

## 4. Elastic Net — L1 + L2

$$J = \text{MSE} + \lambda \left(\rho \sum |\beta_j| + (1-\rho) \sum \beta_j^2 \right)$$

- $\rho$ controls the L1 / L2 mix (0 → Ridge, 1 → Lasso)
- Best of both: feature selection + stability

```python
from sklearn.linear_model import ElasticNet
en = ElasticNet(alpha=0.1, l1_ratio=0.5)
```

---

## 5. Why L1 zeros out and L2 shrinks — the geometric intuition

```
L1 (diamond)                          L2 (circle)
                                      
       │                                     │
       ╱╲                                  ─────
      ╱  ╲     <-- coefficient                │
     ╱    ╲        constraint                 │
    ╱      ╲                                  │
                                      
   Sharp corners at axes:           Smooth curve everywhere:
   minimum often hits a corner      minimum lands on the curve
   → β_j = 0                        → β_j is small but non-zero
```

The L1 constraint region has corners where one of the coefficients is exactly 0. The MSE-minimizing point inside that constraint often lands on a corner → exact zero coefficients.

The L2 region is smooth, no corners → coefficients shrink toward but rarely *equal* 0.

---

## 6. Picking $\lambda$ (alpha) — cross-validation

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
import numpy as np

ridge = RidgeCV(alphas=np.logspace(-3, 3, 25))
ridge.fit(X_train, y_train)
print(ridge.alpha_)                  # best alpha by CV
```

Or manually with `GridSearchCV`.

---

## 7. Always scale before regularizing

L1 and L2 penalize coefficients by *magnitude*. If `income` is in millions and `age` in 10s, the algorithm will unfairly penalize income's coefficient.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
```

---

## 8. Real workflow on a regression problem

```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import cross_val_score

X, y = ...   # your features and target

models = {
    "OLS":   Pipeline([("s", StandardScaler()), ("m", LinearRegression())]),
    "Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))]),
    "Lasso": Pipeline([("s", StandardScaler()), ("m", Lasso(alpha=0.1))]),
}

for name, m in models.items():
    score = cross_val_score(m, X, y, cv=5, scoring="r2").mean()
    print(f"{name}: R² = {score:.3f}")
```

If Ridge / Lasso outperform OLS → you had overfitting. If they hurt → you had underfitting.

---

## 9. Regularization in classification

Same idea applies:
```python
from sklearn.linear_model import LogisticRegression
LogisticRegression(penalty="l2", C=1.0)        # C = 1/λ — bigger C = less regularization
LogisticRegression(penalty="l1", solver="liblinear")
LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
```

> Note: `C` is the *inverse* of regularization strength. Smaller C = more regularization. Confusing but consistent across logistic regression, SVM, etc.

---

## 10. Regularization in tree-based / NN models

Different mechanisms, same purpose:

| Family | "Regularization" |
|---|---|
| Decision Tree | `max_depth`, `min_samples_leaf`, `max_leaf_nodes` |
| Random Forest | tree depth + ensemble averaging |
| Gradient Boosting | `learning_rate` (shrinkage), tree depth, early stopping |
| Neural Net | weight decay (L2), dropout, early stopping |

The principle is universal: **constrain the model to prevent it from memorizing the training set.**

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgetting to scale before L1/L2 | unfair penalization across scales | StandardScaler in pipeline |
| Setting alpha way too high | underfit | use CV to find sweet spot |
| Lasso on highly correlated features | picks one arbitrarily | use Elastic Net |
| Comparing alphas across different scales | inconsistent units | log-spaced grid: `np.logspace(-3, 3, 25)` |
| Confusing C with alpha (sklearn classifiers) | wrong direction | C = 1 / λ |

## Self-check

- [ ] Why does L1 zero out coefficients but L2 doesn't?
- [ ] When use Ridge vs Lasso vs Elastic Net?
- [ ] Why must features be scaled before regularization?
- [ ] What does `alpha=0` mean in Ridge / Lasso?
- [ ] What is `C` in LogisticRegression and how does it relate to alpha?
- [ ] How do you pick `alpha` automatically?
- [ ] What's the equivalent of regularization in random forest? In XGBoost? In neural nets?
- [ ] If your training and test errors are both high, will adding more regularization help? Why or why not?
