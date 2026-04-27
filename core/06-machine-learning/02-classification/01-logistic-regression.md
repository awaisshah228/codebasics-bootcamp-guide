# Classification 1 — Logistic Regression

## Lectures covered
- Introduction to Classification
- Logistic Regression: Binary Classification
- Logistic Regression: Multiclass Classification
- Cost Function: Log Loss

---

## 1. Why not linear regression for classification

If you fit a line to predict 0/1, predictions can be < 0 or > 1 — not probabilities. Decision boundary moves around with extreme x's. Linear regression isn't designed for it.

We need a function that maps any real number into [0, 1].

---

## 2. The sigmoid function

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- $z \to -\infty$: σ → 0
- $z = 0$: σ = 0.5
- $z \to +\infty$: σ → 1

```
         1 ┤    ────────
           │  ─/
       0.5 ┤ /
           │/
         0 ┤────────
           └──────────►
                  z
```

---

## 3. Logistic regression — the model

$$P(y = 1 \mid x) = \sigma(\beta_0 + \beta_1 x_1 + \dots + \beta_p x_p)$$

A linear combination passed through the sigmoid. Decision rule: classify as 1 if probability > 0.5 (default threshold).

### sklearn shape
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)              # 0/1
y_prob = model.predict_proba(X_test)[:, 1]  # P(y=1)
```

---

## 4. Log Loss (the cost function for classification)

For binary:
$$L = -\frac{1}{n}\sum_i \left[ y_i \log p_i + (1 - y_i) \log(1 - p_i) \right]$$

- When y = 1, you want p close to 1 → -log(p) small
- When y = 0, you want p close to 0 → -log(1-p) small
- Confidently wrong predictions are heavily penalized (log goes to -∞)

This is what `LogisticRegression` minimizes via gradient descent.

```python
from sklearn.metrics import log_loss
log_loss(y_test, y_prob)
```

---

## 5. Decision threshold — not always 0.5

Default threshold is 0.5. You can shift it for cost-sensitive problems:

```python
threshold = 0.3
y_pred_adjusted = (y_prob > threshold).astype(int)
```

- Lower threshold → more positives flagged → higher recall, lower precision
- Higher threshold → fewer positives flagged → higher precision, lower recall

For fraud detection with high cost of missing a fraud: lower threshold.
For spam where false-positives annoy users: higher threshold.

---

## 6. Multiclass — two strategies

### One-vs-Rest (OvR / OvA)
Train k binary classifiers, each "class i vs everyone else."
```python
LogisticRegression(multi_class="ovr")
```

### Softmax (multinomial logistic regression)
Replace sigmoid with softmax over k logits:
$$P(y = c \mid x) = \frac{e^{z_c}}{\sum_k e^{z_k}}$$

Loss becomes **categorical cross-entropy**.

```python
LogisticRegression(multi_class="multinomial", solver="lbfgs")
```

> Modern sklearn defaults to multinomial. Use it unless OvR is specifically wanted.

---

## 7. Interpreting coefficients (odds ratios)

For a feature $x_j$ with coefficient $\beta_j$:
- $e^{\beta_j}$ = odds ratio for a 1-unit increase in $x_j$
- $e^{0.3} \approx 1.35$ → 1-unit increase multiplies odds by 1.35

```python
import numpy as np
odds_ratios = np.exp(model.coef_[0])
for name, or_ in zip(X.columns, odds_ratios):
    print(name, or_)
```

This makes logistic regression great for **interpretable** classification (think credit risk, healthcare).

---

## 8. Regularization (recap)

```python
LogisticRegression(penalty="l2", C=1.0)        # default
LogisticRegression(penalty="l1", solver="liblinear")
LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=0.5)
```

`C = 1/λ` — bigger C = less regularization. Always scale features first.

---

## 9. Real example — credit default

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("lr", LogisticRegression(class_weight="balanced", C=1.0)),
])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
y_prob = pipe.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
```

`class_weight="balanced"` automatically up-weights the minority class — crucial for imbalanced problems like default prediction.

---

## 10. When logistic regression shines

- Need interpretable coefficients
- Need calibrated probabilities (logistic gives them out of the box)
- Linear-ish decision boundary
- High-dimensional sparse data (e.g., text features after TF-IDF)
- A baseline before trying complex models

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to scale | slower convergence | scale via pipeline |
| Imbalanced classes ignored | predicts majority class always | `class_weight="balanced"` or threshold tuning |
| Threshold 0.5 on imbalanced | recall on minority is 0 | lower threshold + use AUC |
| One-hot encoded everything blindly | huge dimensionality | use OneHotEncoder with min frequency |

## Self-check

- [ ] Write the sigmoid formula.
- [ ] Why is squared error wrong for classification?
- [ ] Walk through log loss intuition for y=1, p=0.9 vs y=1, p=0.1.
- [ ] Difference between OvR and softmax multiclass.
- [ ] Interpret a coefficient of 0.4 in a logistic model.
- [ ] What does `class_weight="balanced"` do?
- [ ] When would you shift the decision threshold below 0.5?
- [ ] Name 3 reasons to choose logistic regression over a random forest.
