# Classification 1 — Logistic Regression

## Lectures covered
- Introduction to Classification
- Logistic Regression: Binary Classification
- Logistic Regression: Multiclass Classification
- Cost Function: Log Loss

---

## In one sentence
**Logistic regression** is linear regression's cousin for yes/no questions — it computes a weighted sum of features then squashes it into a probability between 0 and 1.

## Real-world analogy
A bouncer at a club assigns each guest a "vibe score" by quickly summing things like age, attire, mood (each weighted by experience). Then she runs that score through a quick mental rule: very low score → definitely no; very high → definitely yes; middling → flip a coin (50%). The squashing step is **the sigmoid function**. The weights come from past data — guests she let in who turned out fine, vs. those she shouldn't have.

## The intuition (plain English)
1. Compute `z = β₀ + β₁·age + β₂·income + …` — a regular linear combination.
2. Pass z through the **sigmoid** `σ(z) = 1 / (1 + e^(−z))`. This turns any real number into a probability in (0, 1).
3. Predict class 1 if probability ≥ 0.5, else class 0. Adjust the threshold for cost-sensitive problems (fraud, cancer).
4. Train by minimizing **log loss** — the cost that punishes confident wrong predictions hardest.

Despite the name "regression," it's a **classifier**. The "regression" refers to the linear `z` step.

## Mini worked example — predicting customer churn

You have a fitted model: `z = −3 + 0.05·tenure_days − 0.6·support_calls`. For three customers:

```
                         tenure   calls    z              σ(z) = P(churn)   predict
Anna  (loyal)              900      1     -3 + 45 - 0.6 = 41.4         ≈ 1.0           churn? YES
Ben   (new, no issues)      30      0     -3 + 1.5 - 0  = -1.5         ≈ 0.18          churn? NO
Cara  (frustrated)         180      8     -3 + 9 - 4.8  =  1.2         ≈ 0.77          churn? YES
```

Wait — Anna has tenure 900 and *will* churn? Sign of β₁ is positive, so longer tenure = more likely to churn in this fictional fit. Coefficients tell you directionality once you trust the model — and you'd verify against domain knowledge.

How log loss penalizes:
- Anna: actual = 1, predicted prob = 1.0 → log loss ≈ 0 (great)
- Cara: actual = 1, predicted prob = 0.77 → log loss = −log(0.77) ≈ 0.26 (mild)
- A confidently-wrong prediction (actual=1, predicted=0.05) → −log(0.05) ≈ 3.0 (heavy penalty)

That asymmetry is why log loss is the right cost for classification.

## At-a-glance — the model

```
                      sigmoid               threshold
features ─► Σ(β·x) ─► σ(z) ─► probability ─► 0 or 1
   x          z       1/(1+e⁻ᶻ)              (default 0.5)

Output of sigmoid:
            1 ─┤              ───
              │           ──/
            0.5┤        ──/
              │     ──/
            0 ─┤────/
              └───────────────► z
              -inf       0       +inf
```

## Why this matters
- **The most-used classifier in industry.** Banks, hospitals, marketing — anywhere you need an interpretable yes/no model.
- **Outputs probabilities, not just labels.** Lets you set business-driven thresholds and rank cases.
- **Coefficients are interpretable** as log-odds — `e^β` is the odds ratio. Compliance teams love this.
- **Foundation of neural networks.** A single neuron with sigmoid activation *is* logistic regression.
- **Credit-risk project** uses it as the regulator-friendly baseline.

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

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Classification** | Predicting a category (spam/not, default/not, malignant/benign) |
| **Binary classification** | Just two classes |
| **Multiclass classification** | Three or more mutually exclusive classes |
| **Logistic regression** | Linear regression's coefficients passed through a sigmoid → probability |
| **Sigmoid function** | `σ(z) = 1/(1 + e⁻ᶻ)` — squashes any real number into (0, 1) |
| **Logit** | The reverse: `z = log(p / (1−p))` — the linear combination before squashing |
| **Probability output** | What `predict_proba` returns — a number in (0, 1) per class |
| **Decision threshold** | Cutoff for converting probability to a class label (default 0.5) |
| **Log loss / cross-entropy** | The classification cost function — punishes confident wrong predictions heavily |
| **Odds** | `p / (1 − p)` — "for every loss, how many wins?" |
| **Odds ratio** | `e^β` — how odds are multiplied per 1-unit increase in feature |
| **Softmax** | Multiclass generalization of sigmoid — turns k logits into k probabilities summing to 1 |
| **One-vs-Rest (OvR / OvA)** | Multiclass strategy: train k binary "this class vs everyone else" models |
| **Multinomial / Softmax regression** | The "real" multiclass logistic regression — single model, k outputs |
| **Class weight** | Multiplier on each class's loss term — used for imbalanced classes |
| **`class_weight="balanced"`** | sklearn auto-sets weights inversely proportional to class frequencies |
| **Stratified split** | Train/test split that preserves class proportions — use `stratify=y` |
| **`predict()`** | Returns the predicted class label (after applying threshold) |
| **`predict_proba()`** | Returns the predicted probability for each class |
| **Calibration** | How well predicted probabilities match real-world frequencies |
| **C (in sklearn)** | Inverse of regularization strength: `C = 1/λ` — bigger C = less regularization |
| **L1 penalty** | Lasso-style regularization — produces sparse coefficients |
| **L2 penalty** | Ridge-style regularization — shrinks all coefficients smoothly |
| **`solver`** | Optimization algorithm sklearn uses (`lbfgs`, `liblinear`, `saga`) — different solvers support different penalties |
| **TF-IDF** | Term-frequency / inverse document frequency — turns text into sparse numeric features that logistic regression handles well |

## Further reading
- Previous: [../01-foundations/07-regularization.md](../01-foundations/07-regularization.md) — same L1/L2 ideas applied here
- Next: [02-classification-metrics.md](02-classification-metrics.md) — judging classifiers
- Class imbalance handling: [06-class-imbalance.md](06-class-imbalance.md)
- Threshold tuning: [07-roc-auc.md](07-roc-auc.md)
- Credit-risk project: [../06-projects/02-credit-risk-classification.md](../06-projects/02-credit-risk-classification.md)
- Math foundation — odds & log-odds: [../../05-math-statistics/01-foundations/04-distributions.md](../../05-math-statistics/01-foundations/04-distributions.md)
