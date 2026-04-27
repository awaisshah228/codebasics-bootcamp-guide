# Ensemble 1 — Voting, Bagging, Random Forest

## Lectures covered
- Introduction to Ensemble Learning
- Majority Voting, Average and Weighted Average
- Bagging
- Random Forest

---

## 1. Why ensembles work

A single model has bias + variance + irreducible noise. Combine multiple **diverse** models and:
- Errors partially cancel out
- Variance drops
- Often a small bias-variance trade for net win

Ensembles dominate Kaggle leaderboards and most production tabular ML.

---

## 2. Voting / Averaging — the simplest ensemble

Train multiple different models. Combine their predictions.

### Hard voting (classification)
Each model votes a class; majority wins.

### Soft voting (classification)
Average the predicted probabilities; pick highest.

### Averaging (regression)
Average their predicted numbers.

### Weighted variants
Give better-performing models higher weight.

```python
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

vote = VotingClassifier(
    estimators=[
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier()),
        ("svc", SVC(probability=True)),
    ],
    voting="soft",     # uses .predict_proba
)
vote.fit(X_train, y_train)
```

The diversity matters — combining 5 logistic regressions gives almost the same as one. Combine *different families*.

---

## 3. Bagging — Bootstrap AGGregating

The recipe:
1. Draw N **bootstrap samples** (sample with replacement) from the training set
2. Train one base model on each sample
3. Average / vote their predictions

Bootstrap sampling means each model sees a slightly different dataset → models become diverse.

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag = BaggingClassifier(
    estimator=DecisionTreeClassifier(max_depth=None),
    n_estimators=100,
    bootstrap=True,
    n_jobs=-1,
    random_state=42,
)
bag.fit(X_train, y_train)
```

### Why bagging works
Reduces **variance** without increasing bias. Especially helpful for high-variance models (deep trees).

### Out-of-Bag (OOB) evaluation
About 37% of training samples are not in any given bootstrap. Use them as a "free" validation set.

```python
BaggingClassifier(..., oob_score=True)
print(bag.oob_score_)
```

---

## 4. Random Forest — bagging + feature randomness

Random Forest = Bagging + at each split, a random subset of features is considered.

The two layers of randomness (bootstrap rows + random feature subsets) decorrelate the trees → more variance reduction → better accuracy.

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,         # let trees grow
    min_samples_leaf=1,
    max_features="sqrt",    # √(n_features) per split
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train, y_train)
```

### Key hyperparameters
| Param | What |
|---|---|
| `n_estimators` | # of trees — more = better, slower (diminishing returns) |
| `max_depth` | tree depth — None = unlimited |
| `max_features` | features per split — `"sqrt"` (classification), `1.0` (regression) |
| `min_samples_leaf` | min rows per leaf |
| `class_weight` | for imbalance |
| `n_jobs=-1` | use all CPU cores |
| `oob_score` | use OOB instead of validation set |

### Strengths
- Robust out-of-the-box; minimal tuning needed
- Handles non-linearity, interactions, mixed types
- Feature importance for free
- Parallel training (fast on multi-core)
- Hard to overfit massively

### Weaknesses
- Memory: stores all trees
- Slower than logistic regression at predict time
- Less accurate than gradient boosting on most tabular problems
- Less interpretable than a single tree

---

## 5. Feature importance

```python
import pandas as pd
fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(fi.head(10))
```

For a more honest version (handles correlated features):
```python
from sklearn.inspection import permutation_importance
result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
imp = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)
```

---

## 6. Random Forest as a baseline

For any tabular problem:
1. Logistic Regression / Linear Regression (linear baseline)
2. Random Forest (non-linear baseline, no tuning)
3. XGBoost / LightGBM (top performance)

If RF's score is ≥ logistic's, you have non-linear signal — proceed to gradient boosting.

---

## 7. Real example — credit risk

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=20,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))
```

---

## 8. ExtraTrees — even more random

```python
from sklearn.ensemble import ExtraTreesClassifier
ExtraTreesClassifier(n_estimators=300, n_jobs=-1)
```

Like Random Forest, but split thresholds are also chosen randomly (not optimally). Slightly faster, sometimes slightly better — try both.

---

## 9. Bagging vs Boosting (preview)

| | Bagging / RF | Boosting (next file) |
|---|---|---|
| Trees trained | in parallel | sequentially |
| Each tree fits | bootstrap of data | residuals of previous |
| Goal | reduce variance | reduce bias |
| Speed | fast (parallel) | slower (sequential) |
| Often more accurate? | rarely | usually (XGBoost wins most Kaggle) |

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| `n_estimators=10` | high variance ensemble | use 100–500 |
| Forgetting `n_jobs=-1` | training slow | parallelize |
| Trusting `feature_importances_` blindly | biased toward high-cardinality features | use `permutation_importance` |
| RF with deep trees on tiny data | overfit | use `min_samples_leaf` |
| Reading per-feature impacts as "causal" | trees give associations only | for causal, use causal inference methods |

## Self-check

- [ ] Why do ensembles often beat single models?
- [ ] Difference between hard and soft voting?
- [ ] What does bagging do to bias and variance?
- [ ] What's bootstrap sampling?
- [ ] What does Random Forest do that plain bagging-of-trees doesn't?
- [ ] Walk through `oob_score` — why is it useful?
- [ ] Why is `max_features="sqrt"` a sensible default?
- [ ] When should you reach for ExtraTrees over Random Forest?
