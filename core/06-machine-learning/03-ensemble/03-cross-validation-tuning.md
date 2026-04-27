# Ensemble 3 — Cross Validation & Hyperparameter Tuning

## Lectures covered
- K-Fold Cross Validation
- Stratified K-Fold Cross Validation
- Hyperparameter Tuning

---

## 1. Why cross-validation

A single train/test split has variance — your one test number could be lucky/unlucky. Cross-validation averages over multiple splits → **more stable, more honest** performance estimate.

---

## 2. K-Fold

Split the data into k equal parts (folds). For each fold:
- Hold it out as validation
- Train on the other k−1
- Score on the held-out fold

Average the k scores.

```
Fold 1 — used for validation, rest for training
Fold 2 — used for validation, rest for training
...
Fold k — used for validation, rest for training
```

```python
from sklearn.model_selection import KFold, cross_val_score

kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring="r2")
print(scores.mean(), "±", scores.std())
```

Common k:
- **5** — default, good balance
- **10** — when data is plentiful
- **3** — when each fit is expensive

---

## 3. Stratified K-Fold (for classification)

Plain k-fold can produce folds with skewed class distributions. **Stratified** ensures each fold has the same class proportions.

```python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cross_val_score(model, X, y, cv=skf, scoring="f1")
```

For classification, **always** stratify. `cross_val_score` does this automatically when you pass an integer `cv=5` and the estimator is a classifier — but explicit is clearer.

---

## 4. Time-aware CV (when data has order)

For time-series, plain k-fold leaks the future into training. Use **TimeSeriesSplit**:

```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

Each fold trains on data up to time t, validates on t+1.

---

## 5. Group K-Fold (when rows aren't independent)

If you have multiple samples per user/customer, plain k-fold may put the same user in train *and* validation → leakage.

```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
gkf.split(X, y, groups=user_ids)
```

---

## 6. Putting it together

```python
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, f1_score

results = cross_validate(
    pipeline, X, y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring={"f1": make_scorer(f1_score), "auc": "roc_auc"},
    return_train_score=True,
    n_jobs=-1,
)

print("train F1:", results["train_f1"].mean())
print("test  F1:", results["test_f1"].mean())
print("train AUC:", results["train_auc"].mean())
print("test  AUC:", results["test_auc"].mean())
```

Train vs test gap → tells you if you're overfitting.

---

## 7. Hyperparameter tuning approaches

### Grid Search — exhaustive
Try every combination of given values.

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [3, 5, 7, None],
    "min_samples_leaf": [1, 5, 20],
    "n_estimators": [100, 300, 500],
}
gs = GridSearchCV(rf, param_grid, cv=5, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)
print(gs.best_params_, gs.best_score_)
```

Pros: thorough.
Cons: explodes — 4 × 3 × 3 = 36 fits × 5 folds = 180 models.

### Random Search — sample at random
Faster; usually finds nearly-optimal configs much sooner than grid search.

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    "max_depth": randint(3, 15),
    "min_samples_leaf": randint(1, 50),
    "learning_rate": uniform(0.01, 0.2),
}
rs = RandomizedSearchCV(model, param_dist, n_iter=50, cv=5, scoring="f1", random_state=42, n_jobs=-1)
rs.fit(X_train, y_train)
```

### Bayesian / Smart Optimizers — Optuna
The modern Kaggle / production default. Builds a model of the loss surface to pick promising configs.

```bash
pip install optuna
```
```python
import optuna

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = xgb.XGBClassifier(**params, random_state=42)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc").mean()
    return score

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
print(study.best_params)
```

Optuna typically finds great configs in 50–100 trials where grid search would take 10,000+.

---

## 8. Nested cross-validation — when you need a clean number

If you tune hyperparameters with CV, your "best CV score" is biased upward (selected the best, which favors lucky folds). Nested CV gives an honest estimate.

```python
from sklearn.model_selection import cross_val_score, GridSearchCV

inner = GridSearchCV(model, param_grid, cv=3)
outer_scores = cross_val_score(inner, X, y, cv=5, scoring="f1")
print(outer_scores.mean())   # honest estimate of tuned model's performance
```

Slow but the gold standard for paper-quality numbers.

---

## 9. Pipeline + GridSearchCV (the right way)

Always wrap preprocessing in a pipeline, then tune the whole thing:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("clf",   LogisticRegression()),
])

param_grid = {
    "clf__C": [0.01, 0.1, 1, 10, 100],
    "clf__penalty": ["l1", "l2"],
    "clf__solver": ["liblinear"],
}

gs = GridSearchCV(pipe, param_grid, cv=5, scoring="f1", n_jobs=-1)
gs.fit(X_train, y_train)
```

Note the `clf__C` syntax — `<step>__<param>` lets you tune any step.

---

## 10. Tuning strategy that actually works

1. **Coarse random search** over a wide space (50 trials)
2. **Fine grid search** around the best random config
3. **Optuna** if you have time/budget for 100+ trials and want the last 1–2%

Don't grid-search 5 hyperparameters with 5 values each on day one — that's 3,125 fits.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| K-fold without shuffle on ordered data | leakage / weird folds | `shuffle=True` |
| Tuning with the test set | inflated final number | hold out test, tune on val/CV |
| Reporting tuned CV score as "test" | optimistic | use nested CV or held-out test |
| Comparing models with different CV setups | apples to oranges | same CV splitter for all |
| Overlooked group structure | leakage | use GroupKFold |
| Tuning n_estimators by grid | wasteful | use early stopping |

## Self-check

- [ ] Why is k-fold better than a single train/test split?
- [ ] When use stratified k-fold vs plain k-fold?
- [ ] Why is plain k-fold wrong for time-series?
- [ ] What is GroupKFold for?
- [ ] Difference between grid search and random search?
- [ ] When use Optuna over RandomizedSearchCV?
- [ ] What's nested CV and when do you need it?
- [ ] Walk through tuning a pipeline `[StandardScaler, LogisticRegression]`.
