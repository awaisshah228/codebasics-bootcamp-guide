# Model Selection & Tuning — Step-by-Step Guide

> Companion to `ML_AI_ENGINEER_WORKFLOW.md` (Stage 6).
> You have clean data and engineered features. Now: which model do you train,
> how do you train it, and how do you tune it?

> Sister files:
> - [ML_AI_ENGINEER_WORKFLOW.md](ML_AI_ENGINEER_WORKFLOW.md)
> - [DATA_CLEANING_GUIDE.md](DATA_CLEANING_GUIDE.md)
> - [FEATURE_ENGINEERING_GUIDE.md](FEATURE_ENGINEERING_GUIDE.md)
> - [MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md) ← what comes next

---

## The 10-Step Sequence (TL;DR)

```
 1. Frame the problem      — task type + constraints
 2. Build a dumb baseline  — the floor every model must clear
 3. Pick algorithm family  — match algorithm to data/task
 4. Set up validation      — splits, CV strategy
 5. Train a simple model   — one strong default, well-tuned
 6. Iterate on candidates  — try 3-5 contenders side by side
 7. Tune hyperparameters   — grid → random → Bayesian
 8. Address imbalance/bias — class weights, resampling, calibration
 9. Compare on validation  — metric + complexity + cost + interpretability
10. Lock the final model   — single test-set evaluation; freeze
```

---

## Step 1 — Frame the Problem

Match the task to the model space. This is mostly about **what shape is the answer**.

| Task type | Output | Examples |
|---|---|---|
| Binary classification | 0 / 1 (or probability) | Churn, fraud, spam |
| Multi-class classification | one of K classes | Image labels, intent detection |
| Multi-label classification | subset of K classes | Tags on a blog post |
| Regression | continuous number | Price, demand, temperature |
| Ranking | ordered list | Search results, recommendations |
| Time series forecasting | future values | Sales next 7 days |
| Clustering | unlabelled groups | Customer segments |
| Anomaly detection | outlier score | Fraud, system faults |
| Sequence-to-sequence | text → text | Translation, summarisation |
| Generation | sample from a distribution | LLM, image generation |

### Constraints that shape the choice
- **Latency budget** — < 10 ms? rules out giant ensembles.
- **Memory budget** — edge / mobile? small models only.
- **Training data size** — 200 rows vs 200 million.
- **Interpretability** — must explain decisions to regulators? linear/tree.
- **Cost per prediction** — LLM API per call vs cached tabular model.
- **Update frequency** — retrain hourly vs yearly.

Write the constraints down before picking the model — they often eliminate
80% of options instantly.

---

## Step 2 — Build a Dumb Baseline

Always. No exceptions. Before any "real" model.

| Task | Dumb baseline |
|---|---|
| Classification | Predict majority class. `DummyClassifier(strategy="most_frequent")` |
| Regression | Predict the mean / median. `DummyRegressor(strategy="mean")` |
| Time series | Predict yesterday's value (naïve), or seasonal naïve |
| Ranking | Random permutation, or popularity baseline |
| Recommendation | Top-N most popular |

```python
from sklearn.dummy import DummyClassifier
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("baseline:", dummy.score(X_val, y_val))
```

If your fancy model can't beat this, **something is wrong** — usually leakage in
the baseline (rare) or a bug in your training pipeline (common).

Also try a **slightly less dumb baseline**: logistic regression with default
hyperparameters. That's the bar a more complex model must clear by enough margin
to justify its cost.

---

## Step 3 — Pick the Algorithm Family

### 3.1 The cheat sheet

| You have | Try first | Then maybe |
|---|---|---|
| Tabular, < 1K rows | Logistic / linear regression | Small RF |
| Tabular, 1K–1M rows | XGBoost / LightGBM / CatBoost | Stacked ensemble |
| Tabular, > 1M rows | LightGBM (fastest) | Distributed XGBoost |
| Many features, sparse | Logistic regression with L1/L2 | LightGBM |
| Image classification | Pre-trained CNN / ViT (transfer) | Fine-tune full |
| Object detection / segmentation | YOLO / Mask R-CNN / DETR | — |
| Text classification | DistilBERT / RoBERTa fine-tune | Linear SVM on TF-IDF |
| Text generation | LLM API + prompting / RAG | Fine-tune small open model |
| Tabular time series | LightGBM with lag features | Prophet, ARIMA |
| Multivariate time series | Temporal Fusion Transformer | NHITS, N-BEATS |
| Recommendations | Matrix factorisation / two-tower | LightFM, neural CF |
| Clustering | K-means | DBSCAN, HDBSCAN, GMM |
| Anomaly detection | Isolation Forest | One-class SVM, autoencoder |

### 3.2 Why each family wins

- **Linear / logistic regression** — fast, interpretable, hard to overfit on tiny data, gives calibrated probabilities, a great baseline.
- **Tree ensembles (XGBoost/LightGBM/CatBoost)** — handle missing values, mixed types, non-linearity, interactions; minimal preprocessing. **Default winner on tabular data.**
- **Random Forest** — almost as good as boosting, less tuning, robust to noise.
- **SVM** — strong on small-to-medium tabular when features are dense; sensitive to scaling.
- **K-Nearest Neighbours** — simple, good when local structure matters; scales poorly.
- **Naïve Bayes** — fast text baseline; assumes feature independence.
- **Neural networks** — when data is large and signal is non-tabular (image, text, audio); wasted on small tabular sets where boosting wins.
- **Transformers** — modern default for text; for tabular, only worth it on very large datasets (TabPFN / Tabular Transformers are an exception).

### 3.3 Build a "candidate list" of 3-5
You don't need 50 models. Pick 3-5 that span **simple → complex**:
- 1 dumb baseline
- 1 simple linear / tree model
- 1-2 strong defaults (LightGBM, XGBoost)
- Optionally 1 neural / specialised model

That's enough to know whether complexity is paying off.

---

## Step 4 — Set Up Validation

### 4.1 Splits — already covered in [ML_AI_ENGINEER_WORKFLOW.md](ML_AI_ENGINEER_WORKFLOW.md), recap:

| Data shape | Strategy |
|---|---|
| IID tabular | Random 70/15/15 |
| Time series | Chronological split |
| Grouped (multiple rows per user) | GroupKFold — no group in two splits |
| Imbalanced classification | Stratified |

### 4.2 Cross-validation strategies

| CV | When | sklearn |
|---|---|---|
| K-Fold (k=5 or 10) | IID, plenty of data | `KFold` |
| Stratified K-Fold | Classification, imbalanced | `StratifiedKFold` |
| Group K-Fold | Multiple rows per entity | `GroupKFold` |
| Time-Series Split | Temporal data | `TimeSeriesSplit` |
| Repeated K-Fold | Small data, want low variance estimate | `RepeatedKFold` |
| Leave-One-Out | Tiny data (< 100 rows) | `LeaveOneOut` |
| Nested CV | Hyperparameter tuning + honest evaluation | manual |

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
print(scores.mean(), "+/-", scores.std())
```

### 4.3 The golden rule
**Lock the test set away.** No CV, no tuning, no peeking. You evaluate on it
*once*, at the end. Touching it more makes your "test" score optimistic.

---

## Step 5 — Train a Strong Default

Pick **one** strong model and train it well before iterating. Usually this is
LightGBM for tabular, fine-tuned BERT for text, fine-tuned ResNet for images.

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=0,
)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
```

This single, well-trained model is your **reference**. Every later candidate
must beat it on validation by enough to justify its complexity.

---

## Step 6 — Iterate on Candidates

For each candidate model:
1. Train with sane defaults.
2. Evaluate on the same validation split.
3. Record the metric, train time, model size, latency.

```python
results = []
for name, m in candidates.items():
    m.fit(X_train, y_train)
    score = roc_auc_score(y_val, m.predict_proba(X_val)[:, 1])
    results.append({"model": name, "auc": score})
print(pd.DataFrame(results).sort_values("auc", ascending=False))
```

### When to stop adding candidates
- Two models score within ~1 standard deviation of each other → diminishing returns.
- Time budget is up.
- The simplest acceptable model already meets the business metric.

---

## Step 7 — Hyperparameter Tuning

### 7.1 Tune the right things
Each model has 2-5 hyperparameters that matter. Don't waste budget on the rest.

| Model | Top hyperparameters |
|---|---|
| Logistic regression | `C` (inverse of regularisation), `penalty` (l1/l2/elasticnet) |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_leaf`, `max_features` |
| XGBoost / LightGBM | `n_estimators`, `learning_rate`, `max_depth`/`num_leaves`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda` |
| SVM | `C`, `kernel`, `gamma` |
| KNN | `n_neighbors`, `weights`, `metric` |
| Neural network | learning rate, batch size, weight decay, dropout, depth/width |

### 7.2 Search strategies

| Strategy | When | Tool |
|---|---|---|
| **Grid search** | < 100 combinations, definite ranges | `GridSearchCV` |
| **Random search** | Many params, large space | `RandomizedSearchCV` — usually better than grid for the same budget |
| **Bayesian / TPE** | Expensive to train, want sample efficiency | **Optuna**, Hyperopt |
| **Hyperband / ASHA** | Can early-stop bad trials | Optuna pruners, Ray Tune |
| **Population-based training** | Neural nets at scale | Ray Tune, Weights & Biases sweeps |

### 7.3 Optuna — the modern default
```python
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

def objective(trial):
    params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "learning_rate": trial.suggest_float("lr", 1e-3, 0.3, log=True),
        "num_leaves":    trial.suggest_int("num_leaves", 15, 255),
        "max_depth":     trial.suggest_int("max_depth", -1, 12),
        "min_child_samples": trial.suggest_int("min_child", 5, 100),
        "subsample":     trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample", 0.5, 1.0),
        "reg_alpha":     trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
        "reg_lambda":    trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }
    model = lgb.LGBMClassifier(n_estimators=2000, **params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50)])
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=0),
                            pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=100, timeout=3600)
print("best:", study.best_value, study.best_params)
```

### 7.4 Tuning hygiene
- Always use **early stopping** (when supported) — saves enormous time.
- Tune on **validation** or via CV; never on test.
- **Set seeds** so results are reproducible.
- Use **log-uniform** sampling for `learning_rate`, `C`, regularisation strengths.
- Diminishing returns kick in fast — most of the gain comes in the first 30 trials.
- If random search and Bayesian end up at similar scores, your **features** are
  the bottleneck, not your hyperparameters.

---

## Step 8 — Imbalance, Class Weights, Calibration

### 8.1 Class imbalance treatments

| Approach | When | How |
|---|---|---|
| **Do nothing** | Imbalance < ~70/30 and AUC-style metric | — |
| **Class weights** | Most cases (cheap, no data change) | `class_weight="balanced"` or `scale_pos_weight=neg/pos` (XGBoost) |
| **Threshold tuning** | You only need correct ranking, not the default 0.5 cutoff | Tune `t` on validation |
| **Resampling** | Severe imbalance (≥ 95/5) | SMOTE, ADASYN, random over/under-sampling — **train only** |
| **Focal loss** | Detection / NN settings | Custom loss function |
| **Anomaly framing** | One class extremely rare | Isolation Forest, one-class SVM |

```python
from imblearn.over_sampling import SMOTE
X_res, y_res = SMOTE(random_state=0).fit_resample(X_train, y_train)
# fit on resampled, evaluate on ORIGINAL distribution validation
```

### 8.2 Probability calibration
If your downstream system uses the probability (not just the class), you must
calibrate. Tree ensembles often output uncalibrated scores.

```python
from sklearn.calibration import CalibratedClassifierCV
cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
cal.fit(X_val, y_val)        # use a held-out set the base model didn't see
```

Check calibration with a **reliability diagram** (`sklearn.calibration.calibration_curve`).

---

## Step 9 — Compare Candidates Holistically

Don't pick on metric alone. Score each candidate on:

| Axis | Why it matters |
|---|---|
| **Validation metric** | Does it predict well? |
| **Variance across folds** | Is it stable? Pick the lower-variance model when scores are close. |
| **Train time** | Can we retrain often? |
| **Inference latency** | Can we serve it at SLA? |
| **Model size** | Memory, deployment target (edge vs cloud). |
| **Explainability** | Can we audit decisions? |
| **Maintenance cost** | Custom code? unusual deps? GPU required? |

Make a small comparison table and discuss with the team. The right answer is
often "the *second*-best model, because it's 5× faster".

```python
import pandas as pd
summary = pd.DataFrame([
    {"model": "logreg",   "auc_val": 0.81, "latency_ms": 0.3, "size_mb": 0.01, "explain": "high"},
    {"model": "lightgbm", "auc_val": 0.89, "latency_ms": 1.2, "size_mb": 4.2,  "explain": "med"},
    {"model": "xgboost",  "auc_val": 0.89, "latency_ms": 1.8, "size_mb": 6.0,  "explain": "med"},
    {"model": "deep nn",  "auc_val": 0.90, "latency_ms": 8.0, "size_mb": 35.0, "explain": "low"},
])
```

---

## Step 10 — Lock the Final Model

1. Pick one winner from validation comparison.
2. Optional: **refit on train + val combined** with the chosen hyperparameters
   to use all available data.
3. **Evaluate once** on the held-out test set.
4. Report: metric + 95% bootstrap CI, latency, size, training data version,
   commit hash, hyperparameters.
5. **Save** the trained pipeline (Step 12 of feature-engineering guide).
6. **Freeze** — no more tuning.

```python
# Bootstrap CI on the test metric
import numpy as np
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
preds = model.predict_proba(X_test)[:, 1]
boots = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    boots.append(roc_auc_score(y_test.iloc[idx], preds[idx]))
print(f"AUC = {np.mean(boots):.3f} (95% CI {np.percentile(boots,2.5):.3f}-{np.percentile(boots,97.5):.3f})")
```

If the test number is much worse than validation, you overfit to validation
during tuning — go back, simplify, retune.

---

## Stacking, Blending, Ensembling

When a single model has plateaued, sometimes 1-2% extra comes from combining
diverse models.

| Technique | How |
|---|---|
| **Voting** | Average predictions from N models; simplest |
| **Weighted blend** | Tune weights with a small linear search on validation |
| **Stacking** | Train a meta-model (often LR) on the predictions of base models, using out-of-fold predictions to avoid leakage |

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
stack = StackingClassifier(
    estimators=[("lgb", lgb_model), ("rf", rf_model), ("lr", lr_model)],
    final_estimator=LogisticRegression(),
    cv=5, passthrough=False,
).fit(X_train, y_train)
```

Caveats: ensembles are slower, harder to deploy, and gain little when one
model is already strong. Use only when the marginal % is worth the ops cost.

---

## Anti-Patterns

- **Tuning with the test set in the loop.** Inflated scores; real-world disappointment.
- **Random shuffling time-series data.** Future leaks into past.
- **Not seeding randomness.** Can't reproduce results.
- **Comparing models on different splits.** Use the same CV folds.
- **Tuning before training a strong default.** Waste of trials.
- **Tuning every hyperparameter.** Most don't matter — focus on the top 3-5.
- **Picking the model with the highest single fold score.** Pick by mean *and* variance.
- **Throwing away a simple baseline because it's "boring".** Often the best ROI in production.
- **No early stopping.** You've trained for 1000 epochs to overfit; 50 was enough.
- **Trusting one metric.** Combine metric + latency + size + explainability.
- **Refitting on test set.** Test is sacred. Refit on train+val if anything.

---

## The 10-Step Checklist (Print This)

- [ ] 1. Task type, output shape, and constraints written down
- [ ] 2. Dumb baseline trained and beaten by a simple model
- [ ] 3. Candidate list of 3-5 models spanning simple → complex
- [ ] 4. Validation strategy (split + CV) chosen and locked
- [ ] 5. One strong default trained well and used as reference
- [ ] 6. Each candidate trained, scored, with train time + size + latency
- [ ] 7. Top candidates tuned with random/Bayesian search + early stopping
- [ ] 8. Class imbalance handled appropriately; probabilities calibrated if needed
- [ ] 9. Models compared on metric + variance + cost + explainability
- [ ] 10. Final model refit, evaluated **once** on test set, reported with CI, saved

When this is ticked, move on to detailed evaluation —
[MODEL_EVALUATION_GUIDE.md](MODEL_EVALUATION_GUIDE.md).
