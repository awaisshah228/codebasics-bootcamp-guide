# Model Evaluation — Step-by-Step Guide

> Companion to `ML_AI_ENGINEER_WORKFLOW.md` (Stage 7).
> A model's job isn't to score high — it's to make a real-world decision well.
> Evaluation is how you find out *whether* it does, *where* it doesn't, and
> *how it will behave when it sees real data*.

> Sister files:
> - [ML_AI_ENGINEER_WORKFLOW.md](ML_AI_ENGINEER_WORKFLOW.md)
> - [DATA_CLEANING_GUIDE.md](DATA_CLEANING_GUIDE.md)
> - [FEATURE_ENGINEERING_GUIDE.md](FEATURE_ENGINEERING_GUIDE.md)
> - [MODEL_SELECTION_GUIDE.md](MODEL_SELECTION_GUIDE.md) ← what came before

---

## The 10-Step Sequence (TL;DR)

```
 1. Pick metrics tied to the decision
 2. Build the right comparison (vs baseline, vs current system)
 3. Confusion matrix / residuals — where are the errors?
 4. Threshold tuning (classification)
 5. Per-slice evaluation — fairness + segment performance
 6. Calibration — are probabilities meaningful?
 7. Robustness — perturbations and OOD inputs
 8. Statistical confidence — bootstrap CIs, significance vs baseline
 9. Qualitative error analysis — read 100 wrong predictions
10. Final test set protocol — touch once, report fully
```

---

## Step 1 — Pick Metrics Tied to the Decision

A metric is a **proxy** for what you actually care about. Picking the wrong
proxy is the most common reason models that "work" in evaluation fail in production.

### Classification

| Metric | Range | When to use |
|---|---|---|
| **Accuracy** | 0-1 | Balanced classes, all errors equally bad |
| **Precision** | 0-1 | False positives are costly (spam → real email blocked) |
| **Recall (Sensitivity)** | 0-1 | False negatives are costly (cancer screening) |
| **F1 / Fβ** | 0-1 | Trade off P and R; β > 1 favours recall, β < 1 favours precision |
| **ROC-AUC** | 0.5-1.0 | Ranking quality, threshold-free, balanced classes |
| **PR-AUC** | 0-1 | Imbalanced classes (rare positives) |
| **Log-loss / Cross-entropy** | 0-∞ | Probability calibration matters |
| **Brier score** | 0-1 | Mean-squared error of probabilities |
| **MCC** | -1 to 1 | Balanced summary; robust to imbalance |
| **Cohen's κ** | -1 to 1 | Agreement above chance |
| **Top-k accuracy** | 0-1 | Multi-class with many classes |

### Regression

| Metric | When |
|---|---|
| **MAE** | Robust to outliers; same units as target |
| **RMSE** | Penalises large errors more; same units |
| **MAPE / sMAPE** | Percentage error; care with zeros |
| **R²** | Variance explained; relative to mean baseline |
| **Quantile loss / Pinball** | Estimating quantiles, not the mean |
| **MASE** | Time series — scaled by naïve forecast |

### Ranking / Recommendation
- **NDCG@k** — discounted gain at position k.
- **MAP@k** — mean average precision.
- **MRR** — mean reciprocal rank.
- **Hit Rate / Recall@k** — did the right answer make the top k?

### Generation (LLM, summarisation, translation)
- **BLEU / ROUGE** — n-gram overlap; cheap but noisy.
- **BERTScore / chrF** — semantic similarity.
- **LLM-as-a-judge** — model evaluates model output. Cheap, biased, useful as a screen.
- **Human evaluation** — golden ground truth; expensive, irreplaceable.

### Business / decision metrics
The metric that matters in the end. Always derive and report at least one:
- **Revenue uplift** if we deploy.
- **Cost saved per prediction.**
- **Conversion / retention rate change.**
- **Net dollar value** at the chosen operating threshold.

A model with 0.85 AUC and 0.5% conversion lift can beat one with 0.92 AUC and
0.3% lift — never lose sight of the decision.

---

## Step 2 — Build the Right Comparison

Numbers are meaningless without context. Always compare against:

| Comparison | Why |
|---|---|
| **Dumb baseline** | Confirms the model learned anything (Step 5 of selection guide) |
| **Simple model** (logistic / linear) | Confirms complexity is paying off |
| **Current production system** | The bar that actually matters; if you can't beat it, you don't ship |
| **Human performance** | Especially for classification with ground truth |
| **Theoretical ceiling** | Bayes error / inter-annotator agreement |

```python
results = {
    "dummy":    auc(y_test, dummy.predict_proba(X_test)[:,1]),
    "logreg":   auc(y_test, lr.predict_proba(X_test)[:,1]),
    "lightgbm": auc(y_test, lgb.predict_proba(X_test)[:,1]),
    "current":  auc(y_test, current_system_preds),
}
```

---

## Step 3 — Confusion Matrix / Residuals

### 3.1 Classification — confusion matrix is non-negotiable
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)   # rows = true, cols = predicted
ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"]).plot()
```

For a binary classifier, the four cells are:
```
                   pred neg     pred pos
actual neg     |    TN       |    FP        |
actual pos     |    FN       |    TP        |
```
Then derive precision = TP/(TP+FP), recall = TP/(TP+FN), specificity = TN/(TN+FP).

For multi-class: look at the off-diagonal — which classes are confused with which?
That's where you focus feature engineering / data labelling effort.

### 3.2 Regression — look at residuals
```python
import numpy as np, matplotlib.pyplot as plt
resid = y_test - y_pred
plt.scatter(y_pred, resid); plt.axhline(0)
```

Things to spot:
- Residuals scaled with `y_pred` → variance grows; consider log-transforming target.
- Pattern in residuals → model missed structure (try more features / non-linear model).
- Residuals biased away from 0 in some range → systematic error.
- Heavy tails → MAE may be more honest than RMSE; or treat outliers separately.

---

## Step 4 — Threshold Tuning (Classification)

The default threshold of `0.5` is **arbitrary**. The right threshold depends on
the cost of false positives vs false negatives.

```python
from sklearn.metrics import precision_recall_curve, f1_score
import numpy as np

p, r, t = precision_recall_curve(y_val, model.predict_proba(X_val)[:, 1])
f1 = 2 * p * r / (p + r + 1e-9)
best_t = t[np.nanargmax(f1[:-1])]
print("best threshold:", best_t)
```

### Choose the threshold by the *decision*

| Goal | How |
|---|---|
| Maximise F1 | argmax over precision-recall curve |
| Hit a target precision | smallest threshold with `precision ≥ X` |
| Hit a target recall | largest threshold with `recall ≥ X` |
| Maximise expected $ value | `argmax over t` of `TP*v_TP - FP*c_FP - FN*c_FN` |

Always tune the threshold on **validation**, not test.

---

## Step 5 — Per-Slice Evaluation

Aggregate metrics hide problems. Always slice by:
- **Demographic / sensitive groups** (age, gender, region, ethnicity, language).
- **Behavioural segments** (new user vs returning, free vs paid).
- **Volume tiers** (high-volume vs long-tail products).
- **Time** (recent vs older test data).
- **Hard segments** (long inputs, edge cases, rare classes).

```python
slices = {"new_users": X_test["is_new"] == 1,
          "old_users": X_test["is_new"] == 0,
          "country_us": X_test["country"] == "US",
          "country_other": X_test["country"] != "US"}
for name, mask in slices.items():
    s = roc_auc_score(y_test[mask], preds[mask])
    print(f"{name:15s}  n={mask.sum():6d}  AUC={s:.3f}")
```

A model at 0.90 overall but 0.55 on women is **broken**, even if the overall
metric looks great. Catch this *before* shipping.

### Fairness metrics worth a look
- **Demographic parity** — equal positive prediction rate per group.
- **Equalised odds** — equal TPR and FPR per group.
- **Equal opportunity** — equal TPR per group (for the "advantaged" outcome).
- **Disparate impact ratio** — group A positive rate / group B positive rate ≥ 0.8 (rule of thumb).

Tools: `fairlearn`, `aif360`.

---

## Step 6 — Calibration

A 0.7 prediction should be right **70% of the time**. Many models — especially
boosted trees and SVMs — are not calibrated by default.

```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_val, model.predict_proba(X_val)[:, 1],
                                         n_bins=10, strategy="quantile")
# plot prob_pred (x) vs prob_true (y); diagonal = perfect calibration
```

Quantitative: **Brier score** — lower is better.

If miscalibrated:
- `CalibratedClassifierCV(method="isotonic")` — flexible, needs more data.
- `CalibratedClassifierCV(method="sigmoid")` — Platt scaling; small data.
- Use a **held-out calibration set** the base model didn't see.

When does calibration matter?
- Downstream decisions multiply probability by a value (expected revenue / cost).
- Stakeholders interpret probabilities directly ("we are 80% sure...").
- You compare scores across models or across time.

---

## Step 7 — Robustness Checks

A model that scores 0.92 in evaluation can fall to 0.70 the moment the data
changes slightly. Test it before production does.

### 7.1 Perturbation tests
- Add small Gaussian noise to numeric features → does score collapse?
- Swap categorical values for unseen ones → does it crash or default sensibly?
- Drop random features → which ones cause biggest drops? (sanity vs leakage)

### 7.2 Adversarial / edge cases
- Empty inputs, all-zero inputs, repeated tokens, very long / very short text.
- Mis-spellings, multilingual, emojis (text).
- Inputs from a *future* time period (does it generalise temporally?).

### 7.3 Out-of-distribution detection
Detect inputs that don't look like training data and refuse to predict.
Tools: `alibi-detect`, simple distance-to-train metrics, per-feature drift checks.

### 7.4 Stress testing the system
- Latency under load (p50/p95/p99 at expected QPS).
- Memory behaviour at peak.
- Error handling when a feature service times out.

---

## Step 8 — Statistical Confidence

A test-set score is a **point estimate** with noise. Always report a CI.

### 8.1 Bootstrap confidence intervals
```python
import numpy as np
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(0)
def boot_ci(y, p, n=1000, q=(2.5, 97.5)):
    scores = []
    n_obs = len(y)
    for _ in range(n):
        idx = rng.integers(0, n_obs, n_obs)
        scores.append(roc_auc_score(y.iloc[idx], p[idx]))
    return np.mean(scores), np.percentile(scores, q)

mean, (lo, hi) = boot_ci(y_test, preds)
print(f"AUC = {mean:.3f} (95% CI {lo:.3f}-{hi:.3f})")
```

### 8.2 Comparing two models
Use **paired** bootstrap or McNemar's test — never independent comparisons,
because the two models are evaluated on the *same* test rows.

```python
# Paired bootstrap of the difference
diffs = []
for _ in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    diffs.append(roc_auc_score(y_test.iloc[idx], preds_a[idx])
               - roc_auc_score(y_test.iloc[idx], preds_b[idx]))
diffs = np.array(diffs)
print(f"Δ AUC = {diffs.mean():.4f}, 95% CI {np.percentile(diffs,2.5):.4f}-{np.percentile(diffs,97.5):.4f}")
```
If the CI **excludes 0**, the improvement is significant.

### 8.3 In production: A/B testing
Offline metrics estimate; online A/B confirms. Decide:
- Sample size needed for the effect you care to detect (power analysis).
- Randomisation unit (user / session / request).
- Primary metric, guardrails, and stop criteria, **before** launching.

---

## Step 9 — Qualitative Error Analysis

Pull 50-100 of the worst predictions and read them like a human.

```python
import pandas as pd
err = (
    pd.DataFrame({"y": y_test, "p": preds, "abs_err": np.abs(y_test - preds)})
      .sort_values("abs_err", ascending=False)
      .head(100)
)
# Join with original X_test for full context
```

Group the failures into **buckets** ("noisy labels", "missing context", "feature X is broken").
Each bucket suggests a fix:
- Bad labels → relabel a sample, or remove from training.
- Missing context → engineer a new feature.
- Bug → fix code, retrain.
- Genuinely ambiguous → set expectation, not solvable by ML.

This is the single highest-ROI activity in the entire pipeline. The instinct
to "go straight to a fancier model" almost never beats reading the errors.

---

## Step 10 — Final Test Set Protocol

The protocol every model must follow before ship:

1. **Lock the test set** at the very start of the project.
2. **No model selection** done with the test set. No early peeks.
3. After training + tuning + validation, run the chosen model **once** on test.
4. Compute **all** the things in one report:
   - Headline metric + 95% bootstrap CI.
   - Confusion matrix / residual plot.
   - Per-slice metrics (with sample sizes).
   - Calibration curve + Brier score.
   - Robustness summary (perturbation, OOD, stress).
   - 50 worst predictions + analysis notes.
   - Comparison vs baseline + vs current production system.
   - Hyperparameters, training data version, code commit hash.
5. **Sign off** with stakeholders on whether to ship.
6. If you reject this model and try another, you must use a **fresh** test set
   for the next round, or pay for the increased optimism honestly.

### Build a model card
A short markdown file shipped alongside the model:
```
# Model: churn-v1.2 (committed 2026-05-04)

## Intended use
Predict 30-day churn for paid US users. NOT for free users.

## Data
- Source: warehouse table users.events (snapshot 2026-04-30)
- Train: 480k rows (2024-01 -> 2026-03), Val: 60k (2026-04), Test: 60k (frozen)

## Performance
- AUC test = 0.892 (95% CI 0.886-0.898)
- @ threshold 0.42: precision 0.71, recall 0.55
- Per-slice AUC: US 0.89, EU 0.84, APAC 0.79

## Known limitations
- Trained only on paid users — do not score free users.
- Recall drops on accounts < 14 days old.

## Failure modes
- Returns nonsense for users with > 365d inactivity (rare).

## Maintenance
- Retrain weekly; review monthly.
- Owner: <name@email>
```

---

## Anti-Patterns

- **Picking accuracy on imbalanced data.** Predicting "all negative" gives 99% accuracy and zero value.
- **Tuning on the test set.** Inflated numbers; production disaster.
- **Reporting one number with no CI.** Looks confident; isn't.
- **Aggregate-only metrics.** Hides per-segment failures.
- **Comparing two models on different splits / seeds.** Apples to oranges.
- **Calling a 0.001 AUC win "significant" with no test.** Often within noise.
- **Skipping qualitative error analysis.** You'll miss systematic bugs.
- **Ignoring calibration when downstream multiplies by probability.** Wrong $ decisions.
- **Evaluating on the *same* time period as training.** Real prod data is in the future.
- **Stopping at offline metrics with no plan for online A/B.** Offline is necessary, not sufficient.

---

## The 10-Step Checklist (Print This)

- [ ] 1. Metrics chosen, tied to the decision; secondary metrics named
- [ ] 2. Comparison set up: dumb baseline, simple model, current system
- [ ] 3. Confusion matrix / residual plot inspected
- [ ] 4. Threshold tuned on validation, not default 0.5
- [ ] 5. Per-slice evaluation done; no group is silently failing
- [ ] 6. Calibration checked and fixed if downstream uses probabilities
- [ ] 7. Robustness tested (perturbation, OOD, edge cases, latency)
- [ ] 8. CIs around metrics; significance vs baseline established
- [ ] 9. Qualitative error analysis done (100 worst predictions read)
- [ ] 10. Final test eval run **once**, full report + model card produced

When this is ticked, the model is ready to *consider* shipping (Stage 8 onward
in the main workflow: packaging → deployment → monitoring → feedback loop).
