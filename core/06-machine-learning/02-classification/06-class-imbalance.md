# Classification 6 — Handling Class Imbalance

## Lectures covered
- Handle Class Imbalance

---

## 1. The problem

Real datasets often have **skewed class distributions**:
- 99% non-fraud, 1% fraud
- 95% non-default, 5% default
- 99% healthy, 1% diseased

A naive model predicts "majority always" and gets 99% accuracy — useless. We need techniques to make the model attend to the minority.

---

## 2. Three strategy families

1. **Algorithmic** — change the model's loss function via class weights
2. **Data-level** — resample (oversample minority, undersample majority)
3. **Threshold-level** — tune the decision threshold using calibrated probabilities

You can combine them.

---

## 3. Class weights (the easiest first move)

Most sklearn classifiers accept `class_weight`:

```python
from sklearn.linear_model import LogisticRegression

LogisticRegression(class_weight="balanced")              # auto: inversely proportional to class freq
LogisticRegression(class_weight={0: 1, 1: 10})           # manual: minority gets 10x weight
```

Same for `RandomForestClassifier`, `SVC`, etc.

Effect: the loss penalizes minority-class mistakes more heavily, biasing the model toward catching them.

---

## 4. Random under/oversampling

### Undersample majority
Drop random rows from the majority class until classes balance.
- Pro: fast, smaller dataset
- Con: throws away real data

### Oversample minority (random)
Duplicate minority rows.
- Pro: keeps all data
- Con: minority points repeated → may overfit

### imbalanced-learn
```bash
pip install imbalanced-learn
```
```python
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X_train, y_train)
```

---

## 5. SMOTE — Synthetic Minority Oversampling

Instead of duplicating minority points, **synthesize new ones** by interpolating between existing minority points and their neighbors.

```python
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42, k_neighbors=5)
X_res, y_res = sm.fit_resample(X_train, y_train)
```

### Variants
- **SMOTE** — standard
- **BorderlineSMOTE** — focuses on minority points near decision boundary
- **ADASYN** — generates more synthetic samples in harder regions

> **Always SMOTE only on the training set, never on test.** Inside a pipeline:
> ```python
> from imblearn.pipeline import Pipeline as ImbPipeline
> pipe = ImbPipeline([("smote", SMOTE()), ("lr", LogisticRegression())])
> ```
> (This pipeline applies SMOTE only on `.fit`, not `.transform`/`.predict` — correct behavior.)

---

## 6. Combining over- and under-sampling

```python
from imblearn.combine import SMOTETomek
ovsmt = SMOTETomek(random_state=42)
X_res, y_res = ovsmt.fit_resample(X_train, y_train)
```

Often gives the best balance.

---

## 7. Threshold tuning instead of resampling

Many problems can be solved without changing the data — just lower the decision threshold.

```python
from sklearn.metrics import precision_recall_curve
import numpy as np

y_prob = model.predict_proba(X_val)[:, 1]
prec, rec, thr = precision_recall_curve(y_val, y_prob)

# pick threshold for, say, 90% recall:
target_recall = 0.90
ix = np.where(rec >= target_recall)[0][-1]
threshold = thr[ix]

y_pred = (y_prob >= threshold).astype(int)
```

This is often **more honest** than resampling — you trained on the real distribution and just chose a different operating point.

---

## 8. Cost-sensitive learning

If you can quantify the actual *dollar cost* of false positives vs false negatives, optimize for total expected cost rather than accuracy:

```python
def expected_cost(y_true, y_pred, c_fp=10, c_fn=1000):
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return c_fp * fp + c_fn * fn
```

For fraud: a missed fraud costs $10,000; a false alarm costs $5 of analyst time. Threshold = where marginal cost of FN = marginal cost of FP.

---

## 9. Algorithm choices that handle imbalance well

- **Tree-based ensembles** with `class_weight` — robust
- **XGBoost / LightGBM** with `scale_pos_weight` (= negative_count / positive_count)
- **Anomaly detection** algorithms (IsolationForest, OneClassSVM) when the minority is truly anomalous
- **Focal loss** (originally from object detection; available in some DL libs) — down-weights easy examples

---

## 10. Real workflow on imbalanced data

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

pipe = ImbPipeline([
    ("scale", StandardScaler()),
    ("smote", SMOTE(random_state=42)),
    ("lr",    LogisticRegression(max_iter=2000)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="f1")
print(scores.mean(), "±", scores.std())
```

`StratifiedKFold` ensures each fold has the same class distribution.

---

## 11. Evaluation under imbalance

- **Don't use accuracy** as your metric
- **Do use** Macro F1, ROC-AUC, PR-AUC, recall at fixed precision
- **PR-AUC > ROC-AUC** when imbalance is extreme — ROC-AUC stays optimistic when most negatives are easy
- Always look at the **confusion matrix** at your chosen threshold

---

## 12. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Applying SMOTE before train/test split | leakage | always inside a pipeline / after split |
| Reporting accuracy on imbalanced | inflated and useless | report F1 / AUC / PR-AUC |
| Oversampling test set too | optimistic test metrics | only resample train |
| Using ROC-AUC alone on extreme imbalance | misleading | also report PR-AUC |
| Forgetting `stratify=y` in train_test_split | uneven distribution per split | always stratify |

## Self-check

- [ ] Why is accuracy a bad metric on imbalanced data?
- [ ] Three families of strategies for handling imbalance.
- [ ] Difference between random oversampling and SMOTE?
- [ ] Why must SMOTE be applied only to training data?
- [ ] What's `class_weight="balanced"` doing under the hood?
- [ ] When use threshold tuning vs SMOTE?
- [ ] Why is PR-AUC often preferred over ROC-AUC for rare events?
- [ ] What's `scale_pos_weight` in XGBoost?
