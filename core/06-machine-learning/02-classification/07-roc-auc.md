# Classification 7 — ROC Curve, AUC, Cost-Benefit Analysis

## Lectures covered
- Model Evaluation: ROC Curve & AUC
- Cost Benefit Analysis Using ROC in Sklearn

---

## 1. The setup

A binary classifier outputs a probability for each example. To turn that into a 0/1 prediction, you choose a threshold.

Different thresholds → different (TPR, FPR) pairs. The **ROC curve** plots all these tradeoffs.

---

## 2. TPR and FPR

- **True Positive Rate** = Recall = TP / (TP + FN) — "of true positives, how many caught?"
- **False Positive Rate** = FP / (FP + TN) — "of true negatives, how many wrongly flagged?"

We want TPR high and FPR low.

---

## 3. The ROC curve

Y-axis: TPR (recall). X-axis: FPR. Sweep threshold from 0 → 1, plot the resulting (FPR, TPR) point.

```
TPR (recall)
1 ┤              ──────
  │           ──/
  │        ──/
  │     ──/
0 ┤  ──/  diagonal = random
  └──────────────────► FPR
  0                    1
```

- **Top-left corner (0, 1)** is perfection
- **Diagonal line** is random guessing
- **Better classifier** = curve hugs the top-left corner

```python
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr); plt.plot([0,1], [0,1], "k--")
plt.xlabel("FPR"); plt.ylabel("TPR")
```

---

## 4. AUC — Area Under the ROC Curve

A single number summarizing the ROC curve.
- 0.5 = random
- 1.0 = perfect
- 0.7–0.8 = decent
- 0.8–0.9 = good
- > 0.9 = strong

```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_prob)
```

### What AUC actually measures
AUC = probability that a random positive is ranked higher than a random negative. **It's a ranking metric**, not a calibration metric.

### Why AUC is popular
- Threshold-independent — judges the model's ranking quality
- Single number, easy to compare across models
- Same value regardless of class distribution (mostly)

### Where AUC misleads
- **Severe imbalance**: AUC can stay high even when precision is poor
- **Different operating points** between models — AUC doesn't tell you which is better at *your* threshold
- For "PR-AUC > ROC-AUC at extreme imbalance" use `average_precision_score`

---

## 5. Precision-Recall curve & PR-AUC

When positives are rare, plot Precision (y) vs Recall (x).

```python
from sklearn.metrics import precision_recall_curve, average_precision_score
prec, rec, thr = precision_recall_curve(y_test, y_prob)
plt.plot(rec, prec)
print(average_precision_score(y_test, y_prob))
```

- AUC for this curve = **Average Precision**
- More informative than ROC-AUC when positives are <10%

---

## 6. Picking the operating point

### Geometric — closest to (0, 1)
```python
import numpy as np
distances = np.sqrt(fpr**2 + (1 - tpr)**2)
best_ix = np.argmin(distances)
best_threshold = thresholds[best_ix]
```

### Youden's J — TPR - FPR
```python
j_scores = tpr - fpr
best_ix = np.argmax(j_scores)
```

### Business-driven
"We can review 1000 cases per day." Set threshold so the top-1000 highest-probability cases are flagged.

```python
y_prob_sorted = np.sort(y_prob)[::-1]
threshold = y_prob_sorted[999]                 # 1000th-highest prob
```

---

## 7. Cost-benefit analysis using ROC

Codebasics walks through this. The idea:
- Each TP gives you **benefit** B (e.g., $100 saved per detected fraud)
- Each FP costs you C (e.g., $5 of review)
- Each FN costs you K (e.g., $10,000 of unrecovered fraud)
- Each TN is neutral (~0)

For each threshold:
$$\text{Profit} = TP \cdot B - FP \cdot C - FN \cdot K$$

Pick the threshold maximizing profit.

```python
import numpy as np
from sklearn.metrics import confusion_matrix

best_profit, best_threshold = -np.inf, 0.5
for t in np.linspace(0.01, 0.99, 99):
    y_pred = (y_prob >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    profit = tp * 100 - fp * 5 - fn * 10000
    if profit > best_profit:
        best_profit, best_threshold = profit, t

print(best_threshold, best_profit)
```

This grounds threshold choice in **dollars**, which executives understand instantly.

---

## 8. Multiclass ROC

For k classes, compute ROC per class (One-vs-Rest):
```python
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_prob_matrix, multi_class="ovr", average="macro")
```

---

## 9. Calibration — when probabilities matter

ROC-AUC is about ranking. If you need probabilities to *match reality* (e.g., "30% chance of default" should mean ~30 of 100 such cases default), check **calibration**:

```python
from sklearn.calibration import calibration_curve
prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
plt.plot(prob_pred, prob_true, marker="o"); plt.plot([0,1],[0,1],"k--")
```

Curve hugging the diagonal = well-calibrated. SVMs and tree ensembles are often poorly calibrated → wrap in `CalibratedClassifierCV` for true probabilities.

---

## 10. Real workflow

```python
from sklearn.metrics import (roc_curve, roc_auc_score,
                              precision_recall_curve, average_precision_score)

y_prob = model.predict_proba(X_test)[:, 1]

print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")
print(f"PR-AUC:  {average_precision_score(y_test, y_prob):.3f}")

fpr, tpr, _ = roc_curve(y_test, y_prob)
prec, rec, _ = precision_recall_curve(y_test, y_prob)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(fpr, tpr); ax[0].plot([0,1],[0,1],"k--"); ax[0].set_title("ROC")
ax[1].plot(rec, prec); ax[1].set_title("Precision-Recall")
```

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Reporting AUC on extreme imbalance | misleads about precision | also report PR-AUC |
| Reading AUC as "accuracy" | different concept | AUC = ranking quality |
| Assuming model probabilities are calibrated | downstream cost miscalc | calibrate explicitly |
| Different thresholds across models in same comparison | apples to oranges | compare at same operating point or via AUC |
| Plot ROC without zoom | hard to see differences | zoom into top-left corner |

## Self-check

- [ ] Define TPR and FPR.
- [ ] What does AUC = 0.85 actually mean (in ranking terms)?
- [ ] When prefer PR-AUC over ROC-AUC?
- [ ] How do you pick a decision threshold using cost-benefit?
- [ ] What's calibration and when does it matter?
- [ ] Why do tree ensembles often have poor calibration?
- [ ] Walk through a decision: "should we use threshold 0.3 or 0.5?" given a confusion matrix.
- [ ] Compute Youden's J and explain when use it.
