# Classification 2 — Metrics: Confusion Matrix, Precision, Recall, F1

## Lectures covered
- Model Evaluation: Accuracy, Precision and Recall, F1 Score, Confusion Matrix

---

## 1. Confusion matrix — the source of all classification metrics

For binary classification:

|  | Predicted Negative | Predicted Positive |
|---|---|---|
| **Actual Negative** | TN (true negative) | FP (false positive / Type I error) |
| **Actual Positive** | FN (false negative / Type II error) | TP (true positive) |

```python
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
```

Plot it:
```python
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
```

Every metric below derives from these four numbers.

---

## 2. Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

"Of all predictions, how many were correct?"

### When accuracy lies
With a 99% / 1% class split, predicting "always majority" gives 99% accuracy and zero usefulness. **Never use accuracy alone on imbalanced data.**

```python
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)
```

---

## 3. Precision — "of those I flagged, how many are truly positive?"

$$\text{Precision} = \frac{TP}{TP + FP}$$

High precision → few false alarms.
Use when **false positives are costly**:
- Email spam filter (don't junk legit email)
- Recommending high-cost interventions

```python
from sklearn.metrics import precision_score
precision_score(y_test, y_pred)
```

---

## 4. Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

"Of all true positives, how many did I catch?"

High recall → few missed cases.
Use when **false negatives are costly**:
- Cancer screening (don't miss a sick patient)
- Fraud detection
- Security threats

```python
from sklearn.metrics import recall_score
recall_score(y_test, y_pred)
```

---

## 5. The precision-recall tradeoff

Lower decision threshold → more positives flagged:
- Recall ↑
- Precision ↓ (some flags are wrong)

Higher threshold → fewer positives flagged:
- Precision ↑
- Recall ↓ (miss some true positives)

You can't have both. Pick the side that matters for your business.

---

## 6. F1 Score — harmonic mean of precision and recall

$$F_1 = 2 \cdot \frac{\text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}$$

A single number that punishes imbalance between P and R. Used when you want both to be reasonable.

```python
from sklearn.metrics import f1_score
f1_score(y_test, y_pred)
```

### F-beta — weighted variant
$$F_\beta = (1 + \beta^2) \cdot \frac{P \cdot R}{\beta^2 P + R}$$

- β = 0.5 → emphasize precision
- β = 2 → emphasize recall

Use when one matters more but you still care about both.

---

## 7. classification_report — the cheat sheet

```python
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
```

Output (multiclass example):
```
              precision    recall  f1-score   support

           0       0.95      0.97      0.96       350
           1       0.78      0.71      0.74        50

    accuracy                           0.94       400
   macro avg       0.87      0.84      0.85       400
weighted avg       0.93      0.94      0.93       400
```

- **support**: # of true samples per class
- **macro avg**: average across classes (each class equally weighted)
- **weighted avg**: weighted by support (large classes dominate)

For imbalanced problems, **macro F1** is usually the metric to optimize.

---

## 8. Multiclass — confusion matrix at scale

For k classes:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred, labels=class_labels)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels, yticklabels=class_labels)
```

Each row sums to the number of true samples in that class. Diagonal = correct.

### Per-class metrics
- **Macro precision**: avg of per-class precision
- **Weighted precision**: avg weighted by support
- **Micro precision**: aggregate TP/FP across classes (= accuracy in single-label classification)

---

## 9. Picking the right metric — flowchart

```
Is the dataset balanced?
   │
   ├─ yes → accuracy is fine, F1 nice as backup
   │
   └─ no
        │
        Is one error type far more costly?
           │
           ├─ FP costly → optimize precision
           │
           ├─ FN costly → optimize recall
           │
           └─ both matter → F1 (or macro-F1 for multiclass)
```

If you can't pick: use **ROC-AUC** for ranking, **PR-AUC** for imbalanced.

---

## 10. Threshold tuning workflow

```python
import numpy as np
from sklearn.metrics import precision_recall_curve

y_prob = model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)

# pick threshold that maximizes F1
f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
best_idx = np.argmax(f1s[:-1])
best_threshold = thresholds[best_idx]
y_pred_best = (y_prob >= best_threshold).astype(int)
```

In production, often you'd pick the threshold that hits a *business target* ("we can review 200 cases/day → set threshold so ~200 flagged").

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Reporting accuracy on imbalanced data | misleadingly high | report precision, recall, F1 |
| Confusing FP and FN | wrong tradeoff chosen | draw the confusion matrix |
| Comparing two models on different thresholds | noise | compare at same threshold or use AUC |
| Optimizing recall to 100% | precision crashes | balance via F1 or business cost |
| Multiclass `precision_score` without `average` | sklearn picks "binary" → error | pass `average="macro"` or `"weighted"` |

## Self-check

- [ ] Write out the confusion matrix and label TN/FP/FN/TP.
- [ ] Define precision and recall in plain English.
- [ ] When optimize precision vs recall? Give a concrete example for each.
- [ ] What does F1 measure? When use F-beta instead?
- [ ] When do you NOT use accuracy?
- [ ] What's the difference between macro and weighted average precision?
- [ ] How do you tune the decision threshold?
- [ ] You're building a fraud classifier. Which metric is the north star?
