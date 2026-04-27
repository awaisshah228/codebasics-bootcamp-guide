# Classification 3 — Support Vector Machine (SVM)

## Lectures covered
- Support Vector Machine (SVM)
- Data Pre-processing: Scaling (covered alongside SVM since it requires it)
- Sklearn Pipeline (recap, used heavily with SVM)

---

## 1. The geometric intuition

SVM finds the **hyperplane** that separates two classes with the **maximum margin** — the largest gap on either side.

```
class A points    │  margin   │   class B points
   • • •          │           │      ◯ ◯
  •   •           │   ────────│      ◯  ◯
   •              │  /        │       ◯
                  │ /         │
                  │/  decision boundary
```

The **support vectors** are the few training points closest to the boundary — they're what define it. Move other points around and the boundary doesn't budge.

---

## 2. Hard-margin vs soft-margin

### Hard margin
Perfect separation required. Fragile — one outlier breaks it.

### Soft margin (used in practice)
Allows some violations, controlled by hyperparameter **C**:
- Large C → strict margin, few violations, risk overfit
- Small C → looser margin, more violations, risk underfit

```python
from sklearn.svm import SVC
SVC(C=1.0)
```

`C` here is the regularization knob (inversely — bigger C = less regularization).

---

## 3. The kernel trick — non-linear boundaries

Linear SVM only finds straight boundaries. Real data often isn't linearly separable.

**Kernels** implicitly map data into a higher-dimensional space where it *is* linearly separable, without ever explicitly computing the higher-dim coordinates.

| Kernel | Formula | When use |
|---|---|---|
| **linear** | $K(x, x') = x \cdot x'$ | linearly separable; high-dim sparse data (text) |
| **poly** | $K(x, x') = (\gamma x \cdot x' + r)^d$ | polynomial decision boundaries |
| **rbf** | $K(x, x') = \exp(-\gamma \|x - x'\|^2)$ | the default; very flexible |
| **sigmoid** | tanh-based | rarely worth it |

```python
from sklearn.svm import SVC

# default: RBF kernel
SVC(kernel="rbf", C=1.0, gamma="scale")

# linear (faster on large/sparse data)
SVC(kernel="linear")

# polynomial degree 3
SVC(kernel="poly", degree=3)
```

---

## 4. The two key hyperparameters: C and gamma

### C (regularization)
- Small C: wider margin, smoother boundary, may underfit
- Large C: narrow margin, complex boundary, may overfit

### gamma (RBF / poly only)
- Controls how far one example's influence reaches
- Small gamma: smooth boundaries, simple
- Large gamma: tight wiggly boundaries, overfits

### Tuning grid
```python
from sklearn.model_selection import GridSearchCV
param_grid = {
    "svc__C": [0.1, 1, 10, 100],
    "svc__gamma": [0.001, 0.01, 0.1, "scale"],
}
gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_macro")
gs.fit(X_train, y_train)
```

---

## 5. Always scale features for SVM

SVM is distance-based — features on different scales dominate. Always pipe through `StandardScaler`.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svc",   SVC(kernel="rbf", C=1.0, gamma="scale")),
])
pipe.fit(X_train, y_train)
```

This is one of the most common SVM mistakes — forgetting to scale.

---

## 6. SVM for regression (SVR)

Same idea, applied to regression: find a hyperplane within an ε-tube around the data.

```python
from sklearn.svm import SVR
SVR(kernel="rbf", C=1.0, epsilon=0.1)
```

Useful when relationships are non-linear and you want a smooth predictor.

---

## 7. Strengths and weaknesses

### Strengths
- Very effective in **high-dimensional** spaces (text classification!)
- Memory-efficient (only support vectors matter)
- Versatile via kernels
- Works well with limited data

### Weaknesses
- **Slow on large datasets** (n > 100k) — training scales between $O(n^2)$ and $O(n^3)$
- Doesn't natively output probabilities (need `probability=True`, which is slow)
- Hyperparameter tuning is trickier than tree-based models
- Less interpretable than logistic regression

---

## 8. Probability outputs (when needed)

```python
SVC(kernel="rbf", probability=True)        # adds Platt scaling — slower training
```

Or use `LinearSVC` for big linear problems (much faster, no built-in probabilities) and calibrate afterwards.

---

## 9. Multiclass SVM

`SVC` does One-vs-One internally for multiclass:
- Trains $\binom{k}{2}$ binary SVMs
- Each pair votes; majority wins

For very large multiclass: prefer `LinearSVC` with OvR (faster).

---

## 10. Real example — iris dataset (classic SVM demo)

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = datasets.load_iris(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

pipe = Pipeline([
    ("scale", StandardScaler()),
    ("svc",   SVC(kernel="rbf", C=1.0, gamma="scale")),
])
pipe.fit(X_tr, y_tr)
print(classification_report(y_te, pipe.predict(X_te)))
```

Typical accuracy: 97%+ on iris.

---

## 11. When to reach for SVM

- Text classification (linear SVM on TF-IDF) — historic gold standard before deep NLP
- Small-to-medium structured data with non-linear patterns
- High-dimensional, low-sample-count problems

When NOT to: huge datasets, when you want probabilities natively (use logistic), when you want feature importance (use tree models).

---

## 12. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to scale | bad accuracy, slow training | always pipe with StandardScaler |
| Default RBF on huge dataset | training takes hours | switch to LinearSVC, or sample down |
| `probability=True` always | slows training a lot | only set when probabilities needed |
| Ignoring class imbalance | poor minority recall | `class_weight="balanced"` |
| C and gamma both maxed | overfit, decision boundary noise | tune via grid search |

## Self-check

- [ ] What's a "support vector"?
- [ ] What's the kernel trick? Why does it help?
- [ ] When use linear vs RBF kernel?
- [ ] Difference between C and gamma — what does each control?
- [ ] Why is scaling especially important for SVM?
- [ ] Which kernel would you try first for text classification?
- [ ] Why is SVM slow on big data?
- [ ] Walk through SVM's geometric idea on a 2D dataset.
