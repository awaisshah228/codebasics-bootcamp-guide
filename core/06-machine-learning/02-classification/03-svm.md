# Classification 3 — Support Vector Machine (SVM)

## Lectures covered
- Support Vector Machine (SVM)
- Data Pre-processing: Scaling (covered alongside SVM since it requires it)
- Sklearn Pipeline (recap, used heavily with SVM)

---

## In one sentence
**Support Vector Machine (SVM)** draws the *widest possible alley* between two classes — the farther apart the two crowd lines, the more confident future predictions can be.

## Real-world analogy
Imagine two rival fan groups arriving at a stadium and you have to place a barrier between them. You wouldn't put it right next to one group — you'd push it as far from *both* as possible so a small movement doesn't cause a fight. SVM does exactly that: it finds the boundary that maximizes the gap (the **margin**) to the nearest fans on each side. Those nearest fans are called **support vectors**.

For groups whose territories swirl together in messy shapes, you can't draw a straight barrier — but you can imagine lifting them onto a 3D dome where the messy shapes magically separate. That's the **kernel trick**.

## The intuition (plain English)
1. Training data has two classes. SVM picks the decision boundary that's *farthest* from the closest points of each class — maximum margin.
2. Only the points right next to the boundary (**support vectors**) matter. Move other points around and the line doesn't budge — making SVM memory-efficient.
3. With **soft margin** (parameter `C`), SVM allows some violations to keep the boundary smooth — high C is strict (overfit risk); low C is forgiving (underfit risk).
4. With **kernels** (RBF, polynomial), SVM can carve curved boundaries — math handles the curvature without you ever computing the curved coordinates.

## Mini worked example — picking the boundary on 6 points

```
class A (●):  (1, 1), (2, 1), (1.5, 0.5)
class B (○):  (4, 4), (5, 4), (4.5, 5)

Two candidate lines through the gap:
  Line L1: x + y = 4    -- closest points are 0.7 from each side
  Line L2: x + y = 3.5  -- closest A point 0.35 away, closest B 1.0 away   (off-center)

SVM picks L1: max-margin = 0.7 + 0.7 = 1.4 (vs 0.35 + 1.0 = 1.35 for L2).
Support vectors: (2, 1) on the A side, (4, 4) on the B side. Other points don't affect L1.
```

If you now add a 7th point at (10, 10), the boundary doesn't change — only the support vectors do.

## At-a-glance — the SVM idea

```
class A      |  margin   |   class B
   ● ●       |           |        ○ ○
  ●  ●       |   ─────── │       ○  ○        ← only the dots ON the dotted lines
   ●         |  / decision        ○             are "support vectors"
             | / boundary
             |/
support      ↑ widest gap        support
vectors    (no points in here)   vectors
```

For non-linear data, the **kernel trick** maps points into a higher-dim space:
```
Original 2D (impossible to separate with a line):
   ○  ●  ○  ●  ○

Lifted into 3D via RBF kernel — now a flat plane separates them.
```

## Why this matters
- **Strong on small/medium datasets** with non-linear patterns. The historic best classifier before deep learning.
- **Memory-efficient at predict time** — only support vectors are stored, not the full training set.
- **Text classification**: linear SVM on TF-IDF features was the gold-standard NLP baseline for a decade.
- **Always pipe with StandardScaler** — SVM is distance-based and goes haywire on unscaled data.

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

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **SVM (Support Vector Machine)** | Classifier that maximizes the margin between classes |
| **Hyperplane** | An (n−1)-dimensional flat boundary in n-dim space (a line in 2D, plane in 3D) |
| **Margin** | Distance from the decision boundary to the closest points |
| **Support vectors** | The few training points closest to the boundary — they alone define it |
| **Hard margin** | Demands perfect separation — fragile, breaks on outliers |
| **Soft margin** | Allows some violations; controlled by `C` |
| **C (regularization)** | Penalty for margin violations. Big C = strict (overfit risk); small C = relaxed (underfit risk) |
| **Kernel** | A function that lets SVM behave like it's in a higher-dim space, without computing those coordinates |
| **Kernel trick** | The math identity that makes kernels possible — only inner products are needed |
| **Linear kernel** | `K(x, x') = x · x'` — straight boundary; great for high-dim sparse data (text) |
| **RBF kernel** | `K(x, x') = exp(−γ‖x − x'‖²)` — flexible curves; the default |
| **Polynomial kernel** | `(γ x · x' + r)^d` — polynomial decision boundaries |
| **gamma (γ)** | RBF/poly kernel parameter; small = smooth boundary, big = wiggly |
| **Decision boundary** | Where the model predicts 0.5 / switches class |
| **`SVC`** | sklearn's support vector *classifier* |
| **`SVR`** | sklearn's support vector *regressor* — fits a hyperplane within ε of points |
| **`LinearSVC`** | Fast linear-only SVM — no kernel trick, scales to bigger data |
| **Platt scaling** | Post-hoc method to turn SVM scores into probabilities (`probability=True`) |
| **One-vs-One (OvO)** | Multiclass strategy SVC uses — train a binary SVM for every class pair |
| **One-vs-Rest (OvR)** | Train one binary SVM per class against all others |
| **Standardization** | Scaling features to mean 0, std 1 — essential for SVM |
| **TF-IDF** | Numeric representation of text — combines well with linear SVM |
| **`class_weight="balanced"`** | Reweights classes to handle imbalance |
| **GridSearchCV** | Exhaustive hyperparameter search — used for tuning C and gamma |

## Further reading
- Previous: [02-classification-metrics.md](02-classification-metrics.md) — how you'll evaluate the SVM
- Next: [04-naive-bayes.md](04-naive-bayes.md)
- Scaling deep-dive: [../01-foundations/05-preprocessing-encoding.md](../01-foundations/05-preprocessing-encoding.md)
- Tuning C and gamma: [../03-ensemble/03-cross-validation-tuning.md](../03-ensemble/03-cross-validation-tuning.md)
