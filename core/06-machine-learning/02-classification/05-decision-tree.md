# Classification 5 — Decision Tree

## Lectures covered
- Decision Tree

---

## 1. The intuition

A decision tree is a flowchart of yes/no questions. Each internal node asks about a feature; each leaf gives a prediction.

```
                 income > 80k?
                 /        \
              yes          no
              /              \
        credit > 700?      age > 50?
        /     \           /     \
       Y       N         Y       N
       │       │         │       │
     APPROVE  REVIEW   REVIEW  DENY
```

Trees are **the** poster child for interpretable ML.

---

## 2. How splits are chosen

At each node, the algorithm picks the feature + threshold that **best separates** the classes (for classification) or **most reduces variance** (for regression).

### For classification — two impurity measures

#### Gini impurity
$$\text{Gini} = 1 - \sum_c p_c^2$$
Lower is purer. Range 0 (perfectly pure) to ~0.5 (50/50).

#### Entropy
$$\text{Entropy} = -\sum_c p_c \log_2 p_c$$
Information-theoretic. 0 = pure, log(k) = uniform.

In practice both give similar trees. **Gini is faster** (no log) and is sklearn's default.

### Information gain
$$IG = \text{impurity}_{\text{parent}} - \sum \frac{n_{\text{child}}}{n_{\text{parent}}} \cdot \text{impurity}_{\text{child}}$$

The split with highest IG wins.

---

## 3. sklearn shape

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
clf.fit(X_train, y_train)
```

For regression: `DecisionTreeRegressor` with `criterion="squared_error"`.

---

## 4. Visualizing a tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=X.columns, class_names=["no", "yes"], filled=True, fontsize=8)
plt.show()
```

Or text export:
```python
from sklearn.tree import export_text
print(export_text(clf, feature_names=list(X.columns)))
```

---

## 5. Hyperparameters that matter

| Param | Effect |
|---|---|
| `max_depth` | max levels — primary overfitting knob |
| `min_samples_split` | min rows to allow a split (prevents tiny-leaf overfit) |
| `min_samples_leaf` | min rows in any leaf |
| `max_features` | how many features each split considers |
| `class_weight="balanced"` | reweights for imbalanced classes |
| `ccp_alpha` | cost-complexity pruning strength |

### Classic overfitting fix
Cap `max_depth` at 3–10 for trees you'll inspect by eye. Beyond that, you usually want a Random Forest.

---

## 6. Strengths

- **Interpretable**: every prediction traces back to a sequence of yes/no questions
- **No scaling needed**: trees are invariant to monotonic transforms
- Handles **mixed types** naturally (numeric + categorical, with encoding)
- **Captures non-linear** patterns and interactions
- Robust to outliers in features (just affects one branch)

## 7. Weaknesses

- **High variance**: small data changes → very different trees
- **Overfits** quickly without pruning / depth limit
- Single trees rarely top leaderboards — beaten by ensembles (RF, GBM)
- Diagonal decision boundaries can't be expressed (only axis-aligned splits)

---

## 8. Pruning — taming overfitting

### Pre-pruning (limits during construction)
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

### Post-pruning (cost-complexity / "alpha pruning")
Build the full tree, then snip back branches that don't justify their complexity:

```python
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

# train trees with different alphas, pick best by CV
```

---

## 9. Feature importance

Trees compute importance as how much each feature contributes to impurity reduction across all splits.

```python
import pandas as pd
fi = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(fi.head(10))
```

> **Caveat**: importance is biased toward high-cardinality features. For a more honest version, use `permutation_importance`.

---

## 10. Real example — credit risk

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

clf = DecisionTreeClassifier(
    max_depth=6,
    min_samples_leaf=50,
    class_weight="balanced",
    random_state=42,
)
clf.fit(X_train, y_train)
print(classification_report(y_test, clf.predict(X_test)))
```

A small interpretable tree like this is exactly the kind of model regulators love (banking, insurance) because every decision is auditable.

---

## 11. Decision Tree vs Logistic Regression vs SVM (the comparison)

| | Decision Tree | Logistic Regression | SVM |
|---|---|---|---|
| Interpretable | very | yes | not really |
| Needs scaling | no | yes | yes |
| Captures non-linearity | yes | no (without engineering) | yes (with kernels) |
| Probabilistic | OK | excellent | weak |
| Speed on large data | medium | fast | slow (RBF) |
| Best baseline | for tabular non-linear | for tabular linear | for high-dim |

In modern practice: **start with logistic regression as baseline, then go straight to gradient boosting (XGBoost / LightGBM)** for tabular wins.

Decision trees sit between them as the *educational* stepping stone and the building block of ensembles.

---

## 12. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Tree depth not capped | massive overfit | always set `max_depth` for vanilla trees |
| Reading `feature_importance_` literally | biased toward high-cardinality features | use `permutation_importance` |
| Tree fitted but not visualized | no interpretability gain over RF | always inspect at depth ≤ 5 |
| Using a single tree for production | unstable | use Random Forest / Gradient Boosting |

## Self-check

- [ ] What does Gini impurity measure?
- [ ] How does the algorithm pick a split?
- [ ] Why don't decision trees need feature scaling?
- [ ] What's the difference between pre-pruning and post-pruning?
- [ ] Write code to display a tree with depth ≤ 4.
- [ ] What's the bias problem in `feature_importances_`?
- [ ] Why is a single tree usually beaten by random forest?
- [ ] When would you specifically choose a single decision tree over an ensemble?
