# Classification 5 — Decision Tree

## Lectures covered
- Decision Tree

---

## In one sentence
A **decision tree** is a flowchart of yes/no questions automatically learned from data — each internal question splits the data into purer groups, and each leaf gives a final prediction.

## Real-world analogy
A loan officer's mental flowchart: "Income above $80k? If yes → credit score above 700? If yes → approve. If no → ask for collateral." A decision tree is the data-driven version of that flowchart — except instead of an experienced human picking the questions, the algorithm picks the question + threshold that *best separates* the classes at each step.

## The intuition (plain English)
1. Start with all your training data at the root node.
2. For every feature, find the best yes/no question (e.g., `income > 80k?`) that splits the data into two groups that are more **pure** (more uniformly one class) than the original.
3. Recurse: do the same on each child group.
4. Stop when groups are pure enough, or when a depth/size limit hits.
5. To predict, walk a new sample down the tree — answer each question — and the leaf you land in gives the prediction (majority class for classification, average y for regression).

The cost we minimize at each split is **impurity** — Gini impurity (`1 − Σpc²`) or entropy.

## Mini worked example — should we approve a loan?

8 past applications:

```
income(k)  credit_score   approved
   30         600           no
   40         620           no
   50         670           yes
   60         700           yes
   70         710           yes
   80         500           no
   90         750           yes
   30         800           yes
```

Try splitting by `income > 55`:
```
left  (income ≤ 55):  [no, no, yes, yes]   → 50/50 → high impurity
right (income > 55):  [yes, no, yes, yes]  → 75/25 → some impurity
```

Try splitting by `credit_score > 650`:
```
left  (≤ 650):  [no, no, no]               → 100% no  → 0 impurity
right (> 650):  [yes, yes, yes, yes, yes]  → 100% yes → 0 impurity
```

Score split #2 wins — both children are perfectly pure. The tree has just one node:

```
                 credit_score > 650?
                   /          \
                 yes            no
                  │              │
              APPROVE          DENY
```

This tree is interpretable, fast, and needed no scaling.

## At-a-glance — how a tree predicts

```
                     [age > 30?]                  ← root (the algorithm picked this question)
                    /          \
                  yes            no
                  /                \
         [income > 60k?]      [credit > 700?]
            /     \              /      \
          yes     no            yes      no
          /         \           /          \
       APPROVE   REVIEW    REVIEW         DENY
```

A new applicant walks the tree from root to leaf — each branching answers one question — and lands at a prediction.

## Why this matters
- **Most interpretable model in ML.** Every prediction traces to a sequence of human-readable rules. Bankers, doctors, and regulators love trees for this.
- **No scaling needed.** Trees only care about thresholds, not magnitudes.
- **Handles non-linear, mixed-type data** out of the box.
- **Building block of ensembles.** Random Forest = many trees. XGBoost = sequential trees. Mastering trees is the foundation for the most powerful tabular models.
- **Credit-risk project** uses trees as the regulator-friendly auditable baseline.

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

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Decision tree** | A flowchart of yes/no questions learned from data |
| **Node** | One spot in the tree that asks a question |
| **Root node** | The top question (where prediction starts) |
| **Internal node** | Any non-leaf node — asks a feature/threshold question |
| **Leaf** | Terminal node — gives the prediction (class label or average y) |
| **Branch** | The yes/no edge from a node to a child |
| **Split** | The chosen feature + threshold that divides data at a node |
| **Depth** | Length of the longest root-to-leaf path |
| **Impurity** | A measure of how mixed the classes are in a node — lower is better |
| **Gini impurity** | `1 − Σpc²` — fast to compute; sklearn default |
| **Entropy** | `−Σ pc log₂(pc)` — info-theoretic alternative; gives similar trees |
| **Information gain** | How much impurity drops after a split — algorithm picks the split with the highest |
| **Pure node** | A node where all samples belong to one class (impurity = 0) |
| **`max_depth`** | Cap on tree depth — main overfitting knob |
| **`min_samples_split`** | Minimum samples required to allow a split |
| **`min_samples_leaf`** | Minimum samples required in any leaf |
| **`max_features`** | How many features to consider per split (used by Random Forest) |
| **Pre-pruning** | Limiting tree growth during construction (depth, min_samples) |
| **Post-pruning** | Building a full tree, then snipping back unimportant branches |
| **Cost-complexity pruning (`ccp_alpha`)** | Tunable post-pruning strength |
| **Feature importance** | How much each feature reduces impurity across all splits |
| **Permutation importance** | Honest alternative — shuffle a column and measure performance drop |
| **Axis-aligned split** | A tree only splits on `feature ≤ threshold` (not diagonal lines) |
| **High variance** | A small data change → very different tree — hence ensembles outperform |
| **Bagging** | Training many trees on bootstrap samples, then averaging — fights variance (next module) |
| **`DecisionTreeClassifier` / `DecisionTreeRegressor`** | sklearn's classification and regression tree classes |
| **`plot_tree`** | sklearn helper to visualize the learned tree |
| **`export_text`** | sklearn helper to print the tree as text rules |
| **Class imbalance** | When classes are uneven; trees handle it via `class_weight` |

## Further reading
- Previous: [04-naive-bayes.md](04-naive-bayes.md)
- Next: [06-class-imbalance.md](06-class-imbalance.md)
- Build on this — Random Forest: [../03-ensemble/01-bagging-random-forest.md](../03-ensemble/01-bagging-random-forest.md)
- Boosting: [../03-ensemble/02-boosting-adaboost-gbm-xgb.md](../03-ensemble/02-boosting-adaboost-gbm-xgb.md)
- Credit-risk project: [../06-projects/02-credit-risk-classification.md](../06-projects/02-credit-risk-classification.md)
