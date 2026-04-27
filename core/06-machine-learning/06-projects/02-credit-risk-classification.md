# Project — Credit Risk Modeling (Classification)

## Domain
A **non-banking financial company (NBFC)** issues loans. They want a model that scores each new applicant's **probability of default** to drive lend / decline / refer-to-human decisions.

This is one of the most common and well-paid ML use cases in financial services.

## ML formulation
- **Type**: binary classification
- **Target**: `defaulted` (0 = paid back, 1 = defaulted)
- **Class imbalance**: typically 5–15% defaulters
- **Metrics**: ROC-AUC, PR-AUC, **KS statistic**, **Gini coefficient**, recall at fixed precision

---

## Domain-specific concepts

### KS Statistic (Kolmogorov-Smirnov)
Maximum difference between cumulative distribution of good vs bad applicants across the score range.
- KS > 0.4: strong model
- KS > 0.5: very strong
- KS < 0.3: weak

```python
from scipy.stats import ks_2samp
ks = ks_2samp(y_prob[y_test==1], y_prob[y_test==0]).statistic
```

### Gini coefficient
$$\text{Gini} = 2 \cdot \text{AUC} - 1$$

- Gini > 0.4 reasonable; > 0.6 very strong
- Industry-standard metric for credit models

### WOE / IV — Weight of Evidence and Information Value
Pre-modeling feature transformation widely used in credit risk:
$$\text{WOE} = \ln\left(\frac{P(\text{good}|x)}{P(\text{bad}|x)}\right)$$
$$\text{IV} = \sum (P(\text{good}|x) - P(\text{bad}|x)) \cdot \text{WOE}$$

| IV | Predictive power |
|---|---|
| < 0.02 | useless |
| 0.02–0.1 | weak |
| 0.1–0.3 | medium |
| 0.3–0.5 | strong |
| > 0.5 | suspicious — check for leakage |

```bash
pip install xverse  # or compute manually
```

WOE-encoded features feed cleanly into logistic regression and produce interpretable, monotonic coefficients — regulators love this.

---

## Walkthrough

### 1. Data understanding
Typical fields: age, income, employment length, debt-to-income ratio, credit utilization, # of past inquiries, prior defaults, loan amount, term, purpose.

```python
import pandas as pd
df = pd.read_csv("loans.csv")
print(df["defaulted"].value_counts(normalize=True))     # 90/10 imbalance typical
```

### 2. EDA + outlier treatment
- Income → IQR outlier treatment (right-skewed)
- Debt-to-income > 1 → flag (outlier or borrower in trouble)
- Defaulter rate by employment length, by debt bucket, by credit score bucket

### 3. Feature engineering with WOE/IV
```python
import xverse.transformer as xv

clf = xv.WOE()
X_woe = clf.fit_transform(X, y)
iv = clf.iv_df          # IV per feature

X_filtered = X_woe[iv.query("Information_Value > 0.05")["Variable_Name"]]
```

### 4. Modeling pipeline
```python
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ratio = (y_train == 0).sum() / (y_train == 1).sum()

xgb_clf = xgb.XGBClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    max_depth=4,                # shallow trees for stability
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=ratio,
    eval_metric="auc",
    early_stopping_rounds=100,
    random_state=42,
)
```

Often a calibrated logistic regression on WOE features performs nearly as well — and is far more interpretable. Banks frequently ship that for regulatory reasons.

### 5. Hyperparameter tuning (Optuna)
```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 2000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    clf = xgb.XGBClassifier(**params, scale_pos_weight=ratio, random_state=42)
    return cross_val_score(clf, X_train, y_train, cv=5, scoring="roc_auc").mean()

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

### 6. Evaluation suite
```python
from sklearn.metrics import roc_auc_score, classification_report
from scipy.stats import ks_2samp

y_prob = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_prob)
gini = 2 * auc - 1
ks = ks_2samp(y_prob[y_test == 1], y_prob[y_test == 0]).statistic
print(f"AUC: {auc:.3f}  |  Gini: {gini:.3f}  |  KS: {ks:.3f}")
print(classification_report(y_test, model.predict(X_test)))
```

### 7. Calibration + threshold setting
Often the business wants:
- "Approve top 60% by score" → set threshold at the 40th percentile
- "Maintain ≥ 90% precision on rejections" → use precision-recall curve

```python
y_prob_sorted = np.sort(y_prob)[::-1]
threshold = y_prob_sorted[int(len(y_prob) * 0.4)]
```

### 8. SHAP explanations (regulatory requirement)
```python
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# global: which features matter most
shap.summary_plot(shap_values, X_test)

# per-prediction: why was this customer rejected?
shap.force_plot(explainer.expected_value, shap_values[i], X_test.iloc[i])
```

For banking, **per-decision explanations** are often legally required.

### 9. Streamlit deployment
Same pattern as the healthcare project — form inputs, model.predict, present score with explanation.

### 10. Business presentation
The deliverable should include:
- Confusion matrix at the chosen threshold
- KS / Gini / AUC vs current production
- Expected $ impact (defaults caught × avg loss saved)
- Subgroup fairness check (does the model perform equally on age, gender, region?)

---

## Repo structure
```
credit-risk/
├── data/
│   ├── loans.csv
│   └── data_dictionary.md
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-woe-iv.ipynb
│   ├── 03-baseline-logreg.ipynb
│   ├── 04-xgboost.ipynb
│   ├── 05-tuning-optuna.ipynb
│   └── 06-shap-explanations.ipynb
├── src/
│   ├── train.py
│   ├── score.py
│   └── app.py            # Streamlit
├── reports/
│   ├── model-card.md
│   └── business-deck.pdf
├── model.joblib
├── requirements.txt
└── README.md
```

---

## Self-check

- [ ] Did I report KS, Gini, AUC, PR-AUC?
- [ ] Did I produce per-prediction SHAP explanations?
- [ ] Is my threshold business-driven, not arbitrary?
- [ ] Did I check fairness across age / gender / region?
- [ ] Is my README a defensible business case (impact in $)?
- [ ] Did I post a LinkedIn writeup with a Streamlit demo link?
- [ ] Does my repo include a model card?
