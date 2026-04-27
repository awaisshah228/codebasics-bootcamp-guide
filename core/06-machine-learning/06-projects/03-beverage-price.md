# Project — Beverage Price Range Prediction

## Domain
A beverage company wants to **price new SKUs** before launch. They have a survey dataset where consumers indicate willingness-to-pay across price tiers. ML predicts the **price range bucket** (low / mid / high / premium) a new SKU should target — avoiding both under-pricing (lost revenue) and over-pricing (low sell-through).

## ML formulation
- **Type**: multi-class classification (4 price buckets)
- **Features**: product attributes (size, flavor, sugar level), consumer demographics, region, brand strength, ingredient cost, competitor benchmarks
- **Metric**: macro-F1 (each price bucket equally weighted)

---

## Why this project is unusual

It's **survey-driven**, so:
- Data is small relative to "industry" projects
- Heavy in categorical features
- Class imbalance is moderate but not extreme
- Insights matter as much as accuracy

---

## Walkthrough

### 1. EDA
```python
import pandas as pd, seaborn as sns
df = pd.read_csv("beverage_survey.csv")

print(df["price_range"].value_counts(normalize=True))
sns.countplot(x="price_range", data=df)

# explore feature distributions per bucket
for col in df.select_dtypes(include="object"):
    pd.crosstab(df[col], df["price_range"], normalize="index").plot(kind="bar", stacked=True)
```

Look for:
- Which features differ most across price buckets
- Demographic effects (do younger consumers prefer the budget tier?)
- Brand strength's role

### 2. Cleaning + feature engineering
- One-hot encode categorical features (with `min_frequency` if cardinality is high)
- Scale numeric features
- Engineer features:
  - `cost_to_size_ratio` (ingredient cost / pack size)
  - `brand_index` (brand recognition score from another survey)
  - `competitor_gap` (price difference from nearest competitor)
- Encode the target as 0/1/2/3 (ordinal — preserve order)

### 3. Modeling — try several
```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

models = {
    "logreg":  LogisticRegression(max_iter=1000, multi_class="multinomial"),
    "rf":      RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced"),
    "xgboost": XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, n_jobs=-1),
}

for name, m in models.items():
    score = cross_val_score(m, X, y, cv=5, scoring="f1_macro").mean()
    print(f"{name}: macro-F1 = {score:.3f}")
```

For multi-class with ordinal target, you can also try **ordinal regression** approaches (e.g., `mord` package) — but in practice multi-class with the right metric works fine.

### 4. Tuning + final model
Use Optuna for the winner; report on held-out test set.

### 5. Multiclass evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=["low", "mid", "high", "premium"]))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
```

Pay attention to **off-diagonal patterns** — is the model confusing "mid" and "high" but never "low" and "premium"? That's actually fine business-wise (close mistakes hurt less than swapping ends).

### 6. Feature importance + SHAP
Same pattern as credit risk. Surface the **top 5 drivers of price range** to the marketing team.

### 7. Deliverable for stakeholders
A 1-pager showing:
- Top 5 product attributes that drive each price range
- Confusion matrix → "we rarely confuse 'low' with 'premium'; we sometimes confuse adjacent tiers"
- Recommendation for new SKU X based on its attributes → "model predicts mid (P=0.62), with a 28% chance of high"
- ROI estimate vs current "expert intuition" pricing

### 8. Streamlit demo
```python
import streamlit as st
import pandas as pd, joblib

model = joblib.load("model.joblib")

st.title("Beverage Price Range Predictor")

# inputs
flavor = st.selectbox("Flavor", ["cola", "lemon", "mango", "energy"])
size_ml = st.slider("Size (ml)", 200, 1000, 500)
sugar = st.slider("Sugar (g)", 0, 50, 10)
brand_strength = st.slider("Brand strength (1-10)", 1, 10, 5)

if st.button("Predict"):
    X_new = pd.DataFrame([{
        "flavor": flavor, "size_ml": size_ml,
        "sugar": sugar, "brand_strength": brand_strength,
    }])
    probs = model.predict_proba(X_new)[0]
    for label, p in zip(["low", "mid", "high", "premium"], probs):
        st.write(f"{label}: {p:.0%}")
```

---

## Repo deliverables
```
beverage-price/
├── data/
├── notebooks/
│   ├── 01-eda.ipynb
│   ├── 02-feature-eng.ipynb
│   ├── 03-modeling.ipynb
├── src/
│   ├── train.py
│   └── app.py
├── reports/
│   └── insights-for-marketing.md
├── model.joblib
└── README.md
```

---

## Self-check

- [ ] Did I report macro-F1, not just accuracy?
- [ ] Did I analyze the confusion matrix's off-diagonal pattern?
- [ ] Did I surface SHAP-driven insights for marketing?
- [ ] Is my Streamlit demo intuitive enough for a non-DS to use?
- [ ] Did I quantify business impact ("would have priced 80% of SKUs in the right bucket vs the team's 60%")?
- [ ] Did I post a build-in-public update?
