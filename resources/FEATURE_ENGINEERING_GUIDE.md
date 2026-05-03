# Feature Engineering — The Step-by-Step Guide

> Companion to `ML_AI_ENGINEER_WORKFLOW.md` (Stage 4).
> Once your data is clean, this is where you actually win.
> Better features beat fancier models nine times out of ten.

> Sister files:
> - [ML_AI_ENGINEER_WORKFLOW.md](ML_AI_ENGINEER_WORKFLOW.md) — full lifecycle.
> - [DATA_CLEANING_GUIDE.md](DATA_CLEANING_GUIDE.md) — the step that comes before this.

---

## What Is Feature Engineering?

Taking the columns you have and creating **new** columns the model can learn
from more easily. It's part science (statistics, math), part art (domain
knowledge), and part discipline (no leakage, reproducible pipelines).

**The mental model:** the model is lazy. The closer your features are to the
"shape" of the answer, the less the model has to do, and the better it
generalises with less data.

---

## The 12-Step Sequence (TL;DR)

```
 1. Understand the target  — what shape does the answer have?
 2. Domain brainstorm      — what would a human expert use?
 3. Numeric features       — scale, transform, bin
 4. Categorical features   — encode (one-hot, target, embedding)
 5. Datetime features      — extract, lag, days-since
 6. Text features          — TF-IDF, embeddings, statistics
 7. Geographic features    — distance, region, density
 8. Aggregations / windows — mean of last 7 days, count by user
 9. Interactions           — A × B, A / B, A - B
10. Feature selection      — drop noise, keep signal
11. Leakage check          — re-verify nothing peeks at the future
12. Pipeline + persist     — same code at train and serve time
```

---

## Step 1 — Understand the Target

Before designing features, look at the target carefully.

| Question | Why it matters |
|---|---|
| What does the model output? | A binary class? A continuous number? A ranking? |
| When is the target known? | Anything known *after* that moment is leakage. |
| What's the time horizon? | Predicting "next 30 days" needs features as of `now - 30d`. |
| Is the target rare? | Imbalance shapes which features help (precision vs recall). |
| What does a human use to decide? | That's your starting feature list. |

**Rule of thumb:** write the prediction in plain English first.
> "Given a user as of `t`, will they churn in the next 30 days?"

That sentence pins down the **prediction time `t`** — every feature must
be computable using only data with timestamp ≤ `t`.

---

## Step 2 — Domain Brainstorm

Spend 30 minutes before writing code. Ask:

- If a human expert had to make this decision, what would they look at?
- What ratios / differences matter in this domain?
  (price-to-earnings in finance, conversion rate in marketing, etc.)
- What summary statistics describe a "user" / "product" / "session"?
- What thresholds matter? ("more than 3 logins per week" might be a signal.)
- What time windows are meaningful? (last hour, day, week, month, year?)

Write the candidate features on paper. You won't use them all — but you
won't think of them once you're deep in code.

---

## Step 3 — Numeric Features

### 3.1 Scaling
Decide based on the model:

| Model family | Need scaling? |
|---|---|
| Linear / logistic regression | **Yes** |
| SVM, KNN, K-means, PCA | **Yes** |
| Neural networks | **Yes** (or BatchNorm) |
| Tree-based (XGBoost, RF, LightGBM) | **No** — irrelevant |
| Naïve Bayes | No |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

scaler = StandardScaler().fit(X_train[num_cols])   # fit on TRAIN only
X_train[num_cols] = scaler.transform(X_train[num_cols])
X_val[num_cols]   = scaler.transform(X_val[num_cols])
```

| Scaler | Formula | Use when |
|---|---|---|
| `StandardScaler` | `(x - mean) / std` | Roughly normal data |
| `MinMaxScaler` | `(x - min) / (max - min)` | Bounded scale needed (image pixels) |
| `RobustScaler` | `(x - median) / IQR` | Outliers exist |
| `MaxAbsScaler` | `x / max(|x|)` | Sparse data (preserves zeros) |

### 3.2 Transformations to fix skew

Right-skewed (income, prices, counts):
```python
import numpy as np
df["income_log"]   = np.log1p(df["income"])           # log(1+x), handles 0
df["income_sqrt"]  = np.sqrt(df["income"])
```

Box-Cox (positive only) and Yeo-Johnson (handles negatives):
```python
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method="yeo-johnson").fit(X_train[["income"]])
df["income_yj"] = pt.transform(df[["income"]])
```

### 3.3 Binning (discretisation)

Turning a continuous variable into bins. Useful when the relationship
with the target is non-linear and you want a linear model to see it.

```python
df["age_bin"] = pd.cut(df["age"], bins=[0,18,30,45,60,120],
                       labels=["teen","young","mid","senior","old"])

# Equal-frequency bins (quantile)
df["income_q5"] = pd.qcut(df["income"], q=5, labels=False, duplicates="drop")
```

### 3.4 Polynomial / interaction terms
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(X_train[["age","income"]])   # adds age², income², age*income
```

Use sparingly — it explodes feature count quickly.

### 3.5 Useful one-liners
```python
df["price_per_sqft"] = df["price"] / df["sqft"]
df["age_squared"]    = df["age"] ** 2
df["log_views"]      = np.log1p(df["views"])
df["is_premium"]     = (df["price"] > 100).astype(int)
```

---

## Step 4 — Categorical Features

### 4.1 Encoding decision tree

```
How many unique values?
├── < ~15 unique     → One-hot encoding
├── 15 - 100         → Target encoding or ordinal (if order)
├── 100 - 10,000     → Target / hashing / embedding
└── > 10,000         → Hashing or learned embedding
```

### 4.2 One-hot encoding
```python
import pandas as pd
encoded = pd.get_dummies(df["country"], prefix="country", drop_first=True)
df = pd.concat([df, encoded], axis=1)

# Or sklearn (preferred for pipelines)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X_train[["country"]])
```

`drop_first=True` avoids multi-collinearity for linear models. Trees don't care.

### 4.3 Ordinal encoding (only when order is real)
```python
from sklearn.preprocessing import OrdinalEncoder
size_order = [["S", "M", "L", "XL"]]
df["size_ord"] = OrdinalEncoder(categories=size_order).fit_transform(df[["size"]])
```

**Never** ordinal-encode unordered categories (red/blue/green) — the model
will believe red < blue < green.

### 4.4 Target / mean encoding
Replace category with the mean of the target for that category. Powerful
but **leakage-prone** — must be computed with cross-validation or smoothing.

```python
# Use the category_encoders library — handles smoothing and CV internally.
from category_encoders import TargetEncoder
te = TargetEncoder(smoothing=10).fit(X_train["product"], y_train)
X_train["product_te"] = te.transform(X_train["product"])
X_val["product_te"]   = te.transform(X_val["product"])
```

Smoothing pulls rare categories toward the global mean — prevents overfitting
to small samples.

### 4.5 Frequency / count encoding
```python
freq = X_train["city"].value_counts(normalize=True)
df["city_freq"] = df["city"].map(freq).fillna(0)
```

Cheap, often surprisingly useful.

### 4.6 Hashing trick (for very high cardinality)
```python
from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=64, input_type="string")
X_hash = fh.transform(df["url"].astype(str).map(lambda x: [x]))
```

Fast, no fitting, fixed output size — at the cost of collisions.

### 4.7 Learned embeddings
For neural networks: each category → vector of size `d`. Trained jointly with
the model. Good for very high cardinality (user IDs, product IDs, words).

---

## Step 5 — Datetime Features

Dates contain a *lot* of signal. Extract aggressively.

```python
df["signup"] = pd.to_datetime(df["signup"], utc=True)

# Calendar parts
df["year"]     = df["signup"].dt.year
df["month"]    = df["signup"].dt.month
df["day"]      = df["signup"].dt.day
df["hour"]     = df["signup"].dt.hour
df["dow"]      = df["signup"].dt.dayofweek         # Mon=0
df["quarter"]  = df["signup"].dt.quarter
df["weekofyr"] = df["signup"].dt.isocalendar().week

# Booleans
df["is_weekend"]   = df["dow"] >= 5
df["is_morning"]   = df["hour"].between(6, 11)
df["is_holiday"]   = df["signup"].dt.date.isin(holiday_set)

# Cyclical encoding (so Dec ↔ Jan are close)
import numpy as np
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["hour_sin"]  = np.sin(2 * np.pi * df["hour"]  / 24)
df["hour_cos"]  = np.cos(2 * np.pi * df["hour"]  / 24)

# Deltas (super important)
now = pd.Timestamp.utcnow()
df["days_since_signup"]      = (now - df["signup"]).dt.days
df["days_since_last_login"]  = (now - df["last_login"]).dt.days
df["days_signup_to_first_purchase"] = (df["first_purchase"] - df["signup"]).dt.days
```

**Key insight:** "days since last X" is one of the strongest features in
churn / engagement models.

---

## Step 6 — Text Features

### 6.1 Cheap statistical features (always start here)
```python
df["body_len"]     = df["body"].str.len()
df["body_words"]   = df["body"].str.split().str.len()
df["upper_ratio"]  = df["body"].str.count(r"[A-Z]") / df["body_len"]
df["digit_count"]  = df["body"].str.count(r"\d")
df["url_count"]    = df["body"].str.count(r"https?://")
df["exclaim"]      = df["body"].str.count("!")
```

### 6.2 Bag of words / TF-IDF
```python
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2),
                        min_df=5, stop_words="english")
X_text = tfidf.fit_transform(X_train["body"])
```

Works surprisingly well for classification. Pairs nicely with linear models.

### 6.3 Sentence embeddings (modern default)
```python
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("all-MiniLM-L6-v2")
emb = m.encode(df["body"].tolist(), batch_size=64, show_progress_bar=True)
# emb is (n_rows, 384) — concat with other features
```

For larger budgets: OpenAI / Anthropic / Cohere embedding APIs.

### 6.4 Topic / cluster features
- LDA / NMF for topic distributions.
- KMeans on embeddings → cluster ID feature.

---

## Step 7 — Geographic Features

```python
# Distance between two points (Haversine)
from sklearn.metrics.pairwise import haversine_distances
import numpy as np

def haversine_km(lat1, lon1, lat2, lon2):
    coords = np.radians(np.c_[lat1, lon1, lat2, lon2])
    a = haversine_distances(coords[:, :2], coords[:, 2:])
    return np.diag(a) * 6371   # earth radius in km

df["dist_to_city_km"] = haversine_km(df["lat"], df["lon"],
                                     df["city_lat"], df["city_lon"])
```

Other useful geo features:
- Country / region / postcode (treat as categorical).
- Density: number of users / events within R km.
- Cluster ID from KMeans on (lat, lon).
- Geohash (`pygeohash`) — string label of a grid cell, bin spatial regions.
- Distance to nearest landmark (airport, hospital, store).

---

## Step 8 — Aggregations & Window Features

For data with a "group" (user, product, session), summary stats over a
**time window** are often the most powerful features in the whole pipeline.

### 8.1 Group-level aggregates
```python
agg = events.groupby("user_id").agg(
    total_purchases   = ("amount", "size"),
    sum_amount        = ("amount", "sum"),
    mean_amount       = ("amount", "mean"),
    std_amount        = ("amount", "std"),
    max_amount        = ("amount", "max"),
    n_categories      = ("category", "nunique"),
    days_active       = ("date", lambda s: s.dt.date.nunique()),
).reset_index()

users = users.merge(agg, on="user_id", how="left")
```

### 8.2 Time-windowed aggregates (must respect prediction time)
```python
# Last 7 / 30 / 90 days, computed as of prediction_date
def windowed(events, t, days):
    cut = t - pd.Timedelta(days=days)
    win = events[(events["ts"] >= cut) & (events["ts"] < t)]
    return win.groupby("user_id")["amount"].agg(
        [("count", "size"), ("sum", "sum"), ("mean", "mean")]
    ).add_suffix(f"_{days}d")
```

### 8.3 Rolling windows (time series)
```python
df = df.sort_values(["user_id", "ts"])
df["amount_7d_mean"] = (
    df.groupby("user_id")["amount"]
      .rolling("7D", on="ts").mean().reset_index(level=0, drop=True)
)
df["amount_lag_1"] = df.groupby("user_id")["amount"].shift(1)
df["amount_lag_7"] = df.groupby("user_id")["amount"].shift(7)
df["amount_diff"]  = df["amount"] - df["amount_lag_1"]
```

### 8.4 Ratios that often pop
- `recent / lifetime` (e.g. spend in last 7d / spend lifetime).
- `category_count / total_count` (concentration).
- `current / max` so far.
- `current / rolling_mean` (z-score-ish).

**LEAKAGE WARNING:** every aggregation must use only data with
`timestamp < prediction_time`. Easiest way: filter events first, *then* aggregate.

---

## Step 9 — Interactions

The model can sometimes find these on its own (especially trees). But
hand-crafted interactions help linear/NN models a lot.

```python
df["price_per_room"]     = df["price"] / df["rooms"]
df["beds_to_baths"]      = df["beds"] / df["baths"].replace(0, np.nan)
df["age_x_premium"]      = df["age"] * df["is_premium"]
df["spend_to_income"]    = df["spend"] / df["income"]
df["clicks_per_session"] = df["clicks"] / df["sessions"]
```

Guard against divide-by-zero with `replace(0, np.nan)` or `+ 1`.

---

## Step 10 — Feature Selection

You usually generate too many features. Cut the noisy ones — they hurt
generalisation and slow training.

### 10.1 Filter methods (cheap, model-free)
- **Variance threshold** — drop near-constant columns.
  ```python
  from sklearn.feature_selection import VarianceThreshold
  vt = VarianceThreshold(threshold=0.0).fit(X_train)
  ```
- **Correlation with target** — Pearson (continuous), point-biserial (binary).
- **Mutual information** — captures non-linear relationships.
  ```python
  from sklearn.feature_selection import mutual_info_classif
  mi = mutual_info_classif(X_train, y_train)
  ```
- **Drop one of each highly-correlated pair** (|corr| > 0.95).

### 10.2 Wrapper methods (use a model)
- **Recursive Feature Elimination (RFE)** — fit, drop weakest, repeat.
  ```python
  from sklearn.feature_selection import RFECV
  selector = RFECV(estimator, step=1, cv=5).fit(X_train, y_train)
  ```
- **Forward / backward stepwise selection** — slow but principled.

### 10.3 Embedded methods (free during model training)
- **L1 regularisation (Lasso)** — drives weak features' weights to 0.
- **Tree feature importance** — `model.feature_importances_`.
- **SHAP values** — more reliable than raw importance, slower to compute.
  ```python
  import shap
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_val)
  ```

### 10.4 Permutation importance
Shuffle a column and measure how much performance drops. Model-agnostic, robust.
```python
from sklearn.inspection import permutation_importance
r = permutation_importance(model, X_val, y_val, n_repeats=10, random_state=0)
```

### 10.5 Rule of thumb
- < 50 features: keep them all unless clearly noisy.
- 50–500: prune by importance + drop redundant.
- > 500: aggressive selection or dimensionality reduction (PCA, UMAP for visualisation).

---

## Step 11 — Re-Verify No Leakage

Every time you add a feature, ask:

- [ ] Does this feature use only information available **at prediction time**?
- [ ] Does it use any column that's computed *from* the target?
- [ ] In time series, is every aggregation strictly over the past?
- [ ] Did I fit any encoder/scaler/imputer on the **full** dataset?
  (must be train-only)
- [ ] Are train/val/test splits respected end-to-end?

**Tell-tale leakage signs:**
- A single feature has 99% importance.
- Validation score is wildly higher than what makes business sense.
- A "date" feature is suspicious — sometimes the future is encoded in it.
- A "status" or "outcome" column slipped in.

When unsure: compute the feature **on the day before prediction** in a slow,
explicit way, and compare to your fast version. If they differ → leakage.

---

## Step 12 — Pipeline + Persistence

The biggest production bug in ML: **train and serve compute features differently**.
The fix: put every transformation inside a `Pipeline` and pickle the whole thing.

### 12.1 sklearn Pipelines + ColumnTransformer
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

num_cols = ["age", "income", "n_logins"]
cat_cols = ["country", "device"]

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("ohe",    OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

pre = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

model = Pipeline([
    ("pre",   pre),
    ("clf",   GradientBoostingClassifier()),
])

model.fit(X_train, y_train)

# Save the WHOLE thing
import joblib
joblib.dump(model, "models/v1.joblib")

# At inference time
model = joblib.load("models/v1.joblib")
pred  = model.predict_proba(new_row_df)
```

### 12.2 Custom transformers (when sklearn doesn't have it)
```python
from sklearn.base import BaseEstimator, TransformerMixin

class DateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, col): self.col = col
    def fit(self, X, y=None): return self
    def transform(self, X):
        X = X.copy()
        X[f"{self.col}_dow"]    = X[self.col].dt.dayofweek
        X[f"{self.col}_hour"]   = X[self.col].dt.hour
        X[f"{self.col}_is_wkd"] = X[self.col].dt.dayofweek >= 5
        return X.drop(columns=[self.col])
```

Then drop it into a `Pipeline` like any sklearn step.

### 12.3 Feature stores (for production at scale)
For real-time serving with the same features as training, look at:
- **Feast** — open source, popular.
- **Tecton** — managed, enterprise.
- **Vertex AI Feature Store**, **SageMaker Feature Store** — cloud-native.
- **Hopsworks** — open source, full lifecycle.

A feature store guarantees train/serve parity by being *the* source of truth.

---

## Quick Recipes Cheat-Sheet

| You have | Try first | Then try |
|---|---|---|
| Numeric, normal-ish | StandardScaler | Polynomial / interactions |
| Numeric, skewed | log1p / sqrt | Yeo-Johnson, binning |
| Numeric, heavy-tailed | RobustScaler + clip | Quantile transform |
| Low-card categorical | One-hot | Target encoding |
| High-card categorical | Frequency encoding | Target encoding, embeddings |
| Datetime | dow / hour / month | Cyclical sin/cos, days_since |
| Text (small) | TF-IDF + LR/XGB | Sentence embeddings |
| Text (large/semantic) | Sentence embeddings | Fine-tune transformer |
| Lat/Lon | Haversine distances | Geohash, cluster id |
| User × time events | groupby aggregates | 7d / 30d windows, lags |
| Heavy class imbalance | Class weights | SMOTE on train, threshold tuning |
| Too many features | Variance + correlation prune | Lasso, RFECV, SHAP |

---

## Anti-Patterns

- **Fitting transformers on the full dataset before splitting.** Leakage.
- **Target encoding without smoothing or CV.** Major leakage.
- **Random shuffling time-series data.** Future predicts past — useless model.
- **Different code at train and serve time.** Use one Pipeline object.
- **One-hot encoding a 50,000-cardinality column.** Use hashing or embeddings.
- **Not handling unseen categories at serve time.** Use `handle_unknown="ignore"`.
- **Dropping features because their importance is "low" on a single run.**
  Try permutation importance with multiple seeds.
- **Adding features without re-checking validation score.** If a feature
  doesn't help, kill it — every feature is technical debt at serve time.
- **Computing aggregations using all rows (including future).** Filter first.

---

## The 12-Step Checklist (Print This)

- [ ] 1. Target understood; prediction time `t` written down
- [ ] 2. Domain brainstorm done; candidate feature list on paper
- [ ] 3. Numeric: scaled / transformed / binned as appropriate for the model
- [ ] 4. Categorical: encoding strategy picked per column based on cardinality
- [ ] 5. Datetime: calendar parts + cyclical + days-since-X derived
- [ ] 6. Text: at least cheap statistics; embeddings if budget allows
- [ ] 7. Geographic: distances / region / density derived (if applicable)
- [ ] 8. Aggregations / windows over user / product / session, respecting `t`
- [ ] 9. Interactions / ratios added where domain suggests
- [ ] 10. Feature selection done; noise / redundant features pruned
- [ ] 11. Leakage re-checked across all new features
- [ ] 12. Everything wrapped in a single Pipeline; persisted; reproducible

When this is ticked, you're ready for model training and tuning.
