# Data Cleaning — The Step-by-Step Sequence

> Companion to `ML_AI_ENGINEER_WORKFLOW.md` (Stage 3).
> Once you have raw data in hand, run through these steps **in order**.
> Each step says: what to check, how to detect it, and how to treat it.

You will spend more time here than anywhere else. The good news: this same
sequence works for tabular data 90% of the time. Image / text / audio have
their own variants — see the bottom of this file.

---

## The 14-Step Sequence (TL;DR)

```
 1. First look          — shape, types, memory, sample
 2. Schema validation   — does it match what you expected?
 3. Data types          — fix dtypes BEFORE anything else
 4. Missing values      — count, pattern, treatment
 5. Duplicates          — exact + near
 6. String / category   — whitespace, case, spelling
 7. Outliers            — detect, investigate, treat
 8. Range / constraints — domain rules (age ≥ 0, etc.)
 9. Date / time         — parse, timezone, sanity-check
10. Numerical issues    — skew, scale, transformations
11. Categorical issues  — high cardinality, rare classes
12. Target variable     — imbalance + leakage
13. Feature leakage     — drop look-ahead columns
14. Final validation    — schema test, save clean copy
```

Do them roughly in this order — fixing dtypes first makes every later step
easier; treating missing values before outliers stops you from "imputing"
values you'd then flag as outliers; etc.

---

## Step 1 — First Look (5 minutes)

Before anything else, just *look* at the data.

```python
import pandas as pd

df = pd.read_csv("raw.csv")

df.shape              # (rows, cols) — sanity check
df.head(10)           # what does a row look like?
df.tail(10)           # different from head? sometimes the tail is junk
df.sample(10)         # random rows reveal hidden mess
df.info()             # dtypes + non-null counts + memory
df.describe(include="all")   # numeric + categorical summary
df.columns.tolist()   # any weird column names? trailing spaces?
df.memory_usage(deep=True).sum() / 1e6   # how big is this thing?
```

**Things to notice:**
- Are column names clean? (no `Unnamed: 0`, no spaces, consistent case)
- Are there obvious junk rows at the top/bottom (totals, headers repeated)?
- Are dtypes what you expect? (numbers stored as object = a problem)
- Is the row count what you were promised?

**Treatment:**
- Rename columns to `snake_case`: `df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")`
- Drop junk header/footer rows: `df = df.iloc[1:-2]` (after inspection!).

---

## Step 2 — Schema Validation

Write down what you *expect* the data to look like, then assert it.

```python
import pandera as pa
from pandera import Column, Check

schema = pa.DataFrameSchema({
    "user_id": Column(str, nullable=False, unique=True),
    "age":     Column(int, Check.in_range(0, 120), nullable=True),
    "country": Column(str, Check.isin(["US", "UK", "IN", "PK"])),
    "signup":  Column(pa.DateTime, Check.le(pd.Timestamp.now())),
    "spend":   Column(float, Check.ge(0)),
})

schema.validate(df, lazy=True)   # collects all errors, doesn't stop on first
```

**Why now, before cleaning?**
You want to *know* what's broken. If you clean silently, you'll never
notice the upstream pipeline started sending negative ages.

**Treatment:**
- Fix the violations (next steps), or
- Push back to upstream — the producer of the data should stop emitting it.

---

## Step 3 — Data Types

Wrong dtypes are the #1 cause of weird bugs. Fix them **before** anything else.

| Symptom | Likely cause | Fix |
|---|---|---|
| Numeric column is `object` | Has `"NA"`, `"-"`, `"$1,200"`, or commas | `pd.to_numeric(col, errors="coerce")` after stripping symbols |
| Date column is `object` | Strings, mixed formats | `pd.to_datetime(col, errors="coerce", utc=True)` |
| Boolean is `object` | `"Yes"/"No"`, `"true"/"True"`, `1/0` | Map explicitly: `{"Yes": True, "No": False}` |
| Categorical is `object` and large | Many repeats | `df["x"] = df["x"].astype("category")` saves memory |
| ID is `int` | Risk of losing leading zeros | Cast to `str` |

```python
df["price"] = (
    df["price"].astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)
df["signup"] = pd.to_datetime(df["signup"], errors="coerce", utc=True)
df["is_active"] = df["is_active"].map({"Yes": True, "No": False})
```

**Tip:** `errors="coerce"` turns un-parseable values into `NaN/NaT` —
which means the missing-value step picks them up. That's intentional.

---

## Step 4 — Missing Values

### 4.1 Detect
```python
df.isna().sum().sort_values(ascending=False)
df.isna().mean().sort_values(ascending=False)   # as a fraction
```

Visualise patterns:
```python
import missingno as msno
msno.matrix(df)     # are missings random or correlated across columns?
msno.heatmap(df)    # which columns go missing together?
```

### 4.2 Understand the *type* of missingness
- **MCAR — Missing Completely At Random.** No pattern. Safe to drop or impute.
- **MAR — Missing At Random.** Depends on other observed columns
  (e.g. women skip "salary" more often). Imputation works if you condition
  on those columns.
- **MNAR — Missing Not At Random.** Missingness depends on the missing
  value itself (e.g. high earners hide salary). Imputing biases the result —
  consider modelling missingness explicitly with an "is_missing" flag.

### 4.3 Treatment options

| Strategy | When | Code |
|---|---|---|
| **Drop rows** | Missingness < 5% AND MCAR | `df.dropna(subset=["x"])` |
| **Drop column** | Column is > 60-70% missing AND not critical | `df.drop(columns=["x"])` |
| **Constant fill** | "missing" is meaningful (e.g. "no_phone_number") | `df["x"].fillna("UNKNOWN")` |
| **Mean / median** | Numeric, low skew | `df["x"].fillna(df["x"].median())` |
| **Mode** | Categorical | `df["x"].fillna(df["x"].mode().iloc[0])` |
| **Forward / back fill** | Time series | `df["x"].ffill()` |
| **Group-wise** | Different groups have different means | `df.groupby("country")["x"].transform(lambda s: s.fillna(s.median()))` |
| **Model-based** | Lots of correlated features | `sklearn.impute.IterativeImputer` or KNN |
| **Add "is missing" flag** | MNAR — missingness itself is a signal | `df["x_missing"] = df["x"].isna(); df["x"].fillna(0)` |

**Always:** fit imputers on **train** only, then apply to val/test.
Use `sklearn.impute.SimpleImputer` inside a `Pipeline` so this happens for free.

**Anti-patterns:**
- Don't `fillna(0)` on a numeric column without thinking — 0 is a real value
  that the model will learn from.
- Don't impute the **target** variable. Drop those rows.

---

## Step 5 — Duplicates

### 5.1 Exact duplicates
```python
df.duplicated().sum()
df = df.drop_duplicates()
```

### 5.2 Duplicates ignoring some columns
You often want to dedupe by business key, ignoring `created_at`:
```python
df = df.drop_duplicates(subset=["user_id", "event_id"], keep="last")
```

### 5.3 Near duplicates (fuzzy)
- Strings: `rapidfuzz`, `recordlinkage`, `dedupe` library.
- Records that look the same with different whitespace / case:
  normalise first (Step 6), then `drop_duplicates`.

### 5.4 Treatment
- Investigate first — duplicates can be a **bug** in upstream ingestion.
  If you silently drop them, you won't see the bug.
- For aggregation tables: `groupby` + `agg` is often safer than
  `drop_duplicates`.

---

## Step 6 — String / Categorical Cleanup

This is where small bugs hide. Two strings that look identical to the human
eye are different bytes to Python.

### 6.1 Standard cleanup pipeline
```python
def clean_text_col(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.strip()                        # leading/trailing whitespace
         .str.replace(r"\s+", " ", regex=True)  # collapse multiple spaces
         .str.lower()                        # consistent case (if appropriate)
         .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA, "null": pd.NA})
    )

for c in ["country", "city", "category"]:
    df[c] = clean_text_col(df[c])
```

### 6.2 Common gotchas
- **Smart quotes** — `"` vs `"` vs `"` are different bytes.
- **Unicode lookalikes** — Cyrillic `а` ≠ Latin `a`. Use `unicodedata.normalize("NFKD", s)`.
- **Non-printable chars** — strip with `s.str.replace(r"[\x00-\x1f]", "", regex=True)`.
- **Consistent vocab** — `"USA"`, `"U.S."`, `"United States"`, `"us"` are all the same country.
  Maintain a mapping dict and apply it.

### 6.3 Spelling / fuzzy matching
For free-text categorical (city names, product names):
```python
from rapidfuzz import process
canonical = ["New York", "Los Angeles", "Chicago"]
df["city_clean"] = df["city"].apply(
    lambda x: process.extractOne(x, canonical, score_cutoff=85)[0] if pd.notna(x) else pd.NA
)
```

---

## Step 7 — Outliers

### 7.1 Detect
| Method | Code | Best for |
|---|---|---|
| Histogram / box plot | `df["x"].hist()`, `df.boxplot("x")` | Eyeballing |
| IQR rule | `Q1 - 1.5*IQR`, `Q3 + 1.5*IQR` | Roughly symmetric numeric |
| Z-score | `(x - mean) / std > 3` | Approximately normal data |
| Modified Z-score | uses median + MAD | Robust to outliers themselves |
| Percentile cap | < 1st or > 99th percentile | Heavy-tailed distributions |
| Isolation Forest | `sklearn.ensemble.IsolationForest` | Multivariate outliers |
| Local Outlier Factor | `sklearn.neighbors.LocalOutlierFactor` | Density-based |

```python
q_low, q_hi = df["price"].quantile([0.01, 0.99])
mask = df["price"].between(q_low, q_hi)
```

### 7.2 Investigate, don't auto-delete
Outliers are often **real and important** — fraud, VIP users, system bugs.
Look at 20-30 of them by hand before deciding.

### 7.3 Treatment

| Treatment | When |
|---|---|
| **Keep** (and use a robust model) | The outlier is real (e.g. Bezos's salary) |
| **Cap / winsorize** at 1st-99th percentile | Heavy tail you don't want the model to over-fit to |
| **Log / sqrt transform** | Right-skewed positive data (income, prices) |
| **Drop** | Confirmed data error (negative age, future birthday) |
| **Bin** | Order matters more than magnitude (low / med / high) |
| **Flag** | Outlier itself is a feature (`is_outlier = ...`) |

```python
# Winsorize
df["price_w"] = df["price"].clip(lower=q_low, upper=q_hi)
# Log transform (handle zeros)
df["price_log"] = np.log1p(df["price"])
```

**Tree-based models (XGBoost, RF) handle outliers fine** — don't bother
transforming for them. Linear/NN models are more sensitive.

---

## Step 8 — Range / Constraint Violations

Domain rules your data must satisfy. List them, check them, fix them.

| Rule | Check | Fix |
|---|---|---|
| Age between 0 and 120 | `df["age"].between(0, 120)` | Drop or clip |
| End date ≥ start date | `df["end"] >= df["start"]` | Investigate; often swap or drop |
| Price > 0 | `df["price"] > 0` | Drop free items / treat separately |
| Probability in [0, 1] | `df["p"].between(0, 1)` | Clip — usually a numerical bug |
| Sum of percentages = 100 | row-wise `.sum() == 100` | Renormalise or flag |
| Foreign keys exist | `df["uid"].isin(users["id"])` | Drop orphans |

The pattern is always: write the rule down, count violations, decide a treatment per rule.

---

## Step 9 — Date & Time

Dates are deceptively hard. Always:
1. **Parse** to a `datetime` dtype (Step 3).
2. **Normalise to UTC** at ingest, convert to local at display.
3. **Sanity-check ranges**:
   - No future dates where they shouldn't be.
   - No dates before the system existed (e.g. signups in 1970 — unix epoch leak).
   - DST transitions don't break grouping.
4. **Derive features** thoughtfully:
   - hour of day, day of week, month, quarter
   - days since first event / last event
   - is_weekend, is_holiday (use a holidays library)

```python
df["signup"] = pd.to_datetime(df["signup"], utc=True, errors="coerce")
bad = df[(df["signup"] < "2010-01-01") | (df["signup"] > pd.Timestamp.utcnow())]
```

---

## Step 10 — Numerical Issues

Once values are clean, look at their distribution and decide whether to transform.

### 10.1 Skew
- Right-skewed (income, prices, counts): `np.log1p(x)`, `np.sqrt(x)`, Box-Cox, Yeo-Johnson.
- Left-skewed: reflect then log: `np.log(max - x + 1)`.
- Use `df["x"].skew()` — values > 1 or < -1 are notably skewed.

### 10.2 Scale
- Standardisation: `(x - mean) / std` — for linear models, NN, SVM, KNN, PCA.
- Min-max: `(x - min) / (max - min)` — when bounds are known.
- Robust: `(x - median) / IQR` — when outliers exist.
- Tree models (XGBoost, RF): scaling is irrelevant. Don't bother.

### 10.3 Multi-collinearity
Highly correlated features confuse linear models.
```python
corr = df.corr(numeric_only=True).abs()
# pairs with > 0.9 correlation
high = corr.unstack().sort_values(ascending=False)
high = high[(high < 1) & (high > 0.9)]
```
Drop one of each pair, or combine via PCA, or use regularisation (Ridge/Lasso).

---

## Step 11 — Categorical Issues

### 11.1 High cardinality
A column with 50,000 unique values isn't a "category" — it's an ID or free text.
- Group rare categories: anything with < N occurrences → `"OTHER"`.
- Use **target encoding** (mean of target per category, with smoothing) instead of one-hot.
- Use **embeddings** (entity embeddings via NN, or pretrained sentence encoders).
- Use **hashing trick** (`sklearn.feature_extraction.FeatureHasher`).

```python
counts = df["product"].value_counts()
rare = counts[counts < 50].index
df["product"] = df["product"].where(~df["product"].isin(rare), "OTHER")
```

### 11.2 Rare classes (in the target)
- Stratified split so each fold has all classes.
- Class weights: `class_weight="balanced"` in sklearn.
- Resampling: SMOTE, ADASYN, random over/under-sampling — apply *only on train*.
- Threshold tuning instead of resampling — often simpler and as effective.

### 11.3 Ordinal vs nominal
- Ordinal (low/med/high, S/M/L/XL): map to integers preserving order.
- Nominal (red/blue/green): one-hot or target-encode. Do **not** label-encode
  to integers — the model will assume order.

---

## Step 12 — Target Variable

Always check the target *separately* — bugs here are catastrophic.

### Checks
- **Distribution** — `df["target"].value_counts(normalize=True)` for classification,
  `df["target"].describe()` + histogram for regression.
- **Imbalance** — anything beyond 70/30 starts to matter; 95/5 or worse needs strategy.
- **Missing target** — drop these rows. Never impute the target.
- **Sanity** — does the label match a sample of inputs? Spot-check 50 rows.
- **Label noise** — disagreements between labellers? quantify.
- **Label leakage** — does the target appear (directly or indirectly) in the features?

### Treatment for imbalance
- Use proper metrics (precision/recall/F1/AUC, not accuracy).
- Class weights (cheap, often enough).
- Resampling (SMOTE etc.) — train only.
- Adjust decision threshold — maximise the metric you care about.

---

## Step 13 — Feature Leakage

The single fastest way to ship a useless model. Drop columns that:
- Are computed **from** the target (e.g. `risk_score` was derived from `defaulted`).
- Won't exist at inference time (`closed_at` for an open ticket).
- Use information from the **future** (in time series).
- Came in *after* the target event.

**Smell test:** if a single feature gives you 99% accuracy, it's leakage 95% of the time.

```python
# Compute feature importance early — investigate suspiciously perfect features.
from sklearn.ensemble import RandomForestClassifier
imp = pd.Series(
    RandomForestClassifier().fit(X, y).feature_importances_,
    index=X.columns
).sort_values(ascending=False)
print(imp.head(10))   # look at the top — anything too good to be true?
```

For time series / event data, also check:
- Are aggregations computed using only data **before** the target's timestamp?
- Are train/val/test split **chronologically**, not randomly?

---

## Step 14 — Final Validation & Save

Once cleaned, treat the cleaned dataset as a contract.

```python
# Re-run the schema check from Step 2 — should now pass cleanly.
schema.validate(df, lazy=True)

# Save with format that preserves dtypes (CSV does NOT).
df.to_parquet("data/cleaned/v1.parquet", index=False)

# Document what changed
with open("data/cleaned/v1.md", "w") as f:
    f.write(f"""
# Cleaned Dataset v1
- Source: raw.csv (sha256 = ...)
- Rows: raw {raw_rows} -> clean {len(df)}
- Columns dropped: {dropped}
- Imputation strategy: median for numeric, mode for categorical
- Outlier policy: winsorize at 1-99 percentile for `price`
- Cleaning script: scripts/clean_v1.py @ commit {commit}
""")
```

**Rules for the cleaned dataset:**
- Saved to a separate path — never overwrite raw.
- Versioned (folder name, DVC, git tag).
- Has a small README describing what was done.
- Cleaning logic lives in **a script**, not just a notebook — so you can rerun it.

---

## Putting It All Together — Skeleton Script

```python
import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, Check
from sklearn.impute import SimpleImputer

RAW_PATH   = "data/raw/users.csv"
CLEAN_PATH = "data/cleaned/users_v1.parquet"

# 1. Load
df = pd.read_csv(RAW_PATH)
print("loaded", df.shape)

# 2. Normalise column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# 3. Dtypes
df["signup"] = pd.to_datetime(df["signup"], errors="coerce", utc=True)
df["price"]  = pd.to_numeric(
    df["price"].astype(str).str.replace(r"[\$,]", "", regex=True),
    errors="coerce",
)
df["country"] = df["country"].astype(str).str.strip().str.upper()

# 4. Missing
df = df.dropna(subset=["user_id", "target"])              # drop critical NAs
df["age"] = df["age"].fillna(df["age"].median())          # impute non-critical

# 5. Duplicates
df = df.drop_duplicates(subset=["user_id"], keep="last")

# 6. String cleanup already done above for `country`. Map synonyms:
df["country"] = df["country"].replace({"USA": "US", "U.S.": "US", "UK": "GB"})

# 7. Outliers — winsorise price
lo, hi = df["price"].quantile([0.01, 0.99])
df["price"] = df["price"].clip(lo, hi)

# 8. Range rules
df = df[df["age"].between(0, 120)]
df = df[df["price"] >= 0]
df = df[df["signup"] <= pd.Timestamp.utcnow()]

# 9. Date features
df["signup_dow"]  = df["signup"].dt.dayofweek
df["signup_hour"] = df["signup"].dt.hour

# 10. Numerical transforms
df["price_log"] = np.log1p(df["price"])

# 11. Categorical — group rare countries
rare = df["country"].value_counts()
rare = rare[rare < 50].index
df["country"] = df["country"].where(~df["country"].isin(rare), "OTHER")

# 12-13. Drop leakage / unused
df = df.drop(columns=["internal_score", "closed_at"])

# 14. Validate + save
schema = pa.DataFrameSchema({
    "user_id": Column(str, nullable=False),
    "age":     Column(int,  Check.in_range(0, 120)),
    "price":   Column(float, Check.ge(0)),
    "country": Column(str),
    "signup":  Column(pa.DateTime),
    "target":  Column(int, Check.isin([0, 1])),
})
schema.validate(df, lazy=True)

df.to_parquet(CLEAN_PATH, index=False)
print("saved", CLEAN_PATH, df.shape)
```

---

## Variants for Other Data Types

### Images
1. Verify all files open (corrupt JPEGs are common).
2. Standardise dimensions / aspect ratio / colour space.
3. Strip EXIF / rotate by EXIF orientation.
4. Remove exact + perceptual duplicates (`imagehash` library).
5. Normalise pixel values (0-1 or ImageNet stats).
6. Augment **only on train** (flips, crops, jitter).

### Text
1. Encoding to UTF-8.
2. Lowercase (sometimes).
3. Strip HTML / markdown / control chars.
4. Normalise Unicode (`NFKC`).
5. Remove or replace URLs, emails, numbers (depending on task).
6. Tokenise (whitespace, BPE, SentencePiece).
7. De-duplicate near-duplicates (MinHash / SimHash).
8. PII scrubbing (names, phone numbers).

### Time series
1. Re-index to a regular frequency, fill gaps explicitly.
2. Decide on imputation (forward-fill vs interpolate vs leave NA).
3. Detect and treat anomalies separately from outliers.
4. Detrend / deseasonalise if required.
5. Lag features computed strictly with past data.

### Audio
1. Resample to consistent rate (e.g. 16 kHz).
2. Convert to mono if needed.
3. Trim leading/trailing silence.
4. Normalise loudness (LUFS).
5. Feature extraction: MFCCs, mel-spectrograms.

---

## Anti-Patterns (Things That Will Burn You)

- **Cleaning in the notebook only.** Six months later you can't reproduce it.
  Move logic to a `clean.py` script.
- **Fitting scalers/imputers on the full dataset before splitting.** Leakage.
- **Auto-dropping outliers.** Often the most important rows.
- **Using `fillna(0)` blindly.** 0 is a number; the model learns from it.
- **Forgetting to apply the same cleaning at inference.** Train and serve must
  use the *same* transformer — wrap in `sklearn.Pipeline` or save your steps.
- **Cleaning the test set differently from train.** Use the same code path.
- **Not logging what you dropped.** Always print `before -> after` row counts.
- **Trusting the column name.** A column called `age` may have ages in months,
  in days, or be the user's astrological age. Always check the values.

---

## The 14-Step Checklist (Print This)

- [ ] 1. Loaded, looked at shape / head / tail / sample / info / describe
- [ ] 2. Schema written down and validated against
- [ ] 3. All dtypes correct (dates, numbers, booleans, IDs)
- [ ] 4. Missing values counted, pattern understood, treatment chosen
- [ ] 5. Exact + near duplicates removed (or kept with reason)
- [ ] 6. String columns: stripped, normalised case, mapped synonyms
- [ ] 7. Outliers detected, investigated, treated per column
- [ ] 8. Domain rules (age ≥ 0, end ≥ start, etc.) checked
- [ ] 9. Dates parsed, in UTC, sanity-checked, features derived
- [ ] 10. Numeric distributions inspected; transforms applied where needed
- [ ] 11. Categoricals: rare grouped, encoded appropriately
- [ ] 12. Target variable: distribution, imbalance, no leakage, no missing
- [ ] 13. Leakage features dropped; no future info in past predictions
- [ ] 14. Re-validated, saved as parquet, README + script committed

When this checklist is fully ticked, you're ready to move on to feature
engineering and modelling.
