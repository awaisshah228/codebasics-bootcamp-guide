# Section 9 — EDA: Pandas, Matplotlib, Seaborn

## Lectures covered
- Pandas Intro & Installation · DataFrame Basics · Read/Write Excel & CSV · Handle NA values (1 & 2) · Group By · Concat & Merge · Visualization · Quiz

---

## 1. Pandas — what it actually is

A **DataFrame** is a 2D labeled table — like a spreadsheet, like a SQL table, but Python-native and lightning fast.

A **Series** is a single column with an index.

Pandas sits on top of NumPy (every column is a NumPy array under the hood) and adds: labels, missing-value handling, group-by, time-series, IO for ~30 file formats.

```python
import pandas as pd
```

---

## 2. Creating DataFrames

```python
# from dict of lists
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "age": [30, 25, 35],
    "city": ["NY", "LA", "SF"],
})

# from list of dicts
df = pd.DataFrame([
    {"name": "Alice", "age": 30},
    {"name": "Bob",   "age": 25},
])

# from CSV (most common)
df = pd.read_csv("data.csv")

# from Excel
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")

# from JSON
df = pd.read_json("data.json")

# from SQL
import sqlite3
conn = sqlite3.connect("db.sqlite")
df = pd.read_sql("SELECT * FROM users", conn)
```

### Writing back
```python
df.to_csv("out.csv", index=False)        # ALWAYS pass index=False (default writes the row index as a column)
df.to_excel("out.xlsx", index=False)
df.to_json("out.json", orient="records")
df.to_parquet("out.parquet")             # smaller + faster than CSV
```

---

## 3. Inspecting

```python
df.head()                      # first 5 rows
df.tail(10)                    # last 10
df.shape                       # (rows, cols)
df.columns                     # column names
df.dtypes                      # type per column
df.info()                      # types + non-null count + memory
df.describe()                  # summary stats for numeric cols
df.describe(include="object")  # for string cols
df.sample(5)                   # random 5 rows
df.nunique()                   # unique values per column
df["city"].value_counts()      # frequency table
```

`.head()` + `.info()` should be the **first two lines** of every notebook after loading.

---

## 4. Selecting

### Columns
```python
df["name"]                  # one col → Series
df[["name", "age"]]         # multiple cols → DataFrame
df.name                     # attribute access (avoid — breaks on names like "class")
```

### Rows
```python
df.iloc[0]                  # row by position
df.iloc[0:3]                # rows by position range
df.loc[0]                   # row by label (here label == position)
df.loc[df["age"] > 25]      # boolean mask
```

### Cells
```python
df.loc[0, "name"]
df.iloc[0, 1]
```

### Filtering (boolean indexing)
```python
df[df["age"] > 25]
df[(df["age"] > 25) & (df["city"] == "NY")]   # parens around each condition
df[df["city"].isin(["NY", "LA"])]
df[df["name"].str.startswith("A")]
df.query("age > 25 and city == 'NY'")          # alternative
```

---

## 5. Adding & modifying columns

```python
df["age_doubled"] = df["age"] * 2
df["category"] = df["age"].apply(lambda a: "senior" if a >= 30 else "junior")
df["bin"] = pd.cut(df["age"], bins=[0, 25, 35, 100], labels=["young", "mid", "old"])
df = df.assign(age_in_months=lambda d: d["age"] * 12)         # chainable
df = df.rename(columns={"name": "full_name"})
df = df.drop(columns=["age_doubled"])
```

---

## 6. Handling missing values (NA / NaN)

### Detect
```python
df.isna().sum()            # # of NAs per column
df.isna().any(axis=1)      # rows with any NA
df.isna().mean() * 100     # % of NAs per column
```

### Drop
```python
df.dropna()                              # drop any row with NA
df.dropna(subset=["age"])                # drop rows where age is NA
df.dropna(thresh=3)                      # keep rows with at least 3 non-NA
df.dropna(axis=1)                        # drop columns with any NA
```

### Fill
```python
df["age"].fillna(df["age"].mean())       # numeric → mean / median
df["city"].fillna("unknown")             # categorical → "unknown" or mode
df.ffill()                               # forward-fill (time series)
df.bfill()                               # backward-fill
df.interpolate()                         # numeric interpolation
```

> **Don't blindly fill with mean.** If 30% of `salary` is missing, mean-fill biases your model. Investigate *why* missing first.

---

## 7. GroupBy — the workhorse

```python
# average age per city
df.groupby("city")["age"].mean()

# multiple aggregations
df.groupby("city").agg(
    avg_age=("age", "mean"),
    n=("name", "count"),
    max_age=("age", "max"),
)

# group by multiple keys
df.groupby(["city", "category"])["age"].mean()

# transform — keep original shape, broadcast group stat
df["pct_of_city"] = df["age"] / df.groupby("city")["age"].transform("sum")
```

### `pivot_table` (when you want a wide cross-tab)
```python
df.pivot_table(index="city", columns="category", values="age", aggfunc="mean")
```

---

## 8. Concat and Merge

### Concat — stack DataFrames
```python
pd.concat([df1, df2], axis=0)          # stack rows (UNION ALL)
pd.concat([df1, df2], axis=1)          # stack columns (parallel)
```

### Merge — SQL-style join
```python
pd.merge(df1, df2, on="customer_id", how="inner")        # default
pd.merge(df1, df2, on="customer_id", how="left")         # left join
pd.merge(df1, df2, on="customer_id", how="outer")        # full outer
pd.merge(df1, df2, left_on="cid", right_on="customer_id")
```

### Indicator + suffixes
```python
pd.merge(df1, df2, on="id", how="left", indicator=True, suffixes=("_lhs", "_rhs"))
```

The `indicator=True` column shows whether each row matched left/right/both — invaluable for auditing joins.

---

## 9. Visualization — matplotlib + seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

### matplotlib (raw, control-heavy)
```python
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(df["year"], df["revenue"], marker="o")
ax.set_title("Revenue by year")
ax.set_xlabel("Year")
ax.set_ylabel("Revenue ($)")
plt.tight_layout()
plt.show()
```

### seaborn (high-level, prettier defaults)
```python
sns.histplot(df["age"], bins=20, kde=True)
sns.boxplot(data=df, x="city", y="age")
sns.scatterplot(data=df, x="age", y="income", hue="city")
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
sns.pairplot(df, hue="category")
```

### Plot type → use case quick guide
| Plot | When to use |
|---|---|
| Histogram | distribution of one numeric var |
| Box plot | distribution + outliers, comparing groups |
| Bar chart | counts/aggregates by category |
| Line chart | trend over time |
| Scatter plot | relationship between two numeric vars |
| Heatmap | correlation matrix or 2D density |
| Pair plot | quick all-vs-all view of a small DataFrame |

### Saving figures
```python
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
```

---

## 10. EDA workflow — the playbook

For any new dataset, in order:

1. `df.head()`, `df.shape`, `df.info()`
2. `df.describe(include="all")`
3. `df.isna().sum()` → decide drop/fill strategy
4. `df.duplicated().sum()` → drop or investigate
5. For each numeric column: histogram + box plot
6. For each categorical: `value_counts()` + bar
7. Pairwise: correlation heatmap, scatter matrix on key columns
8. Time-aware columns: parse with `pd.to_datetime`, plot trend
9. Outlier check: IQR, std-dev, domain-knowledge rule
10. Write 3–5 markdown insights at the bottom

This is exactly what Project 1 (Hospitality EDA) walks through — see `02-projects/01-hospitality-eda.md`.

---

## 11. Common pitfalls

| Bug | Cause | Fix |
|---|---|---|
| `SettingWithCopyWarning` | chained indexing on a slice | use `.loc` for assignment |
| `KeyError` on column | typo or whitespace | `df.columns.tolist()` to see exact names |
| Merging changes row count unexpectedly | one-to-many join | check `indicator=True`, sanity-check shape |
| Reading CSV looks weird | bad delimiter/encoding | `sep=";"`, `encoding="latin1"` |
| `df["col"] = something` modifying SQL view | side effect inside a function | always work on copies |
| Date column reads as string | pandas didn't auto-detect | `pd.to_datetime(df["col"])` |

## Self-check

- [ ] How do I read a CSV that uses `;` as separator?
- [ ] What's the difference between `loc` and `iloc`?
- [ ] How do I get the count of NAs per column?
- [ ] What does `groupby("city").agg(...)` produce?
- [ ] What's the difference between `merge` with `how="inner"` and `how="left"`?
- [ ] When would I use a `pivot_table`?
- [ ] How do I plot a histogram of `age` with KDE in seaborn?
- [ ] What does `SettingWithCopyWarning` mean and how do I fix it?
