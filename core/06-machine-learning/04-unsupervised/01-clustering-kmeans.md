# Unsupervised 1 — K-Means Clustering

## Lectures covered
- K Means Clustering

---

## 1. The unsupervised setting

No labels. Just `X`. Goal: discover structure — groups, anomalies, lower-dimensional representations.

**Clustering** = partition observations into groups where members are similar to each other and dissimilar from other groups.

Common DS uses:
- Customer segmentation
- Image color quantization
- Document grouping
- Anomaly detection (lonely points)
- Pre-processing step before supervised models

---

## 2. K-Means — the algorithm

**Inputs**: data X, number of clusters k.

```
1. Pick k random initial centroids
2. Repeat:
   a. Assign each point to its nearest centroid (Euclidean distance)
   b. Update each centroid to the mean of its assigned points
3. Stop when assignments stop changing (or max iterations reached)
```

Visually: each cluster is a Voronoi cell around its centroid.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X_scaled = StandardScaler().fit_transform(X)
km = KMeans(n_clusters=4, n_init="auto", random_state=42)
labels = km.fit_predict(X_scaled)

print(km.cluster_centers_)
print(km.inertia_)             # sum of within-cluster squared distances
```

> Always **scale** features for K-Means — it's distance-based.

---

## 3. Picking k — the elbow method

Plot **inertia** (within-cluster sum of squares) for k = 1, 2, 3, …

```python
import matplotlib.pyplot as plt
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init="auto", random_state=42).fit(X_scaled)
    inertias.append(km.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("k"); plt.ylabel("inertia (lower is tighter)")
```

The **elbow** (where the curve bends sharply) is a reasonable k.

```
inertia
│  •
│   •
│    •
│     ╲
│      • <- elbow at k=4
│        ──
│         ────•
└────────────► k
```

---

## 4. Silhouette score — quantitative quality measure

For each point:
$$s = \frac{b - a}{\max(a, b)}$$
- $a$: avg distance to other points in same cluster
- $b$: avg distance to nearest other cluster

Range: −1 to 1.
- ~1: well-clustered
- ~0: on the boundary
- < 0: probably in the wrong cluster

```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X_scaled, labels)
```

Compare scores across different k; higher = better.

---

## 5. K-Means initialization — `k-means++`

Randomly placed initial centroids can converge to bad local optima. **k-means++** picks initial centroids that are spread out, dramatically improving stability. sklearn uses it by default (`init="k-means++"`).

`n_init` controls how many random restarts to run; the best result wins. Default `n_init="auto"` in modern sklearn picks a sensible number.

---

## 6. K-Means assumptions / failure modes

K-Means works best when clusters are:
- Spherical (Euclidean distance bias)
- Roughly equal in size
- Well-separated

It fails when clusters are:
- Crescent / elongated → use **DBSCAN** or **Spectral Clustering**
- Density-based, varying sizes → DBSCAN
- Hierarchical structure → **Agglomerative Clustering**

```
   K-means OK            K-means fails
   ⚪⚪⚪    ⚫⚫⚫      ╮          ╭
   ⚪⚪⚪    ⚫⚫⚫       ╰─⚪⚪⚪──╯  ⚫⚫⚫
                          ╰⚫⚫⚫╯
```

---

## 7. Customer segmentation — typical workflow

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# features: spend, frequency, recency, etc. (RFM)
features = ["recency_days", "frequency", "monetary_amount"]
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# pick k via elbow / silhouette
km = KMeans(n_clusters=4, n_init="auto", random_state=42).fit(X_scaled)
df["cluster"] = km.labels_

# interpret each cluster
df.groupby("cluster")[features].agg(["mean", "median", "count"])
```

Then name the clusters: "high-value loyalists", "lapsed", "newcomers", "price-sensitive."

This is exactly the AtliQo Bank target-segment exercise (Module 5) extended into an *unsupervised* path.

---

## 8. Other clustering algorithms (brief)

### DBSCAN — density-based
- Doesn't need k upfront
- Finds arbitrary shapes
- Marks noise points

```python
from sklearn.cluster import DBSCAN
DBSCAN(eps=0.5, min_samples=5).fit_predict(X_scaled)
```

### Hierarchical / Agglomerative
- Builds a dendrogram (tree) of clusters
- Useful when hierarchy itself is informative

```python
from sklearn.cluster import AgglomerativeClustering
AgglomerativeClustering(n_clusters=4).fit_predict(X_scaled)
```

### Gaussian Mixture (GMM) — soft clusters
- Each point gets a *probability* of belonging to each cluster
- Better than k-means when clusters overlap

---

## 9. PCA before clustering

Cluster algorithms struggle in high dimensions. Often:
1. Reduce to 2–10 dimensions with PCA
2. Cluster on PCA features
3. Visualize cluster assignments back in 2D

```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2).fit_transform(X_scaled)
```

Or **t-SNE / UMAP** for visualization.

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Skipping scaling | dominant feature kidnaps clusters | always scale |
| Picking k arbitrarily | weird clusters | elbow + silhouette |
| Reading k-means clusters as "true" categories | might be artifacts | always interpret + validate with domain |
| K-means on non-spherical clusters | fails | use DBSCAN / GMM |
| Tiny n_init | unstable | leave at default ("auto") |

## Self-check

- [ ] Walk through K-Means in 4 steps.
- [ ] Why scale before K-Means?
- [ ] Elbow method: how do you read it?
- [ ] What's the silhouette score and when prefer it over inertia?
- [ ] What are 3 ways K-Means can fail?
- [ ] When use DBSCAN over K-Means?
- [ ] What's `n_init` and `k-means++`?
- [ ] How would you cluster customer transaction histories?
