# NLP 4 — Bag of Words, n-grams, TF-IDF

## Lectures covered
- Bag of Words, n-grams, TF-IDF

---

## 1. Bag of Words (BoW)

Treat each document as a **multiset of words** (ignoring order). Each document → a vector of word counts.

### The vocabulary + matrix

Documents:
1. "the cat sat"
2. "the dog ran"
3. "cat and dog"

Vocabulary: `{the, cat, sat, dog, ran, and}`

Document-term matrix:
| | the | cat | sat | dog | ran | and |
|---|---|---|---|---|---|---|
| doc 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| doc 2 | 1 | 0 | 0 | 1 | 1 | 0 |
| doc 3 | 0 | 1 | 0 | 1 | 0 | 1 |

Each row is the doc's vector. Now we have numbers — feed to any classifier.

### sklearn `CountVectorizer`
```python
from sklearn.feature_extraction.text import CountVectorizer

texts = ["the cat sat", "the dog ran", "cat and dog"]
cv = CountVectorizer()
X = cv.fit_transform(texts)         # sparse matrix
print(cv.get_feature_names_out())    # ['and', 'cat', 'dog', 'ran', 'sat', 'the']
print(X.toarray())
```

### Pros / cons
- **Pros**: simple, interpretable, fast, sparse-friendly
- **Cons**: ignores order, ignores meaning ("good" and "great" are unrelated dimensions), high-dim (vocab size)

---

## 2. n-grams — capturing local order

Instead of single words, use **sequences of n words**.

- 1-gram (unigram): "the", "cat", "sat"
- 2-gram (bigram): "the cat", "cat sat"
- 3-gram (trigram): "the cat sat"

```python
cv = CountVectorizer(ngram_range=(1, 2))      # unigrams + bigrams
X = cv.fit_transform(texts)
```

### Why
- Captures phrases ("not good", "New York")
- Helps with sentiment ("not happy" ≠ "happy")
- More features → more capacity, more sparsity

### Watch out
- Vocabulary explodes (~10× per n)
- Use `min_df=2` or `max_features=10000` to control size

```python
cv = CountVectorizer(ngram_range=(1, 2), min_df=3, max_features=20000)
```

---

## 3. TF-IDF — Term Frequency × Inverse Document Frequency

Plain BoW gives equal weight to "the" and "antibiotic". TF-IDF down-weights words that appear in many documents (uninformative) and up-weights rare-but-meaningful words.

### Math
$$\text{TF}(t, d) = \frac{\text{count}(t, d)}{\text{words in } d}$$

$$\text{IDF}(t) = \log\!\left(\frac{N}{1 + |\{d : t \in d\}|}\right)$$

$$\text{TFIDF}(t, d) = \text{TF}(t, d) \cdot \text{IDF}(t)$$

Higher score → "this term is important *for this document* relative to the corpus."

### sklearn `TfidfVectorizer`
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)
X = tfidf.fit_transform(texts)
```

### Useful parameters
| Param | What |
|---|---|
| `ngram_range` | (1, 1) unigram only; (1, 2) uni + bi |
| `max_features` | top N most frequent terms |
| `min_df` | ignore terms in fewer than k docs |
| `max_df` | ignore terms in too many docs (stopword-like) |
| `sublinear_tf=True` | use 1 + log(tf) — dampens high counts |
| `stop_words="english"` | drops English stop-words |
| `analyzer="char_wb"` | character n-grams, robust to misspellings |

---

## 4. Pipeline + classifier — full BoW/TF-IDF baseline

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000)),
    ("clf",   LogisticRegression(max_iter=1000, C=1.0)),
])

scores = cross_val_score(pipe, texts, labels, cv=5, scoring="f1_macro")
print(scores.mean())
```

For text classification on small-to-medium data, **TF-IDF + Logistic Regression** is the classic baseline. Often hits 80–90% of a fine-tuned BERT's accuracy in 1% of the compute.

---

## 5. Why TF-IDF still matters in 2025

- **Speed**: vectorize 1M documents in seconds
- **Sparse**: efficient memory
- **Interpretable**: which features drove the prediction → coefficients are readable
- **Strong baseline**: always run it first
- **RAG**: BM25 (a TF-IDF cousin) is part of every modern hybrid retrieval system
- **Search**: TF-IDF + cosine sim still powers many search bars

---

## 6. BM25 — TF-IDF's cousin (the search engine standard)

BM25 = TF-IDF with diminishing returns on TF and length normalization. Used in **Elasticsearch, Solr, Lucene** — every search engine you've used.

```python
from rank_bm25 import BM25Okapi
corpus = [doc.split() for doc in docs]
bm25 = BM25Okapi(corpus)
scores = bm25.get_scores("query words".split())
```

BM25 + dense embeddings (BERT-style) is the **hybrid retrieval** that powers modern RAG systems. Module 9.

---

## 7. Hashing trick — for huge vocabularies

When vocab is so large you can't fit it:
```python
from sklearn.feature_extraction.text import HashingVectorizer
hv = HashingVectorizer(n_features=2**18)
X = hv.transform(texts)
```

Maps any word to one of N buckets via hashing. No vocabulary stored. Fast, online-learnable. Some collisions but they're tolerable.

---

## 8. End-to-end sentiment classifier (10 lines)

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

data = fetch_20newsgroups(subset="train")
test = fetch_20newsgroups(subset="test")

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000)),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipe.fit(data.data, data.target)
print(classification_report(test.target, pipe.predict(test.data), target_names=test.target_names[:5]))
```

20-newsgroups in 1 minute, ~85% F1.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Calling `vectorizer.fit_transform` on test data | leakage | `fit` on train, `transform` on test |
| Removing punctuation/case before TF-IDF for sentiment | losing signal | leave it; TF-IDF handles |
| `ngram_range=(1, 5)` blindly | massive memory blow-up | start with (1, 2), grow only if needed |
| Forgetting to normalize | length bias | TfidfVectorizer normalizes by default |
| Comparing TF-IDF + LR vs BERT without baseline | over-engineering | always start with TF-IDF |

## Self-check

- [ ] What does Bag of Words ignore?
- [ ] Why are bigrams useful?
- [ ] Define TF-IDF in plain English.
- [ ] What does `min_df=3` do?
- [ ] When use HashingVectorizer over CountVectorizer?
- [ ] What's BM25 and how does it relate to TF-IDF?
- [ ] Build a TF-IDF + LogReg pipeline for sentiment in 10 lines.
- [ ] Why is TF-IDF still relevant in 2025 even with LLMs?
