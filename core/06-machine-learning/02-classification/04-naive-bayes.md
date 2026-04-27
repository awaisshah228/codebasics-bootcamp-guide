# Classification 4 — Naive Bayes

## Lectures covered
- Naive Bayes

---

## 1. The Bayesian intuition

For classification, we want $P(y = c \mid x)$. By Bayes' theorem:
$$P(y = c \mid x) = \frac{P(x \mid y = c) \cdot P(y = c)}{P(x)}$$

For each class, compute the right-hand side; pick the class with the highest probability. The denominator $P(x)$ is the same across classes, so we can ignore it for ranking.

---

## 2. The "naive" assumption

Computing $P(x \mid y = c)$ for vector $x = (x_1, x_2, \dots, x_p)$ is hard — joint distributions in high dimensions are nasty.

**Naive Bayes assumes feature independence** *given the class*:
$$P(x_1, x_2, \dots, x_p \mid y) = \prod_j P(x_j \mid y)$$

This is wildly unrealistic in real data — but works surprisingly well, especially for text.

---

## 3. Three flavors

### Gaussian NB (continuous features)
Assumes $x_j \mid y$ is Normally distributed.

```python
from sklearn.naive_bayes import GaussianNB
GaussianNB().fit(X, y)
```

### Multinomial NB (count / TF features)
Designed for word counts in document classification.

```python
from sklearn.naive_bayes import MultinomialNB
MultinomialNB().fit(X_counts, y)
```

### Bernoulli NB (binary features)
Each feature is 0/1 — "did this word appear at all?"

```python
from sklearn.naive_bayes import BernoulliNB
```

### Complement NB (variant of Multinomial — better on imbalanced text data)
```python
from sklearn.naive_bayes import ComplementNB
```

---

## 4. Spam classifier walk-through

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

# data: emails with "text" column and "spam" 0/1 label
df = pd.read_csv("emails.csv")

pipe = Pipeline([
    ("vec", CountVectorizer(stop_words="english", ngram_range=(1, 2))),
    ("nb",  MultinomialNB()),
])

pipe.fit(df["text"], df["spam"])
print(classification_report(df["spam"], pipe.predict(df["text"])))
```

That's a fully-functional spam classifier in 10 lines.

---

## 5. Strengths

- **Fast** — train and predict in seconds even on millions of examples
- **Tiny memory footprint**
- Surprisingly effective on text classification (the canonical baseline)
- Naturally handles online / incremental learning

## 6. Weaknesses

- The independence assumption is wrong — model overconfident probabilities
- Not great with continuous features unless they're roughly Normal
- Can't capture feature interactions

---

## 7. Smoothing — handling unseen feature combos

Without smoothing, if a word never appeared in spam in training, $P(\text{word} \mid \text{spam}) = 0$ → entire posterior is 0. Bad.

**Laplace / additive smoothing** adds a small constant α (default 1) to all counts:
```python
MultinomialNB(alpha=1.0)
```

Lower α → tighter to data; higher α → smoother priors.

---

## 8. Use Naive Bayes when

- You need a **strong baseline** for text classification
- Speed matters more than top accuracy
- You have very little training data
- Simple, interpretable model is valued

For modern text problems, fine-tuned BERT / sentence-transformers usually outperform — but NB is the right starting line.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Using GaussianNB on count data | wrong distribution assumption | use MultinomialNB |
| MultinomialNB on data with negatives | invalid (counts must be ≥ 0) | use GaussianNB or scale to 0+ |
| Using probabilities as well-calibrated | NB probs are often miscalibrated | use `CalibratedClassifierCV` if you need true probs |
| Skipping stop-words removal in text | noisy features | `stop_words="english"` in `CountVectorizer` |

## Self-check

- [ ] What's the "naive" assumption?
- [ ] When use Multinomial vs Gaussian NB?
- [ ] Why is smoothing important?
- [ ] Why is Naive Bayes a great baseline for text?
- [ ] Why are NB probabilities often poorly calibrated?
- [ ] Build a sentiment classifier on tweets in 15 lines.
- [ ] What's the difference between Multinomial NB and Bernoulli NB?
