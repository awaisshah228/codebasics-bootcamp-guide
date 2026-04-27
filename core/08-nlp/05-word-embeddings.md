# NLP 5 — Word Embeddings (Word2Vec, GloVe, FastText)

## Lectures covered
- Word Embeddings (×2 lectures in the curriculum)

---

## 1. The problem with one-hot

If your vocab has 50,000 words, each word is a vector of length 50,000 with a single 1. Disadvantages:
- **No similarity**: dot product between any two words = 0
- **Huge dimensionality**: linear models with 50k features overfit
- **No generalization**: knowing "happy" tells you nothing about "joyful"

Solution: dense, low-dim, **learned** vectors where similar words are close.

---

## 2. Word2Vec (2013)

A small neural network trained on a simple task. Two flavors:

### Skip-gram
Given a center word, predict surrounding context words.
"the cat sat on the mat"
- center: "cat" → predict {"the", "sat"}

### CBOW (Continuous Bag of Words)
Given context, predict the center word.
- context: {"the", "sat"} → predict "cat"

The middle layer's weights become the **word embeddings**.

### Famous result
$$\text{vec(king)} - \text{vec(man)} + \text{vec(woman)} \approx \text{vec(queen)}$$

The model captures **semantic relationships** through geometry. Similar property:
- vec(Paris) − vec(France) + vec(Italy) ≈ vec(Rome)
- vec(walking) − vec(walked) + vec(swam) ≈ vec(swimming)

### Train your own with gensim
```python
from gensim.models import Word2Vec

sentences = [["the", "cat", "sat", "on", "the", "mat"], ...]    # tokenized
model = Word2Vec(sentences, vector_size=100, window=5, min_count=2, workers=4)
model.wv["cat"]                          # 100-dim vector
model.wv.most_similar("cat")             # [(dog, 0.85), (mouse, 0.78), ...]
model.wv.similarity("cat", "dog")
```

---

## 3. GloVe — Global Vectors (2014)

Different math. Trained on the **co-occurrence matrix** of the entire corpus instead of local windows.

```
"cat" co-occurs with "dog" often → close vectors
"cat" rarely co-occurs with "asteroid" → distant vectors
```

In practice, very similar quality to Word2Vec. Often easier to use because Stanford released pre-trained GloVe vectors trained on Common Crawl + Wikipedia.

```python
import gensim.downloader as api
glove = api.load("glove-wiki-gigaword-100")
glove["cat"]
glove.most_similar("cat")
```

---

## 4. FastText (2016)

Same recipe as Word2Vec but trained on **subwords** (character n-grams).

- "playing" → "<pl", "pla", "lay", "ayi", "yin", "ing", "ng>"
- Sum subword vectors → word vector

Advantages:
- Handles **out-of-vocabulary** — even if "uncertainly" wasn't seen, its subwords were
- Better for morphologically rich languages (German, Turkish, Finnish)
- Better for misspellings ("recieve" gets a vector close to "receive")

```python
import gensim.downloader as api
ft = api.load("fasttext-wiki-news-subwords-300")
ft["uncertainly"]               # works even if not in vocab
```

---

## 5. Pre-trained embeddings worth knowing

| Model | Dim | Source | Notes |
|---|---|---|---|
| GloVe (Wikipedia + Gigaword) | 50, 100, 200, 300 | Stanford | Classic |
| GloVe (Common Crawl) | 300 | Stanford | Bigger corpus |
| Word2Vec (Google News) | 300 | Google | Original |
| FastText (Wikipedia, multilingual) | 300 | Facebook | OOV-friendly |
| ConceptNet Numberbatch | 300 | ConceptNet | Includes commonsense relations |

### For 2025 production: contextual embeddings beat these
- BERT / DistilBERT — covered in `04-sequence/05-bert-huggingface.md`
- `sentence-transformers` MiniLM, BGE — for sentence-level similarity
- text-embedding-3-small (OpenAI), Cohere embeddings, Voyage — for RAG

But Word2Vec/GloVe are still useful for:
- **Educational** (embedding intuition)
- **Lightweight** apps (no GPU needed)
- **Initialization** for downstream models
- **Edge** deployments

---

## 6. Using embeddings in your model

### As input to a NN
```python
import torch.nn as nn

embed_layer = nn.Embedding.from_pretrained(
    torch.tensor(pretrained_matrix, dtype=torch.float32),
    freeze=False,                      # allow fine-tuning
)
```

### As average for sentence embedding (crude but useful)
```python
def sentence_vec(words, kv):
    vecs = [kv[w] for w in words if w in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(kv.vector_size)
```

The "averaged Word2Vec" sentence vector is a decent baseline for similarity search before reaching for sentence-transformers.

---

## 7. Visualizing embeddings — t-SNE / UMAP

```python
import gensim.downloader as api
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

glove = api.load("glove-wiki-gigaword-100")
words = ["king", "queen", "man", "woman", "prince", "princess",
         "dog", "cat", "puppy", "kitten",
         "italy", "rome", "france", "paris"]
vecs = np.array([glove[w] for w in words])
proj = TSNE(n_components=2, perplexity=5).fit_transform(vecs)

plt.scatter(proj[:, 0], proj[:, 1])
for i, w in enumerate(words):
    plt.annotate(w, proj[i])
```

You'll see country-capital pairs cluster, gender pairs align, animal-baby pairs align. Excellent demo for any audience.

---

## 8. Caveats — what embeddings get wrong

- **Homographs**: "bank" (river) and "bank" (financial) collapse to one vector. Contextual embeddings (BERT) fix this.
- **Bias**: trained on internet text → reproduce gender, racial, cultural bias. Documented and active research area.
- **Stable vocabulary**: vocabulary is fixed at training time. Subword tokenizers + transformer models handle vocab dynamically.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Using GloVe vectors with case-sensitive task | "Apple" missing | use lowercase + GloVe; or BERT |
| Confusing one-hot and embedding | conceptual error | one-hot is sparse; embedding is dense |
| Averaging Word2Vec for sentence sim | weak baseline | use sentence-transformers |
| Training Word2Vec on tiny corpus | meaningless vectors | use pre-trained on large corpus |
| Forgetting OOV handling | NaN propagates | check `w in kv` first |

## Self-check

- [ ] Why do we need word embeddings at all (vs one-hot)?
- [ ] What's the famous "king − man + woman ≈ queen" example showing?
- [ ] Difference between Word2Vec Skip-gram and CBOW?
- [ ] How does FastText handle OOV?
- [ ] Train Word2Vec on a list of tokenized sentences using gensim.
- [ ] Why don't Word2Vec/GloVe handle homographs well?
- [ ] When still use Word2Vec in 2025?
- [ ] What replaces Word2Vec for modern sentence-similarity tasks?
