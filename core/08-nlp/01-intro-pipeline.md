# NLP 1 — Intro, Pipeline, Tools Overview

## Lectures covered
- Introduction to NLP
- NLP Pipeline
- Tools Overview

---

## 1. What NLP is (and isn't)

NLP = "make computers understand and produce human language."

Tasks:
- **Understand**: classify, extract entities, parse syntax, summarize, retrieve
- **Generate**: chat, translate, write code, write articles
- **Both**: question answering, dialogue systems

In 2025, the line between "NLP" and "Generative AI" has blurred — LLMs do most NLP tasks better than traditional pipelines. But NLP fundamentals still matter for:
- **Cost** — small classifiers are way cheaper than LLM calls
- **Latency** — DistilBERT is 100× faster than GPT-4
- **Privacy** — keep data on-prem
- **Reliability** — deterministic pipelines vs hallucination
- **As building blocks** — RAG, search, embedding, classification all use these tools

---

## 2. The classical NLP pipeline

```
raw text
  │
  ▼
1. Sentence segmentation     ("split this. into sentences." → ["split this.", "into sentences."])
  │
  ▼
2. Tokenization              ("hello world!" → ["hello", "world", "!"])
  │
  ▼
3. Lowercasing               (sometimes)
  │
  ▼
4. Stop-word removal         (remove "the", "a", "is")
  │
  ▼
5. Stemming or lemmatization (reduce words to roots: "running" → "run")
  │
  ▼
6. POS tagging               (tag each token: noun, verb, adj)
  │
  ▼
7. Named Entity Recognition  (find PERSON, ORG, DATE, LOCATION)
  │
  ▼
8. Parsing                   (build dependency or constituency tree)
  │
  ▼
9. Vectorize                 (TF-IDF, word embeddings, contextual embeddings)
  │
  ▼
10. Downstream task          (classification, NER, similarity, search, ...)
```

Modern transformer-based NLP **skips most of these** — the model learns implicitly. But knowing them lets you mix-and-match.

---

## 3. Tool landscape

### NLTK (Natural Language Toolkit)
- The original Python NLP library (2001)
- Great for **education** — every algorithm is exposed
- Lots of corpora, stemmers, classic features
- Slow for production

```python
import nltk
nltk.download("punkt")
from nltk.tokenize import word_tokenize, sent_tokenize
```

### spaCy
- **Production-grade** pipelines: tokenization, POS, NER, dependency parsing — fast
- Loads pre-built language models (English, multilingual, etc.)
- Great for: extracting structured info from text at scale

```python
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is opening a store in San Francisco next year.")
for ent in doc.ents:
    print(ent.text, ent.label_)
# Apple ORG
# San Francisco GPE
# next year DATE
```

### gensim
- Topic modeling (LDA), Word2Vec, Doc2Vec, FastText
- Great for: training your own embeddings, topic models on document corpora

### scikit-learn
- `CountVectorizer`, `TfidfVectorizer` — classic text-to-features
- Plug into any sklearn classifier

### HuggingFace `transformers`
- The standard for **modern, transformer-based NLP**
- 100,000+ pre-trained models
- Pipelines API for instant inference

```python
from transformers import pipeline
clf = pipeline("sentiment-analysis")
clf("I love this bootcamp!")
```

### `sentence-transformers`
- Sentence embeddings (semantic similarity, retrieval)
- Foundation for RAG

```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(["sentence one", "sentence two"])
```

### Specialized
- **fast.ai** — high-level wrapper around PyTorch for NLP/CV
- **Stanza** (Stanford) — alternative to spaCy with stronger linguistic features
- **flair** — word + sentence embeddings, NER

---

## 4. When to use which (quick decision)

| Task | Default tool |
|---|---|
| Tokenize, lowercase, stop-words | NLTK or spaCy |
| Industrial NER, POS, parsing | spaCy |
| TF-IDF + simple classifier | scikit-learn |
| Word embeddings from your own corpus | gensim |
| Sentence embeddings / semantic search | `sentence-transformers` |
| Sentiment / classification with SOTA accuracy | HuggingFace + DistilBERT |
| Chat / generation / agents | LLM API (OpenAI / Anthropic) — Module 9 |

---

## 5. Languages — multilingual considerations

- **English-only models** (DistilBERT-base) → smallest, fastest
- **Multilingual** (`xlm-roberta-base`, `bge-m3`) → for Asian/European/African scripts
- **Indic-specific** (e.g., MuRIL for Indian languages) → if working in Hindi, Urdu, Tamil, Bengali, etc.

For Pakistani / Indian context: MuRIL or XLM-R are good defaults.

---

## 6. NLP-specific evaluation

- **Classification** — accuracy, F1, macro-F1
- **NER** — exact-match F1 (entity-level)
- **Translation** — BLEU, chrF, COMET
- **Summarization** — ROUGE-1, ROUGE-2, ROUGE-L
- **Embedding quality** — STS Benchmark (Semantic Textual Similarity)
- **QA (extractive)** — exact match, F1
- **LLM eval** — human eval, LLM-as-judge, RAGAS (for RAG)

---

## 7. The 2025 NLP project mental model

```
┌────────────────────────────────────────────┐
│  Stage 1: prompt an LLM                    │   <- start here
│  (works for many tasks zero-shot)          │
└────────────────────────────────────────────┘
            │   not good enough?
            ▼
┌────────────────────────────────────────────┐
│  Stage 2: few-shot prompting + RAG          │
│  (give context + examples)                  │
└────────────────────────────────────────────┘
            │   still not good?
            ▼
┌────────────────────────────────────────────┐
│  Stage 3: fine-tune small model             │   <- BERT-class
│  (cheaper, faster, more controllable)       │
└────────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────┐
│  Stage 4: classical NLP pipeline            │   <- only when previous fail
│  (hand-crafted, fully controllable)         │
└────────────────────────────────────────────┘
```

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Lowercasing for case-sensitive tasks (NER) | drops signal | only lowercase for tasks that benefit |
| Removing stop-words for sentiment | "not good" → "good" | be careful; stop-words depend on task |
| Using NLTK in production | slow | spaCy or HuggingFace |
| Same tokenizer for all models | wrong vocab | always pair tokenizer + model from same checkpoint |
| Wrong evaluation metric | optimizing wrong thing | match metric to task (F1 for NER, BLEU for translation) |

## Self-check

- [ ] List the steps of the classical NLP pipeline.
- [ ] When use NLTK vs spaCy?
- [ ] What's `sentence-transformers` used for?
- [ ] When prefer fine-tuning a small model over prompting an LLM?
- [ ] What metric for: news classification / NER / translation / RAG?
- [ ] Why is "remove stop-words" not always a good idea?
- [ ] What languages should I worry about for South Asian text?
- [ ] Walk through the 2025 mental model — which stage do you start at?
