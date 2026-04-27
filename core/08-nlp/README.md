# Module 8 — Natural Language Processing (NLP)

> **Status**: 🔒 Locked
> **Tools**: spaCy · NLTK · scikit-learn · HuggingFace transformers

## Why this module exists separately from DL

NLP has its own pipeline (text preprocessing) and many classical methods (TF-IDF, n-grams, NER) that still matter even in the LLM era — for cost, latency, interpretability, and as building blocks for RAG / search.

The bootcamp covers NLP **end-to-end**: classical preprocessing → embeddings → classification → BERT fine-tuning.

## Curriculum (verbatim from brochure)

- Introduction to NLP · NLP Pipeline · Tools Overview
- Tokenization Techniques
- Stemming & Lemmatization
- Part of Speech (POS) Tagging
- Stop Words
- Named Entity Recognition (NER)
- Regular Expressions (Regex)
- Bag of Words, n-grams, TF-IDF
- Word Embeddings (Word2Vec, GloVe, FastText)
- News Classification with Spacy
- Introduction to Hugging Face
- Pipelines and Tokenizers
- BERT and Model Fine Tuning

## Files

| # | File | Topic |
|---|---|---|
| 01 | [intro-pipeline.md](01-intro-pipeline.md) | What is NLP · the pipeline · tools (NLTK · spaCy · HF) |
| 02 | [text-preprocessing.md](02-text-preprocessing.md) | Tokenization · stemming · lemmatization · stop words |
| 03 | [pos-ner-regex.md](03-pos-ner-regex.md) | POS tagging · NER · regex for extraction |
| 04 | [bow-ngrams-tfidf.md](04-bow-ngrams-tfidf.md) | Bag of Words · n-grams · TF-IDF · `CountVectorizer` |
| 05 | [word-embeddings.md](05-word-embeddings.md) | Word2Vec · GloVe · FastText · gensim |
| 06 | [news-classification-spacy.md](06-news-classification-spacy.md) | End-to-end news classification |
| 07 | [bert-finetuning-huggingface.md](07-bert-finetuning-huggingface.md) | HuggingFace pipelines · fine-tuning BERT |

## Module-level goal

After this module:
- Process raw text into ML-ready features
- Build a sentiment / news classifier with multiple approaches (TF-IDF + Logistic vs DistilBERT) and explain trade-offs
- Use spaCy for production NER / POS pipelines
- Fine-tune a HuggingFace transformer for any text-classification task
- Set yourself up perfectly for the Gen AI / RAG module that follows

## Module self-check

- [ ] Difference between stemming and lemmatization?
- [ ] Why TF-IDF still matters in 2025?
- [ ] Walk through self-attention for text.
- [ ] How to handle out-of-vocabulary words (modern: BPE / WordPiece tokenization).
- [ ] When use spaCy vs HuggingFace?
- [ ] Build a sentiment classifier with TF-IDF + logistic regression.
- [ ] Fine-tune DistilBERT on a small custom dataset.
- [ ] What's the connection between this module and the upcoming Gen AI module?
