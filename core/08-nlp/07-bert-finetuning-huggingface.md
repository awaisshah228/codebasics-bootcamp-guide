# NLP 7 — HuggingFace Pipelines, Tokenizers, BERT Fine-Tuning

## Lectures covered
- Introduction to Hugging Face
- Pipelines and Tokenizers
- BERT and Model Fine Tuning

---

## 1. The Hugging Face universe

Three core libraries:

### `transformers`
Pre-trained models for every NLP / vision / audio task. The standard.

### `datasets`
Standardized datasets with caching, streaming, splits.

### `tokenizers`
Fast Rust-based tokenizers. Often used through `transformers` indirectly.

Plus: `accelerate` (distributed training), `peft` (parameter-efficient fine-tuning), `evaluate` (metrics), `trl` (RLHF), `diffusers` (image generation).

```bash
pip install transformers datasets tokenizers
```

---

## 2. Pipelines — instant inference

```python
from transformers import pipeline

# sentiment
clf = pipeline("sentiment-analysis")
clf("I love this bootcamp")
# [{'label': 'POSITIVE', 'score': 0.9997}]

# zero-shot classification (no training!)
zsc = pipeline("zero-shot-classification")
zsc("I want to deposit money", candidate_labels=["banking", "shopping", "travel"])
# {'labels': ['banking', 'shopping', 'travel'], 'scores': [0.94, 0.04, 0.02]}

# NER
ner = pipeline("ner", aggregation_strategy="simple")
ner("Awais lives in Lahore and works at Codebasics.")

# Question answering
qa = pipeline("question-answering")
qa(question="Who created Codebasics?", context="Dhaval Patel founded Codebasics in 2016.")

# Summarization
summ = pipeline("summarization")
summ("very long article ...")

# Translation
tr = pipeline("translation_en_to_de")
tr("Hello, how are you?")

# Text generation
gen = pipeline("text-generation", model="gpt2")
gen("The bootcamp will", max_length=30, num_return_sequences=2)

# Fill-in-the-blank
mlm = pipeline("fill-mask")
mlm("The cat sat on the [MASK].")
```

For a quick demo or prototype, `pipeline` is unbeatable.

---

## 3. Tokenizers (a closer look)

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

inputs = tok(
    "Hello, my dog is cute",
    padding="max_length",
    truncation=True,
    max_length=64,
    return_tensors="pt",
)
# inputs.input_ids:      (1, 64) — token IDs
# inputs.attention_mask: (1, 64) — 1 for real, 0 for padding
```

### Useful methods
```python
tok.tokenize("unbelievable")           # ['un', '##bel', '##iev', '##able']
tok.encode("hello")                     # token IDs
tok.decode([7592, 1010, 2026])          # back to text
tok.convert_ids_to_tokens([7592])
```

### Special tokens (BERT-family)
- `[CLS]` (id 101) — start of every sequence
- `[SEP]` (id 102) — separator (between sentence pairs)
- `[PAD]` (id 0) — padding
- `[UNK]` — unknown
- `[MASK]` — for MLM

For pairs of sentences:
```python
tok("first sentence", "second sentence", padding=True, truncation=True, return_tensors="pt")
# token_type_ids: 0s for first, 1s for second
```

---

## 4. The full fine-tuning recipe (DistilBERT, sentiment / classification)

```python
import numpy as np
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                           Trainer, TrainingArguments, DataCollatorWithPadding)
from datasets import load_dataset
import evaluate

# 1. Data
ds = load_dataset("imdb")               # binary sentiment
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch): return tok(batch["text"], truncation=True, max_length=256)
ds = ds.map(tokenize, batched=True)
ds = ds.rename_column("label", "labels")

collate = DataCollatorWithPadding(tokenizer=tok)

# 2. Model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

# 3. Metrics
acc_metric = evaluate.load("accuracy")
def compute_metrics(p):
    preds = p.predictions.argmax(-1)
    return acc_metric.compute(predictions=preds, references=p.label_ids)

# 4. Trainer
args = TrainingArguments(
    output_dir="out",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=ds["train"].shuffle(seed=42).select(range(5000)),    # subsample for speed
    eval_dataset=ds["test"].shuffle(seed=42).select(range(1000)),
    tokenizer=tok,
    data_collator=collate,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model("models/distilbert-imdb")
```

5,000 training examples on a free Colab T4 → ~92% accuracy in ~10 min.

---

## 5. Inference after fine-tuning

```python
from transformers import pipeline
clf = pipeline("text-classification", model="models/distilbert-imdb", tokenizer=tok)
clf(["I love this", "I hate this"])
```

Or push to the Hub and load from anywhere:
```bash
huggingface-cli login
```
```python
trainer.push_to_hub()
```

---

## 6. PEFT / LoRA — fine-tuning huge models cheaply

For full LLMs (Llama, Mistral), full fine-tuning is expensive. **LoRA** (Low-Rank Adaptation) freezes the base model and trains only small "adapter" matrices — 1% of parameters, near-equivalent quality.

```bash
pip install peft
```
```python
from peft import LoraConfig, get_peft_model, TaskType

config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1)
model = get_peft_model(model, config)
```

For the bootcamp's BERT-class projects, you don't strictly need LoRA — full fine-tune fits on free Colab. Mention it for context.

---

## 7. Common HuggingFace tasks & model picks

| Task | Recommended model | Fine-tune? |
|---|---|---|
| Binary sentiment | `distilbert-base-uncased-finetuned-sst-2-english` | already fine-tuned |
| Multi-class topic | `distilbert-base-uncased` | yes, on your data |
| NER (English) | `dslim/bert-base-NER` | already fine-tuned |
| QA (extractive) | `deepset/roberta-base-squad2` | already fine-tuned |
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | usually no |
| Multilingual | `xlm-roberta-base` | yes for downstream |
| Summarization | `facebook/bart-large-cnn`, `google/flan-t5-base` | depending |
| Translation | `Helsinki-NLP/opus-mt-en-de` | rarely |
| Code | `Salesforce/codet5-base`, `bigcode/starcoder2-3b` | depending |

For Indic / Urdu / Hindi: `ai4bharat/IndicBERT`, `MuRIL`, `xlm-roberta-base`.

---

## 8. Saving / loading / sharing

```python
trainer.save_model("models/my-model")
tok.save_pretrained("models/my-model")

# load anywhere
model = AutoModelForSequenceClassification.from_pretrained("models/my-model")
tok = AutoTokenizer.from_pretrained("models/my-model")
```

To share publicly:
```python
trainer.push_to_hub("myuser/my-imdb-classifier")
```

People can load it with `AutoModel.from_pretrained("myuser/my-imdb-classifier")`. This is **massive** for a portfolio — your name appears on the Hub model page.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Tokenizer from one model + weights from another | wrong embeddings | always pair them |
| Training on CPU | hours per epoch | move to Colab GPU / Kaggle |
| Forgot `padding`/`truncation` | mismatched batch shapes | always set them |
| OOM during training | batch too large | reduce batch size; use gradient accumulation |
| Saving full model when only adapters needed | huge files | use LoRA / PEFT save |
| `push_to_hub` not authenticated | upload fails | `huggingface-cli login` first |

## Self-check

- [ ] Build a sentiment pipeline in 3 lines.
- [ ] What does `padding="max_length"` do?
- [ ] Why use `[CLS]` token's hidden state for classification?
- [ ] Walk through fine-tuning DistilBERT on a custom dataset.
- [ ] Difference between `AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`?
- [ ] When use LoRA / PEFT?
- [ ] Push a fine-tuned model to the Hub and load it back.
- [ ] What's `evaluate` used for?
