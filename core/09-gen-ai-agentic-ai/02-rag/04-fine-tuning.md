# RAG 4 — Fine-Tuning an LLM (and when not to)

## Lectures covered
- Fine Tuning an LLM

---

## 1. The decision framework

Most learners reach for fine-tuning when they should reach for prompting or RAG. Here's the right order:

```
Need new behavior or knowledge?
   │
   ▼
Try better prompts (zero/one/few-shot)
   │   not enough?
   ▼
Add RAG (retrieve relevant info)
   │   still not enough?
   ▼
Add tools / function calling
   │   still not enough?
   ▼
Fine-tune (small model, specific task)
```

### Fine-tune when
- You need a **specific style or format** consistently (your tone, your output schema)
- You want a **smaller model** (cheap + fast) to match a bigger model on a narrow task
- Your task is **highly domain-specific** with patterns the base model doesn't know
- You have **training data** (1k–100k examples)

### Don't fine-tune when
- You just need new facts → use RAG
- The data changes often → keep RAG
- You have <500 examples → few-shot is probably better
- You haven't tried prompt engineering yet → start there

---

## 2. Types of fine-tuning

### Full fine-tune
Update all parameters. Expensive, requires lots of GPU memory.

### LoRA (Low-Rank Adaptation)
Freeze the base model. Train tiny "adapter" matrices. ~1% of params, 95% of quality.

### QLoRA
LoRA on a 4-bit quantized base model. Lets you fine-tune 70B models on a single 24GB GPU.

### Instruction tuning
Take a base model + a dataset of (instruction, response) pairs → produce an instruction-following model.

### RLHF / DPO
Train from human preference data. Last alignment step. Not bootcamp territory.

For your projects: **LoRA on a smaller open-source model** (Llama 3 8B, Mistral 7B, Qwen 2.5 7B) is the sweet spot.

---

## 3. Fine-tuning small classifiers (BERT-class) — recap

For text classification, fine-tuning DistilBERT (already covered in `core/08-nlp/07-bert-finetuning-huggingface.md`) is super practical:
- ~5,000 examples → 92% sentiment accuracy
- 10 minutes on a free Colab GPU
- Tiny model → cheap to deploy

This **is** fine-tuning. Don't dismiss it as "old school."

---

## 4. Fine-tuning generative LLMs — when and how

### When (for the bootcamp's projects)
Mostly skip. Codebasics' Gen AI projects (real-estate, chatbot, agentic) are RAG + tools, not fine-tune.

### Where it's worth it
- Match a stronger model's outputs on a narrow domain (distillation)
- Force a precise output format
- Embed company tone / persona
- Custom code-completion model on your codebase

### Tooling — `unsloth` (modern fast fine-tune)
```bash
pip install unsloth
```
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16, lora_alpha=16, lora_dropout=0.0,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)
```

Fine-tune in ~30 min on a free Colab GPU.

### Tooling — HuggingFace `trl` SFTTrainer
```python
from trl import SFTTrainer

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=200,
        learning_rate=2e-4,
        bf16=True,
    ),
)
trainer.train()
```

### Format
Most fine-tuning datasets follow the chat-template format:
```jsonl
{"messages": [
  {"role": "system", "content": "..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

Hundreds-to-thousands of these. Train. Done.

---

## 5. Fine-tuning closed APIs (OpenAI, Anthropic-via-Bedrock, Gemini)

If you can't host a model, providers let you fine-tune theirs:

### OpenAI
```python
client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
client.fine_tuning.jobs.create(training_file="...", model="gpt-4o-mini-2024-07-18")
```

### Anthropic
Available on AWS Bedrock for Claude in some regions.

### Cost trade-off
Fine-tuned API models cost **3-5× more per token** than base. So fine-tune only if it lets you use a much smaller / cheaper model.

---

## 6. Evaluating fine-tunes

Same evaluation suite you'd use for RAG:
- Held-out test set with golden outputs
- LLM-as-judge for open-ended quality
- Specific metrics for your task (accuracy, BLEU, F1)
- Human spot-checks

**Always compare to**:
1. Base model with no fine-tune
2. Base model with prompt engineering
3. Base model with RAG

If your fine-tune doesn't beat all three, don't ship it.

---

## 7. Fine-tune vs RAG vs prompt — the comparison table

| | Prompt | RAG | Fine-tune |
|---|---|---|---|
| Adds new knowledge | weakly (in context) | yes (retrieval) | yes (in weights) |
| Reflects fresh data | yes | yes | no — need retraining |
| Changes behavior / style | partly | minimally | yes |
| Cost (one-time) | $0 | low (vector DB) | $$ training |
| Cost (per-query) | $$ context tokens | $ small + LLM | $$ per-token + base infra |
| Maintenance | low | low | high (retrain pipeline) |
| Best for | quick prototypes | private / fresh knowledge | style, format, narrow domain |

**RAG covers 80% of "new knowledge" needs. Fine-tuning covers 80% of "consistent behavior" needs.**

For Codebasics' Gen AI projects: prefer RAG + good prompts. Fine-tuning is a brief nod in the curriculum.

---

## 8. Combined patterns

### RAG + fine-tune
- Fine-tune for tone / format
- RAG for facts
Common in customer-support bots.

### Distillation
1. Run a big model on lots of inputs
2. Save (input, output) pairs
3. Fine-tune a small model on those
- Result: small model that mimics the big one, far cheaper to serve

### Self-improvement
1. Fine-tuned model V0
2. Run V0 in production, collect user feedback
3. Curate good outputs as new training data
4. Fine-tune V1
5. Repeat

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Fine-tuning before trying RAG / prompts | wasted effort | always go in order |
| <500 examples | overfitting / no improvement | use few-shot prompting instead |
| No held-out test set | overfit to train | always split |
| Wrong format (not chat-template) | model fits noise | use the model's official template |
| Fine-tune on noisy / wrong-format data | catastrophic | clean before training |
| Forgetting to save adapter only (LoRA) | huge files | save adapter, load with `peft` |

## Self-check

- [ ] When fine-tune vs RAG vs prompt?
- [ ] Difference between full fine-tune and LoRA?
- [ ] What's QLoRA?
- [ ] How many examples do I need for fine-tuning to beat few-shot?
- [ ] Walk through fine-tuning Llama 3 8B with `unsloth`.
- [ ] What's distillation and when use it?
- [ ] Why does fine-tuning a closed API model cost more per token than the base?
- [ ] If RAG + good prompts work, why bother fine-tuning?
