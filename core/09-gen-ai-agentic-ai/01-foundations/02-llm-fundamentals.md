# Foundations 2 — LLM Fundamentals

## Lectures covered
- Large Language Models

---

## 1. What an LLM is, in one paragraph

An LLM is a **transformer decoder** trained on huge amounts of text to predict the **next token** given previous tokens. After enough scale (parameters + data + compute), the model exhibits emergent capabilities — reasoning, instruction-following, world knowledge.

> "Just" predicting the next token, but at scale, that becomes useful for almost everything textual.

---

## 2. The training pipeline (3 stages)

### 1. Pre-training
- Trillions of tokens of internet, books, code
- Self-supervised: predict next token
- Output: a "base model" that completes text
- Cost: $millions to $billions

### 2. Instruction tuning (SFT — Supervised Fine-Tuning)
- Curated (instruction, response) pairs
- Teaches the base model to **follow instructions** rather than blindly continue text
- Output: an instruction-following model

### 3. RLHF / DPO (alignment)
- Humans rank responses; model trained to prefer the preferred ones
- Output: a helpful, harmless, honest assistant
- Modern variants: **DPO** (Direct Preference Optimization), **RLAIF**

After all three: ChatGPT, Claude, Gemini.

---

## 3. Tokens — the unit of LLM accounting

LLMs don't see characters or words — they see **tokens**, ~3-4 characters each on average for English. Different languages tokenize differently.

```
"hello world" → ["hello", " world"]  → 2 tokens
"def my_func():" → ["def", " my", "_", "func", "():"] → ~5 tokens
"নমস্কার" (Bengali) → could be 5-10 tokens
```

### Why this matters for cost

API pricing is **per token** — input + output separately. Examples (rough, late 2025):

| Model | Input $/M tokens | Output $/M tokens |
|---|---|---|
| Claude Opus 4.7 | $15 | $75 |
| Claude Sonnet 4.6 | $3 | $15 |
| Claude Haiku 4.5 | $1 | $5 |
| GPT-4o | $2.50 | $10 |
| GPT-4o-mini | $0.15 | $0.60 |
| Llama 3.1 70B (Together) | $0.90 | $0.90 |

> A single 50-message chat with a 100k-token document loaded can easily cost $1+. For high-traffic apps, cost matters a lot.

### Tokenizing in code
```python
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o")
n_tokens = len(enc.encode("hello world"))
```

For Anthropic:
```python
import anthropic
client = anthropic.Anthropic()
client.messages.count_tokens(model="claude-sonnet-4-6", messages=[...])
```

---

## 4. Roles — system / user / assistant

Modern chat LLMs accept a **list of messages** with roles:

```python
messages = [
    {"role": "system", "content": "You are a helpful data assistant."},
    {"role": "user", "content": "What is mean vs median?"},
    {"role": "assistant", "content": "Mean is the arithmetic average..."},
    {"role": "user", "content": "Now give me an example."},
]
```

- **system** — instructions for the assistant; sets behavior
- **user** — what the human said
- **assistant** — what the model previously said (for multi-turn context)

The model sees the entire conversation each call → must fit in context window.

---

## 5. Streaming responses

For better UX, stream tokens as they're generated:

```python
# OpenAI
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[...],
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

```python
# Anthropic
with client.messages.stream(model="claude-sonnet-4-6", max_tokens=1024,
                              messages=[...]) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

Streamlit and the AI SDK make streaming UIs trivial.

---

## 6. Function / tool calling

Modern LLMs can call **tools** — JSON-defined functions. The model decides when to call, you execute, return result, model continues.

```python
tools = [{
    "name": "get_weather",
    "description": "Get current weather in a given city",
    "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"],
    },
}]

resp = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Lahore?"}],
)
# resp may contain a tool_use block requesting get_weather(city="Lahore")
# you call the real function, then send the result back as a tool_result
```

This is the foundation of **agents**.

---

## 7. Common LLM "modes" you'll write apps for

| Mode | What |
|---|---|
| **One-shot completion** | one prompt, one response |
| **Chat** | multi-turn conversation with history |
| **Tool use** | model decides when to call functions |
| **Structured output** | model returns JSON matching a schema |
| **Streaming** | tokens come as they're generated |
| **Embeddings** | text → vector, for similarity / RAG |
| **Prompt caching** | cache repeated context to save cost |

We'll use most of these in later subfolders.

---

## 8. Choosing a model — the practical heuristic

| Need | Pick |
|---|---|
| Highest quality, complex reasoning | Claude Opus, GPT-4o, Gemini 1.5 Pro |
| Balance quality + cost | Claude Sonnet, GPT-4o, Gemini 1.5 Flash |
| Cheapest workhorse | Claude Haiku, GPT-4o-mini, Llama 3.1 8B |
| Coding tasks | Claude Sonnet (industry leader for code), GPT-4o |
| Multimodal (image + text) | Claude, GPT-4o, Gemini |
| Long context (>200k) | Claude Sonnet, Gemini 1.5 |
| Privacy / on-prem | Llama / Mistral / Qwen via Ollama or vLLM |
| Fastest inference | Groq-hosted Llama / Mixtral |

For Codebasics' projects: usually a mid-tier model (Sonnet, GPT-4o-mini) is the sweet spot.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Sending huge contexts every call | high cost | use prompt caching where supported |
| Treating LLM as deterministic | non-reproducible | set temperature=0 for testing |
| Trusting LLM math | LLMs are bad at arithmetic | tool-call out to a calculator |
| Trusting LLM with private data | data goes to vendor | use enterprise terms or self-host |
| No max_tokens | runaway response | set a sensible cap |

## Self-check

- [ ] Three stages of LLM training?
- [ ] How are LLMs charged — per what unit?
- [ ] What does the system message do?
- [ ] Difference between streaming and non-streaming?
- [ ] What's tool / function calling?
- [ ] Pick a model for: code generation / cheap classification / privacy-sensitive workload.
- [ ] Why is "trust LLM math" a bad idea?
- [ ] Count tokens for a string in your favorite model.
