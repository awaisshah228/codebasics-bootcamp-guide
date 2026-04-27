# Sequence 3 — Transformer Architecture

## Lectures covered
- Transformer Architecture
- Word Embeddings (intro — covered fully in NLP module)

---

## 1. Why Transformers replaced RNNs

| | RNN/LSTM | Transformer |
|---|---|---|
| Sequence processing | sequential | parallel |
| Long-range dependencies | weak (despite gating) | excellent |
| Train speed | slow (sequential) | fast (parallel on GPU) |
| Inference latency | low (per-token) | depends |
| State of the art | almost nowhere | everywhere |

Single insight: **attention is all you need** (Vaswani et al., 2017). Drop recurrence; use attention to let every token attend to every other token directly.

This launched: BERT, GPT, T5, BART, ViT, Whisper, AlphaFold, modern LLMs, modern image models.

---

## 2. The architecture (high-level)

The original Transformer is **encoder-decoder**:

```
       INPUT                           OUTPUT
       SEQUENCE                        SEQUENCE
          ↓                                ↓
     ┌─────────┐                      ┌─────────┐
     │ embed + │                      │ embed + │
     │  posi-  │                      │  posi-  │
     │  tional │                      │  tional │
     └────┬────┘                      └────┬────┘
          ↓                                ↓
   ┌─[Encoder layer × N]─┐         ┌─[Decoder layer × N]─┐
   │  - Self-attention   │   ───►  │  - Masked self-attn │
   │  - FFN              │   info  │  - Cross-attention  │  ← from encoder
   │  - Residual + LN    │         │  - FFN              │
   └─────────────────────┘         │  - Residual + LN    │
                                    └────┬────────────────┘
                                          ↓
                                       linear → softmax
                                          ↓
                                       output token probs
```

Modern variants:
- **Encoder-only** (BERT) → understanding tasks (classification, NER)
- **Decoder-only** (GPT) → generation
- **Encoder-decoder** (T5, BART) → translation / summarization

---

## 3. The encoder block

```
input (B, T, d_model)
   │
   ▼
┌────────────────────────────────────┐
│  Multi-Head Self-Attention         │
│  (Q, K, V from same input)         │
└──────────┬─────────────────────────┘
           │  + residual
           ▼
        LayerNorm
           │
           ▼
┌────────────────────────────────────┐
│  Position-wise Feed-Forward (MLP)  │
│  Linear → ReLU/GELU → Linear       │
└──────────┬─────────────────────────┘
           │  + residual
           ▼
        LayerNorm
           ▼
        output
```

Stack `N` of these (typically 6, 12, 24).

---

## 4. Positional encoding — putting order back in

Transformers process all positions in parallel → they have no built-in sense of order. Solution: **add positional encoding** to each token's embedding.

### Sinusoidal (original paper)
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

Each position gets a unique vector. Different positions are easily distinguishable; nearby positions are similar.

### Learned positional embeddings (BERT, GPT)
Just a learnable parameter per position. Works equally well, max length capped at training length.

### Modern (RoPE — Rotary Positional Embeddings)
Used in LLaMA, Mistral, etc. Encodes position by *rotating* the Q/K vectors by an angle proportional to position. Better at extrapolating beyond training length.

---

## 5. The decoder block

```
input (target tokens shifted right)
   │
   ▼
Masked Self-Attention (can only attend to past tokens)
   │  + residual + LN
   ▼
Cross-Attention (Q from decoder, K/V from encoder output)
   │  + residual + LN
   ▼
Feed-Forward (MLP)
   │  + residual + LN
   ▼
output
```

The "masked" attention is what makes generation autoregressive — token at position $t$ can only see tokens at positions ≤ t, never the future.

---

## 6. Multi-head attention (high level)

Instead of one attention computation, do N (e.g., 8) **in parallel**, each with its own Q/K/V projection. Concatenate the results.

Why: different heads can attend to different relationships (subject-verb, adjective-noun, distant-context, etc.).

Full attention math is in `04-attention.md`.

---

## 7. PyTorch Transformer

```python
import torch.nn as nn

# raw transformer encoder
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512, nhead=8, dim_feedforward=2048,
    dropout=0.1, batch_first=True,
)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

# full encoder-decoder
transformer = nn.Transformer(
    d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6,
    dim_feedforward=2048, dropout=0.1, batch_first=True,
)
```

In practice, we don't write this from scratch — we use HuggingFace pre-trained models (BERT, GPT-2, T5).

---

## 8. Why Transformers scale (the secret weapon)

- **Parallelizable** — process whole sequence at once on GPU
- **Constant path length** — any token can attend to any other in 1 op (vs O(seq_len) for RNN)
- **No vanishing gradient** — gradients flow through residuals
- **Compute scales** — performance keeps improving with more compute, more data, more parameters

This is why we have GPT-4 and Claude. RNNs *couldn't* be scaled this way.

---

## 9. Sizing reference

| Model | Layers | d_model | Heads | Params |
|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 110M |
| BERT-large | 24 | 1024 | 16 | 340M |
| GPT-2 small | 12 | 768 | 12 | 124M |
| GPT-2 large | 36 | 1280 | 20 | 774M |
| GPT-3 | 96 | 12288 | 96 | 175B |
| LLaMA 3 70B | 80 | 8192 | 64 | 70B |

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgetting positional encoding | model treats sequence as a bag | always include PE |
| Forgetting attention mask in decoder | sees the future → can't generate | use causal mask |
| Padding tokens attended | wrong attention weights | use padding mask |
| Trying to write Transformer from scratch | bugs, slow | use `nn.Transformer` or HuggingFace |
| Training a tiny Transformer on tiny data | overfits | use a pre-trained one |

## Self-check

- [ ] Why did Transformers replace RNNs?
- [ ] Three architectural variants and what each is used for?
- [ ] What's the role of positional encoding?
- [ ] What's the difference between self-attention and cross-attention?
- [ ] What's masked self-attention and why is it needed in the decoder?
- [ ] Why is multi-head attention better than single-head?
- [ ] Why are residuals + LayerNorm used heavily?
- [ ] Look up GPT-2 small's params: how many heads, layers, d_model?
