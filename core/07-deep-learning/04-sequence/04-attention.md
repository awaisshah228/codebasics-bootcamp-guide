# Sequence 4 — Attention Mechanism

## Lectures covered
- Attention Mechanism

---

## 1. The intuition

When you read a sentence, you don't process every word with equal attention. To understand "the dog **bit** the man," "bit" depends heavily on "dog" (subject) and "man" (object) — less on "the".

Attention mathematically captures this: each token assigns weights to every other token, decides what's relevant for understanding itself, and creates a weighted summary.

---

## 2. Query, Key, Value — the three projections

For each token, project its embedding into three vectors:
- **Query (Q)**: "what am I looking for?"
- **Key (K)**: "what do I have to offer?"
- **Value (V)**: "what's my actual content?"

A token's attention output = weighted sum of all tokens' values, where the weight is determined by **how much its query matches each token's key**.

---

## 3. Scaled dot-product attention — the formula

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

Steps:
1. **Compute scores**: $QK^T$ — dot products of every Q with every K. Result: (T, T) matrix of compatibility scores.
2. **Scale**: divide by $\sqrt{d_k}$ — keeps softmax in a healthy range as $d_k$ grows.
3. **Softmax**: rows now sum to 1 — these are attention weights.
4. **Weighted sum**: multiply by V to get the output.

Output shape: same as V (number of tokens × value dim).

### Why $\sqrt{d_k}$
Dot products of high-dim vectors have variance proportional to $d_k$. Without scaling, softmax becomes near-binary (one entry = 1, others = 0) → vanishing gradients in attention weights. Dividing by $\sqrt{d_k}$ stabilizes this.

---

## 4. Walking through one token

Suppose token "bit" is at position 2:
- Compute its Q vector: $q_2$
- Compute K vectors for all tokens: $k_0, k_1, k_2, k_3, k_4$ ("the", "dog", "bit", "the", "man")
- Dot products: $q_2 \cdot k_0$, $q_2 \cdot k_1$, ..., $q_2 \cdot k_4$ → 5 scores
- Scale + softmax → 5 weights summing to 1
- Output for "bit" = weighted sum of $v_0, v_1, ..., v_4$

The model learns Q/K/V projections such that "bit" attends most to "dog" and "man".

---

## 5. Self-attention vs cross-attention

### Self-attention
Q, K, V all come from the **same** input. Each token attends to others in its own sequence.
- Used in: encoder layers, decoder's first attention sublayer

### Cross-attention
Q comes from one sequence, K/V from **another**.
- Used in: decoder's second attention sublayer (Q = decoder, K/V = encoder output)
- Example: translation — decoder generates English, attending back to the French encoder output

### Masked self-attention
Same as self-attention but with a mask preventing attending to future tokens. Used in decoders for autoregressive generation.

```python
# causal mask: lower triangular
mask = torch.tril(torch.ones(T, T))
# scores where mask=0 → -inf → softmax → 0
```

---

## 6. Multi-head attention

Run the attention computation `h` times in parallel, each with its own Q/K/V projections.

```
input → split into h heads
   ├── head 1: Q₁ K₁ V₁ → out₁
   ├── head 2: Q₂ K₂ V₂ → out₂
   ├── ...
   └── head h: Qₕ Kₕ Vₕ → outₕ

concat(out₁, ..., outₕ) → linear → final output
```

Each head's projection dim is `d_model / h`, so the total compute is roughly the same as single-head with full dim — but the model can attend to different relationship types simultaneously.

```python
# typical: d_model=512, h=8 → each head has d_k=64
```

### What heads typically learn
- Some attend to syntactic structure (subject-verb)
- Some attend to coreference ("she" → previous noun)
- Some attend to delimiters / punctuation
- Some are nearly redundant (research shows you can prune ~40% of heads with little loss)

---

## 7. PyTorch implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, d = x.shape
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, T, d_k)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = Q @ K.transpose(-2, -1) / self.d_k ** 0.5                  # (B, h, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = weights @ V                                                    # (B, h, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, d)                 # (B, T, d_model)
        return self.W_o(out)
```

PyTorch ships an optimized version: `nn.MultiheadAttention(d_model, n_heads, batch_first=True)`.

---

## 8. Computational cost — the scaling problem

Self-attention is O(T²) in sequence length (every token attends to every other).

For T=1000: 1M attention scores per layer × heads × batch.
For T=100,000: 10B scores. Doesn't fit.

This drives many "long-context" research directions:
- **Sparse attention** (Longformer, BigBird) — each token attends to a subset
- **Linear attention** (Performer, Linformer)
- **State-space models** (Mamba) — alternative to attention
- **Sliding-window attention** (Mistral)
- **Flash Attention** — same math, faster + lower memory implementation

For the bootcamp's needs (≤512 tokens, BERT-style), classic attention is fine.

---

## 9. Visualizing attention

After fine-tuning a Transformer, you can plot attention weights to see what tokens attend to what:

```python
# from a HuggingFace model with output_attentions=True
outputs = model(input_ids, output_attentions=True)
attentions = outputs.attentions          # tuple of (B, heads, T, T) per layer

import seaborn as sns
sns.heatmap(attentions[6][0, 3].cpu(), xticklabels=tokens, yticklabels=tokens)
```

This is the visualization that made attention "the new chunked LSTM hidden state" — interpretable in a way RNNs aren't.

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to scale by $\sqrt{d_k}$ | softmax saturates | always scale |
| Wrong mask shape | "softmax over -inf" → NaN | check shapes |
| Forgot causal mask in decoder | sees future | always mask in autoregressive |
| Using the same dim for Q/K/V across heads without splitting | head doing nothing | split d_model across heads |
| Dropout removed from attention weights | risk of overfit on small data | `nn.MultiheadAttention(dropout=0.1)` |

## Self-check

- [ ] Walk through self-attention computation for one token.
- [ ] What does Q vs K vs V represent intuitively?
- [ ] Why divide by $\sqrt{d_k}$?
- [ ] What's the difference between self-attention and cross-attention?
- [ ] Why use multi-head attention instead of single-head?
- [ ] What's the masked attention used in decoders for?
- [ ] Implement scaled dot-product attention from scratch in PyTorch.
- [ ] Why is attention O(T²) — and what are some workarounds for long contexts?
