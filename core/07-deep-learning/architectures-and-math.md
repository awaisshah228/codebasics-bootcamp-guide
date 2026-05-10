# Deep Learning — Architectures & Math (Visual + Formula Reference)

A single-page reference covering:

1. Regular (Feedforward) Neural Network — diagram + math
2. Recurrent Neural Network — **folded** form (compact view)
3. Recurrent Neural Network — **unfolded** form (across time)
4. The forward-pass calculations behind each
5. Worked numerical example: predicting the next word
6. Backpropagation intuition (gradient descent + BPTT)
7. Quick-reference cheat sheet

> All diagrams are ASCII for clarity. Copy them into notebooks freely.

---

## 1. Regular Feedforward Neural Network (FFN / MLP)

### Architecture diagram

```
   Input Layer        Hidden Layer 1     Hidden Layer 2     Output Layer
   (3 features)        (4 neurons)        (4 neurons)        (2 classes)

        x₁ ───┐ ┌────► h₁⁽¹⁾ ───┐ ┌──► h₁⁽²⁾ ───┐ ┌──► y₁
              │ │                │ │              │ │
        x₂ ───┼─┼────► h₂⁽¹⁾ ───┼─┼──► h₂⁽²⁾ ───┼─┼──► y₂
              │ │                │ │              │ │
        x₃ ───┘ │      h₃⁽¹⁾ ───┘ │    h₃⁽²⁾ ───┘ │
                └────► h₄⁽¹⁾      └──► h₄⁽²⁾      │
                                                   │
              W⁽¹⁾, b⁽¹⁾        W⁽²⁾, b⁽²⁾        W⁽³⁾, b⁽³⁾
              (weights matrix per layer; every line = one weight)
```

Every line in the diagram is a **weight**. Every node applies an **activation function** after summing its inputs.

### Forward-pass math (one layer at a time)

For a layer `l` with input vector `a⁽ˡ⁻¹⁾` and weight matrix `W⁽ˡ⁾`, bias `b⁽ˡ⁾`:

```
z⁽ˡ⁾ = W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾          ← linear combination
a⁽ˡ⁾ = σ( z⁽ˡ⁾ )                      ← non-linearity (ReLU, sigmoid, etc.)
```

Stack these for every layer:

```
a⁽⁰⁾ = x                                           (input)
a⁽¹⁾ = ReLU( W⁽¹⁾ · a⁽⁰⁾ + b⁽¹⁾ )
a⁽²⁾ = ReLU( W⁽²⁾ · a⁽¹⁾ + b⁽²⁾ )
ŷ    = softmax( W⁽³⁾ · a⁽²⁾ + b⁽³⁾ )               (output)
```

### Per-neuron view (what each circle does)

```
        x₁ ──w₁──┐
                 │
        x₂ ──w₂──┼──► Σ ──► z = w₁x₁ + w₂x₂ + w₃x₃ + b ──► σ(z) ──► output
                 │
        x₃ ──w₃──┘
                 ▲
                 b  (bias)
```

### Common activations

| Name | Formula | Range |
|------|---------|-------|
| Sigmoid | `1 / (1 + e⁻ᶻ)` | (0, 1) |
| Tanh | `(eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)` | (−1, 1) |
| ReLU | `max(0, z)` | [0, ∞) |
| Softmax | `eᶻⁱ / Σ eᶻⱼ` | sums to 1 (probability) |

### Tiny worked example (FFN forward pass)

Inputs: `x = [1.0, 2.0]`, hidden size = 2, output size = 1.

```
W⁽¹⁾ = [[0.5, -0.2],            b⁽¹⁾ = [0.1, 0.0]
        [0.3,  0.8]]

z⁽¹⁾ = W⁽¹⁾ · x + b⁽¹⁾
     = [[0.5·1 + (-0.2)·2 + 0.1],
        [0.3·1 +    0.8 ·2 + 0.0]]
     = [0.2, 1.9]

a⁽¹⁾ = ReLU([0.2, 1.9]) = [0.2, 1.9]

W⁽²⁾ = [0.7, -0.4],             b⁽²⁾ = 0.05
z⁽²⁾ = 0.7·0.2 + (-0.4)·1.9 + 0.05 = -0.57
ŷ    = sigmoid(-0.57) ≈ 0.361
```

That's a complete forward pass. Backprop computes how each weight should change to reduce error — covered in section 6.

---

## 2. RNN — Folded (Compact) View

This is how RNNs are usually drawn. The self-loop is the recurrence:

```
                ┌─────────────────┐
                │                 │
                │    ┌────────┐   │
                └───►│   h    │───┘   ← hidden state loops
                     └───┬────┘         back to itself
                         │
                  ┌──────▼──────┐
              x ─►│  RNN cell   │─► y
                  └─────────────┘
```

The folded view is compact but **hides what's really happening across time**. To see the actual computation, we unfold it.

---

## 3. RNN — Unfolded (Across Time) View

Unrolling the loop reveals one cell per timestep — but **all cells share the same weights** (`W_xh`, `W_hh`, `W_hy`):

```
            t=1            t=2            t=3            t=4

  h₀  ──►  ┌────┐  h₁  ──►┌────┐  h₂  ──►┌────┐  h₃  ──►┌────┐  h₄
           │RNN │ ───────►│RNN │ ───────►│RNN │ ───────►│RNN │
           └─┬──┘         └─┬──┘         └─┬──┘         └─┬──┘
             ▲              ▲              ▲              ▲
             │              │              │              │
             x₁             x₂             x₃             x₄
            "The"          "cat"          "sat"          "on"
             │              │              │              │
             ▼              ▼              ▼              ▼
            y₁             y₂             y₃             y₄ ── softmax ─► "mat"
```

Critical point: the SAME weight matrices are reused every timestep. That's why an RNN can process variable-length input with a fixed parameter count.

### Side-by-side: folded vs unfolded

```
       FOLDED (compact)                  UNFOLDED (across time)

                                  t=1 ─► t=2 ─► t=3 ─► t=4
       ┌───┐                      ┌──┐   ┌──┐   ┌──┐   ┌──┐
       │ h │◄─┐         ===►   ──►│  ├──►│  ├──►│  ├──►│  │──► h_T
       └─┬─┘  │                   └─▲┘   └─▲┘   └─▲┘   └─▲┘
         │    │                     │      │      │      │
       x ▼  loop                   x₁     x₂     x₃     x₄
       cell                        (timesteps shown explicitly)
```

---

## 4. RNN Forward-Pass Math

At every timestep `t`:

```
Hidden update:    h_t = tanh( W_xh · x_t  +  W_hh · h_{t-1}  +  b_h )

Output:           y_t = softmax( W_hy · h_t  +  b_y )
```

### Variable dictionary

| Symbol | Meaning | Typical shape |
|--------|---------|---------------|
| `x_t` | input at time t (e.g., word embedding) | (embed_dim,) |
| `h_t` | hidden state ("memory") at time t | (hidden_dim,) |
| `h_{t-1}` | previous hidden state | (hidden_dim,) |
| `y_t` | output prob distribution at time t | (vocab_size,) |
| `W_xh` | input → hidden weights | (hidden_dim, embed_dim) |
| `W_hh` | hidden → hidden weights (the loop!) | (hidden_dim, hidden_dim) |
| `W_hy` | hidden → output weights | (vocab_size, hidden_dim) |
| `b_h`, `b_y` | bias vectors | (hidden_dim,), (vocab_size,) |

### Computational graph at one timestep

```
   x_t ──W_xh──┐
               ▼
              [+]──► [+ b_h] ──► tanh ──► h_t ──W_hy──► [+ b_y] ──► softmax ──► y_t
               ▲
   h_{t-1} ──W_hh──┘
```

Two paths flow into each hidden state:
- the **current input** `x_t` (what's happening now)
- the **previous hidden state** `h_{t-1}` (everything seen so far)

That blend is the entire reason RNNs can model sequences.

---

## 5. Worked Example — Predicting the Next Word

**Task**: predict the next word given `"The cat sat on"` → expect `"mat"`.

### Setup (toy sizes for clarity)

- Vocabulary: `["the", "cat", "sat", "on", "mat"]` → vocab_size = 5
- Embedding dim = 3
- Hidden dim = 4

### Step A — Word embeddings (lookup)

Each word in the vocabulary has a learned 3-dim vector:

```
"The" → x₁ = [ 0.2, -0.1,  0.5]
"cat" → x₂ = [ 0.8,  0.3, -0.2]
"sat" → x₃ = [ 0.1,  0.6,  0.4]
"on"  → x₄ = [-0.3,  0.2,  0.7]
```

### Step B — Initialize hidden state

```
h₀ = [0, 0, 0, 0]      (zeros)
```

### Step C — Walk through each timestep

```
t=1, input "The":
    h₁ = tanh( W_xh · x₁ + W_hh · h₀ + b_h )
       = [0.31, -0.42,  0.18,  0.55]

t=2, input "cat":
    h₂ = tanh( W_xh · x₂ + W_hh · h₁ + b_h )
       = [0.62, -0.15,  0.40,  0.27]
       ↑ already encodes "The cat" combined

t=3, input "sat":
    h₃ = tanh( W_xh · x₃ + W_hh · h₂ + b_h )
       = [0.45,  0.10, -0.20,  0.66]

t=4, input "on":
    h₄ = tanh( W_xh · x₄ + W_hh · h₃ + b_h )
       = [0.71,  0.05,  0.32, -0.18]
       ↑ compressed memory of the entire sentence
```

### Step D — Compute next-word probabilities

```
logits = W_hy · h₄ + b_y                  (shape = (vocab_size,))

       = [ 1.2,  -0.3,  -1.5,   0.8,   2.1]
           ↑      ↑      ↑      ↑      ↑
         "the"  "cat"  "sat"  "on"   "mat"

y₄ = softmax(logits)
   = [ 0.18,  0.04,  0.01,  0.12,  0.65]
                                    ↑
                          predicted word ("mat", 65%)
```

**Predicted next word** = `argmax(y₄)` = `"mat"`.

### Whole pipeline (one image)

```
Word     Embedding          Hidden state                Output
                            (running memory)

"The" ─► [0.2,-0.1,0.5]  ─► h₁ = tanh(...)
                                  │
"cat" ─► [0.8, 0.3,-0.2] ─► h₂ = tanh(...)
                                  │
"sat" ─► [0.1, 0.6, 0.4] ─► h₃ = tanh(...)
                                  │
"on"  ─►[-0.3, 0.2, 0.7] ─► h₄ = tanh(...) ─► W_hy·h₄+b_y ─► softmax ─► "mat"
```

---

## 6. How Learning Happens — Backprop & Gradient Descent

### Step 1 — Loss

For classification, use cross-entropy between prediction `ŷ` and true label `y`:

```
L = − Σ  yᵢ · log(ŷᵢ)
      i
```

If the true word is `"mat"` (one-hot at index 4) and `ŷ₄ = 0.65`:

```
L = − log(0.65) ≈ 0.43
```

Lower probability on the correct word → larger loss.

### Step 2 — Gradient descent (every weight)

Each weight is nudged in the direction that reduces loss:

```
W_new = W_old − η · ∂L/∂W
                ↑       ↑
        learning rate  gradient
```

### Step 3 — Backprop in a feedforward net

Apply the chain rule layer by layer, **right to left**:

```
∂L/∂W⁽³⁾ ──► ∂L/∂W⁽²⁾ ──► ∂L/∂W⁽¹⁾
   (output)     (hidden 2)     (hidden 1)
```

Each gradient is computed from the layer to its right, multiplied by local derivatives.

### Step 4 — Backpropagation Through Time (BPTT) for RNN

Unfold the RNN, then run regular backprop **across both layers and timesteps**:

```
        t=1        t=2        t=3        t=4
        ┌──┐       ┌──┐       ┌──┐       ┌──┐
   ──►  │  │ ──►   │  │ ──►   │  │ ──►   │  │ ──► L
        └▲─┘       └▲─┘       └▲─┘       └▲─┘
         │          │          │          │
         x₁         x₂         x₃         x₄

   gradients flow BACKWARD through:
        ◄────       ◄────       ◄────       ◄────  (across timesteps)
   AND every gradient updates the SAME shared W_xh, W_hh, W_hy.
```

Because `h_t = tanh(W_hh · h_{t-1} + …)`, the gradient at t=1 must flow back through `W_hh` repeatedly:

```
∂L/∂h₁  ∝  W_hh · W_hh · W_hh · …   (multiplied many times)
```

If `W_hh` < 1 → **vanishing gradient** (RNN forgets long-ago words).
If `W_hh` > 1 → **exploding gradient** (training blows up).

This is why **LSTM** and **GRU** were invented (gated cell state preserves gradients), and ultimately why **Transformers** (no recurrence, gradients flow directly through attention) replaced RNNs for long sequences.

---

## 7. Cheat Sheet (one-liners)

| Concept | Formula |
|---------|---------|
| FFN layer | `a⁽ˡ⁾ = σ( W⁽ˡ⁾ · a⁽ˡ⁻¹⁾ + b⁽ˡ⁾ )` |
| RNN hidden update | `h_t = tanh( W_xh·x_t + W_hh·h_{t-1} + b_h )` |
| RNN output | `y_t = softmax( W_hy·h_t + b_y )` |
| Cross-entropy loss | `L = − Σ yᵢ log(ŷᵢ)` |
| Weight update | `W ← W − η · ∂L/∂W` |
| ReLU | `max(0, z)` |
| Sigmoid | `1 / (1 + e⁻ᶻ)` |
| Softmax | `eᶻⁱ / Σⱼ eᶻⱼ` |

---

## 8. Minimal PyTorch Implementations

### Feedforward (MLP)

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)            # apply softmax outside / use CrossEntropyLoss
```

### Vanilla RNN (manual loop, mirrors the formulas)

```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=3, hidden_dim=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.W_xh = nn.Linear(embed_dim,  hidden_dim, bias=False)
        self.W_hh = nn.Linear(hidden_dim, hidden_dim)        # bias = b_h
        self.W_hy = nn.Linear(hidden_dim, vocab_size)        # bias = b_y

    def forward(self, x):                                    # x: (seq_len,) token IDs
        h = torch.zeros(self.W_hh.out_features)
        for t in range(x.size(0)):
            x_t = self.embedding(x[t])
            h   = torch.tanh(self.W_xh(x_t) + self.W_hh(h))
        return self.W_hy(h)                                  # use CrossEntropyLoss
```

The manual loop above is exactly what `nn.RNN` does internally — written this way so you can match every line back to the formulas in section 4.

---

## 9. What to learn next

| Limitation of vanilla RNN | Solution covered in module 04-sequence |
|---------------------------|----------------------------------------|
| Forgets long-range context | LSTM (gated cell state) — see `02-lstm.md` |
| Slow & sequential training | Transformer — see `03-transformer-architecture.md` |
| No parallelism             | Self-attention — see `04-attention.md` |
| Static word vectors        | Contextual embeddings (BERT) — see `05-bert-huggingface.md` |

Each successor architecture solved a specific bottleneck of the previous one. Once you can derive the math above by hand, those upgrades become natural extensions, not new mysteries.
