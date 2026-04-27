# Sequence 2 — LSTM (and GRU)

## Lectures covered
- LSTM

---

## 1. Why LSTM exists

Plain RNNs forget anything more than ~20 steps back. **LSTM (Long Short-Term Memory)** introduces a separate "memory cell" with explicit gating — what to forget, what to remember, what to output.

Result: LSTM can hold information across hundreds of steps, dramatically better at language and long sequences than vanilla RNN.

---

## 2. LSTM cell — the gates

Three sigmoid gates control the flow:

```
                 ┌─────────────────┐
              ──►│   forget gate   │── × ── add ──►   new cell state c_t
                 └─────────────────┘                       │
                 ┌─────────────────┐                       │
              ──►│   input gate    │── × ── tanh ─────────┘
                 └─────────────────┘
                 ┌─────────────────┐
              ──►│   output gate   │── × ── tanh(c_t) ───►  hidden state h_t
                 └─────────────────┘
```

### Equations
- **Forget gate**: $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ — what to drop from the cell
- **Input gate**: $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ — how much new info to add
- **Candidate**: $\tilde{c}_t = \tanh(W_c [h_{t-1}, x_t] + b_c)$ — the new info itself
- **New cell**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$
- **Output gate**: $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$ — how much of the cell to expose
- **New hidden**: $h_t = o_t \odot \tanh(c_t)$

### Two states flow through time
- $c_t$: cell state (long-term memory)
- $h_t$: hidden state (short-term memory, what's exposed to next layer / output)

---

## 3. Why this fixes vanishing gradients

The cell-state update is **additive** (gated). Gradients of $c_t$ wrt $c_{t-1}$ are dominated by $f_t$ — a value between 0 and 1, learned by the model.

Compare to vanilla RNN where $h_t$ involves a multiplication by $W_{hh}$ tanh-derivative-squashed every step → exponential decay.

LSTM's additive gating lets gradients flow back through time without the same shrinkage.

---

## 4. PyTorch LSTM

```python
import torch.nn as nn

lstm = nn.LSTM(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    bidirectional=True,
    dropout=0.2,
    batch_first=True,
)

x = torch.randn(4, 50, 10)
output, (h_n, c_n) = lstm(x)
# output: (4, 50, 40)   — 40 because bidirectional doubles hidden_size
# h_n:    (4, 4, 20)    — final hidden state per (layer * direction, batch, hidden)
# c_n:    (4, 4, 20)    — final cell state
```

The interface returns *both* hidden and cell states because some downstream tasks need both.

---

## 5. GRU — the simpler cousin

**Gated Recurrent Unit** combines the forget and input gates into one and merges cell + hidden:

- **Reset gate**: $r_t$ — how much past to forget
- **Update gate**: $z_t$ — interpolate between old and new hidden

```python
gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
```

Fewer parameters → faster training, slightly lower capacity.
Often roughly equal performance to LSTM in practice.
Modern preference often goes to LSTM for slight edge, but GRU is fine.

---

## 6. Building a sequence classifier

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, n_classes=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head  = nn.Linear(hidden_dim * 2, n_classes)         # ×2 for bidirectional

    def forward(self, x):
        # x: (batch, seq_len) of token ids
        emb = self.embed(x)                                        # (B, T, embed)
        output, (h_n, c_n) = self.lstm(emb)                        # output: (B, T, 2*hidden)

        # take the last time step's output (or pool)
        last = output[:, -1, :]                                    # (B, 2*hidden)
        return self.head(last)
```

For variable-length sequences, use **packed sequences** to skip padding tokens:
```python
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

packed = pack_padded_sequence(emb, lengths=seq_lens, batch_first=True, enforce_sorted=False)
output, (h_n, c_n) = self.lstm(packed)
output, _ = pad_packed_sequence(output, batch_first=True)
```

---

## 7. Encoder-Decoder LSTM (for sequence-to-sequence tasks)

For translation / summarization (before Transformers ate this lunch):

```
input  → [Encoder LSTM] → final_hidden_state → [Decoder LSTM] → output
```

The encoder summarizes the input into a single vector. The decoder generates the output sequence conditioned on it.

This was the dominant approach pre-2017. Now Transformers do this far better.

---

## 8. LSTM use cases that still make sense in 2025

- Time-series forecasting on small datasets
- Real-time / streaming with low latency
- Edge devices (LSTMs are smaller than Transformers)
- Speech / audio (some hybrid models use LSTMs)
- Educational baselines

For NLP from scratch: jump straight to Transformer / BERT (next files).

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to detach states across batches | gradients leak across batches | `h = h.detach()` for stateful inference |
| Wrong handling of variable-length seqs | model "learns" padding tokens | use packed sequences |
| Comparing last hidden to mean-pool | last is one timestep; pool covers all | mean/max pool often better |
| Initializing forget-gate bias at 0 | LSTMs forget aggressively early | initialize forget bias to 1 (some papers) |
| Single-layer LSTM on hard tasks | underfit | stack 2-3 layers |

## Self-check

- [ ] Why does plain RNN fail on long sequences?
- [ ] Walk through the three LSTM gates and what each controls.
- [ ] Difference between cell state and hidden state?
- [ ] Why does LSTM's additive update fix vanishing gradients?
- [ ] When prefer GRU over LSTM?
- [ ] What's a packed sequence and when use it?
- [ ] Build a sentiment classifier with bidirectional LSTM.
- [ ] What's the encoder-decoder pattern and what replaced it?
