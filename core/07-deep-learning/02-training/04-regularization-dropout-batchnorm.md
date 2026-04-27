# Training 4 — Regularization: Dropout, BatchNorm, Weight Decay

## Lectures covered
- Regularization
- Dropout Regularization
- Batch Normalization

---

## 1. Why regularize neural networks

Neural nets are massively over-parameterized — millions of weights vs thousands of training samples. Without regularization they memorize training data and fail on test data.

The toolkit:
1. **Weight decay** (L2 penalty)
2. **Dropout** (random neuron deactivation)
3. **Batch / Layer normalization** (sneaky regularizer + speedup)
4. **Data augmentation** (image-specific; covered in vision subfolder)
5. **Early stopping** (covered in optimizer subfolder)

---

## 2. Weight decay (L2 regularization for NNs)

Adds penalty for large weights to the loss:
$$L_{\text{total}} = L_{\text{task}} + \lambda \sum w^2$$

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

Typical values: 1e-4 to 1e-2.

> **AdamW** decouples weight decay from gradient updates correctly. **Adam's** `weight_decay` parameter implements it incorrectly (mixes it with the moment estimates). Always prefer AdamW.

---

## 3. Dropout

During training, randomly **set some neurons to 0** with probability $p$.

```
without dropout:                with dropout p=0.5:
○─→ ○─→ ○─→ output              ○─→ X ─→ ○─→ output
○─→ ○─→ ○                       ○─→ ○─→ X
○─→ ○─→ ○                       X ─→ ○─→ ○
```

### Why it regularizes
- Forces redundancy — no neuron can rely on a specific other neuron
- Equivalent to training an ensemble of subnetworks
- At inference: turn off dropout, scale activations to expected magnitude (PyTorch handles this)

### PyTorch
```python
import torch.nn as nn

self.fc = nn.Sequential(
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Dropout(p=0.3),                # 30% chance of zeroing each activation
    nn.Linear(128, 10),
)
```

`model.train()` enables dropout; `model.eval()` disables it.

### Typical p values
- 0.1–0.3 for hidden layers in vision/MLP
- 0.5 for very wide layers (original paper recommendation)
- Don't dropout the input usually
- Don't dropout the output

---

## 4. Batch Normalization

For each mini-batch, normalize each feature to mean 0, std 1, then scale + shift with learnable parameters:

$$\hat{x} = \frac{x - \mu_{\text{batch}}}{\sigma_{\text{batch}} + \epsilon}$$
$$y = \gamma \hat{x} + \beta$$

### Why it works
- Stabilizes training: each layer sees a consistent distribution
- Allows much higher learning rates
- Acts as a mild regularizer (introduces noise via batch statistics)
- Reduces sensitivity to weight initialization

### PyTorch
```python
nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1),
    nn.BatchNorm2d(16),                 # for 2D conv outputs
    nn.ReLU(),
    nn.Conv2d(16, 32, 3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(),
    ...
)
```

For MLPs: `nn.BatchNorm1d(num_features)`.

### At inference
Uses the running averages computed during training. `model.eval()` switches to this mode automatically — **don't forget it**.

### Typical placement
After the linear/conv layer, before the activation:
`Linear → BN → ReLU → Linear → BN → ReLU → ...`

Some recent work argues `Linear → ReLU → BN` is slightly better. Either works.

---

## 5. Layer Normalization (the Transformer's choice)

BatchNorm depends on the batch. **LayerNorm** normalizes across features within each sample — independent of batch.

```python
nn.LayerNorm(d_model)
```

Advantages:
- Works even with batch size 1 (BN doesn't)
- More stable for variable-length sequences (RNN, Transformer)
- Default in Transformers, modern LLMs

Use:
- LayerNorm for sequence models (Transformers, RNNs)
- BatchNorm for CNNs / MLPs

There are also `GroupNorm`, `InstanceNorm` for special cases.

---

## 6. The combined modern recipe

```python
class CNNBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

    def forward(self, x):
        return self.block(x)
```

Combined: weight decay (in optimizer) + BN + dropout + early stopping = robust training.

---

## 7. When to reach for which

| Symptom | Try |
|---|---|
| Training loss decreases, val loss increases | dropout ↑ or weight decay ↑ |
| Training is slow / unstable | BatchNorm |
| Big batch → small batch causes problems | switch BN → LayerNorm or GroupNorm |
| Want to train very deep without skip connections | BatchNorm + ReLU + careful init |
| Sequence model | LayerNorm always |
| Already using dropout, still overfitting | more data > more regularization |

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot `model.eval()` at inference | dropout still active, wrong predictions | always toggle |
| BN with batch size 1 | divides by 0 | use LayerNorm or GroupNorm |
| Dropout before BN | order matters; default is BN before dropout | follow the standard recipe |
| `weight_decay` on Adam (not AdamW) | wrong decay | use AdamW |
| Dropout p=0.5 on a tiny network | over-regularize | start at 0.1–0.3 |

## Self-check

- [ ] Why does dropout help generalization?
- [ ] What changes between `model.train()` and `model.eval()`?
- [ ] State BatchNorm's formula.
- [ ] When use LayerNorm over BatchNorm?
- [ ] Why is AdamW preferred over Adam + weight_decay?
- [ ] What's the recommended dropout p range?
- [ ] Where in a `Linear → ReLU` block does BN go?
- [ ] If you're at batch size 1, which normalization do you use?
