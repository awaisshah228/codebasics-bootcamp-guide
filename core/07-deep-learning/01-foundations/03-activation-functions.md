# Foundations 3 — Activation Functions

## Lectures covered
- Activation Function · Sigmoid · ReLU · Tanh · SoftMax

---

## 1. Why activations matter

Without non-linear activations, stacking N linear layers is equivalent to a single linear layer:
$$W_3 (W_2 (W_1 x)) = (W_3 W_2 W_1) x$$

The whole network collapses to a single matrix multiplication. Activations are what make depth *useful*.

---

## 2. The classic activations

### Sigmoid
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```
        1 ┤   ───
          │  /
      0.5 ┤ /
          │/
        0 ┤───
```

- Output ∈ (0, 1)
- **Used for**: binary classification output
- **Problems**: vanishing gradients in deep nets, not zero-centered

### Tanh
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

```
        1 ┤   ───
          │  /
        0 ┤─/─
          │/
       -1 ┤───
```

- Output ∈ (−1, 1)
- Zero-centered → better than sigmoid for hidden layers
- Still has vanishing gradients in deep nets
- Used in older RNNs

### ReLU (Rectified Linear Unit)
$$\text{ReLU}(z) = \max(0, z)$$

```
          ┤      /
          │     /
          │    /
        0 ┤───/
          │
```

- Simple, fast, no exponentials
- Doesn't saturate for positive z → no vanishing gradient
- Default for hidden layers in modern DL
- Problem: "dying ReLU" — neurons stuck at 0

### Leaky ReLU
$$\text{LeakyReLU}(z) = \max(0.01 z, z)$$

Fixes dying ReLU by allowing a tiny slope on the negative side.

### ELU, GELU, SiLU/Swish
- **ELU**: exponential on negative side; smooth
- **GELU**: Gaussian Error Linear Unit; used in BERT, GPT, modern Transformers
- **SiLU / Swish** (z · σ(z)): used in EfficientNet, modern LLMs

For this bootcamp: ReLU is the default. GELU is what you'll see in HuggingFace Transformers.

### SoftMax
$$\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

- Converts a vector of logits into probabilities (sum to 1)
- **Used for**: multi-class classification output

---

## 3. Output layer activation cheat sheet

| Task | Output activation | Loss |
|---|---|---|
| Regression | none (identity) | MSE / MAE |
| Binary classification | Sigmoid | BCE (binary cross-entropy) |
| Multi-class classification (single label) | Softmax | Cross-entropy |
| Multi-label classification | Sigmoid (one per label) | BCE per label |

---

## 4. Hidden layer activation cheat sheet

| Use case | Default |
|---|---|
| Tabular MLP | ReLU |
| CNN | ReLU |
| RNN / LSTM | tanh (built-in) |
| Modern Transformers (BERT, GPT) | GELU |
| New efficient nets | SiLU / GELU |

You rarely need to deviate from ReLU unless you're chasing a paper's exact architecture.

---

## 5. Why ReLU works (intuition)

- **Sparsity**: about half of activations are 0 → sparse representation
- **No saturation for positive z** → gradients flow
- **Computational simplicity**: no exponentials, no division

But:
- "Dying ReLU": If most outputs are negative, neurons get stuck at 0 with no gradient. Lower the learning rate or use Leaky ReLU.

---

## 6. Vanishing gradient problem

When activation derivatives are small (sigmoid, tanh in saturation), gradients shrink as they propagate back through layers. By layer 10, the gradient is near 0 → no learning in early layers.

Sigmoid's derivative max is 0.25 → after 10 layers, gradient is multiplied by 0.25^10 ≈ 10^-7.

ReLU's derivative is 1 (for positive z) → no shrinkage.

This is why ReLU + good initialization (He / Kaiming) was a breakthrough for training deep nets.

---

## 7. PyTorch usage

```python
import torch.nn as nn

# as a module (in nn.Sequential)
nn.ReLU()
nn.Sigmoid()
nn.Tanh()
nn.Softmax(dim=-1)
nn.GELU()
nn.LeakyReLU(0.01)

# as functions
import torch.nn.functional as F
F.relu(x)
F.softmax(x, dim=-1)
F.gelu(x)
```

> For multi-class, use `nn.CrossEntropyLoss` — which **internally combines** softmax + cross-entropy. Don't apply softmax twice.

---

## 8. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot activation between hidden layers | net is just linear | always interleave non-linearity |
| Sigmoid hidden layers in deep nets | vanishing gradient | use ReLU / GELU |
| Manual softmax + `CrossEntropyLoss` | double softmax | let `CrossEntropyLoss` handle it |
| Sigmoid output for multi-class | doesn't sum to 1 | use softmax |
| Using ReLU + high learning rate | dying neurons | reduce LR or use Leaky ReLU |

## Self-check

- [ ] Why do networks need non-linear activations?
- [ ] Sigmoid vs Tanh — both bounded; what's the meaningful difference?
- [ ] Why does ReLU mostly fix vanishing gradients?
- [ ] What's "dying ReLU" and how do you fix it?
- [ ] When use Softmax vs Sigmoid as output activation?
- [ ] Why doesn't `nn.CrossEntropyLoss` need a softmax in the model?
- [ ] What activation goes on a regression output layer?
- [ ] Where will you encounter GELU?
