# Foundations 2 — Neuron, Perceptron, Multilayer Perceptron

## Lectures covered
- Neuron, Perceptron and MLP

---

## 1. The artificial neuron

Inspired (loosely) by biological neurons. A unit that:
1. Takes weighted inputs
2. Sums them + adds a bias
3. Applies a non-linear activation function
4. Outputs a single number

```
inputs:    x₁ ─[w₁]─┐
           x₂ ─[w₂]─┤
           x₃ ─[w₃]─┼─► sum → + bias → activation → output
           ...      │
                    │
                    +b
```

Mathematically:
$$z = \sum_i w_i x_i + b$$
$$y = \sigma(z)$$

That's it. Stack thousands → neural network. Stack with depth → deep neural network.

---

## 2. The perceptron — single-layer NN (1958)

A perceptron is the simplest network: one neuron with a step activation.

```python
def perceptron(x, w, b):
    z = (w * x).sum() + b
    return 1 if z > 0 else 0
```

It can solve **linearly separable** problems (like AND, OR) but famously **fails on XOR** (1969 Minsky/Papert critique).

---

## 3. Why XOR matters historically

XOR truth table:
| x₁ | x₂ | y |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

There's no single line that separates the 1s from 0s. A single perceptron can't solve it.

The fix: **stack multiple layers**. A multilayer perceptron solves XOR easily because the hidden layer learns intermediate representations.

This insight (combined with backpropagation in 1986) launched modern neural networks.

---

## 4. Multilayer Perceptron (MLP)

```
input layer        hidden layer 1     hidden layer 2     output layer
   x₁ ──────►  ⚪ ──────►  ⚪ ──────►  ⚪
   x₂ ──────►  ⚪ ──────►  ⚪ ──────►  ⚪ ──► y
   x₃ ──────►  ⚪ ──────►  ⚪
   ...
```

- Each layer is a **set of neurons**
- Each neuron in one layer connects to **every** neuron in the next ("fully connected" or "dense")
- Forward pass: data flows left → right
- Backward pass (during training): gradients flow right → left

### Math (compact)
For layer $\ell$ with weight matrix $W^{(\ell)}$, bias $b^{(\ell)}$, activation $\sigma$:
$$h^{(\ell)} = \sigma\!\left( W^{(\ell)} h^{(\ell-1)} + b^{(\ell)} \right)$$

Each layer is a matrix multiplication + add + activation.

### What happens with depth
- Layer 1 learns simple patterns (edges)
- Layer 2 combines them (corners, junctions)
- Layer 3 combines those (object parts)
- Top layer combines everything into the prediction

---

## 5. PyTorch — building an MLP

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)

model = MLP(in_dim=20, hidden=128, out_dim=10)
print(model)
```

Note:
- `nn.Linear(in, out)` is a fully-connected layer (= matrix multiply + bias)
- `nn.ReLU()` is the activation
- `forward()` defines the data flow
- PyTorch auto-tracks gradients via `autograd`

---

## 6. Why width vs depth matters

| | Width (more neurons per layer) | Depth (more layers) |
|---|---|---|
| Effect | More expressive at one level | Hierarchical features |
| Problem | Overfit easily without enough data | Vanishing gradients (without skip connections) |
| Modern preference | reasonable | usually deeper > wider |

A 3-hidden-layer MLP with 128 units each is a common starting point for tabular DL.

---

## 7. MLP vs other architectures

MLPs work, but they don't exploit data structure:
- For images: a CNN exploits spatial locality → fewer parameters, better performance
- For sequences: an RNN/Transformer exploits order → fewer parameters, better performance

When to use a vanilla MLP:
- **Tabular** data
- Small models on simple tasks
- The "head" of a more complex network (after a CNN backbone)
- Demos / educational examples

---

## 8. Universal Approximation Theorem (the marketing slogan)

> A neural network with at least one hidden layer and enough neurons can approximate **any continuous function** to arbitrary precision.

Caveats:
- "Enough neurons" can be astronomical
- Doesn't say it's *learnable* via gradient descent
- Doesn't say generalization will be good

But it justifies: "I can model anything with a neural net if I have enough data and a good optimizer."

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgetting activations between layers | the whole net collapses to a linear model | always interleave non-linear activations |
| Output layer with sigmoid for multi-class | wrong output distribution | use softmax for multi-class |
| Output layer with softmax for regression | nonsensical | identity (no activation) for regression |
| Tiny network on huge data | underfit | scale up width / depth |
| Huge network on tiny data | overfit | dropout, weight decay, more data |

## Self-check

- [ ] Draw a single artificial neuron and label its components.
- [ ] Why does a single perceptron fail on XOR?
- [ ] What's the role of activation functions in stacked layers?
- [ ] Write a 3-layer MLP in PyTorch in 10 lines.
- [ ] What does "fully connected" mean?
- [ ] Why are CNNs better than MLPs for images?
- [ ] What's the universal approximation theorem and why is it limited as practical advice?
- [ ] What activation goes on the output layer for: regression / binary classification / multi-class classification?
