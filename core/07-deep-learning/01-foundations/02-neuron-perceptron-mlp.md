# Foundations 2 — Neuron, Perceptron, Multilayer Perceptron

## Lectures covered
- Neuron, Perceptron and MLP

---

## In one sentence
A **neuron** is a tiny decision-maker that mixes its inputs, adds a bias, and squashes the result through a curve; stack many of them in layers and you get a **neural network**.

## Real-world analogy
Think of a college admissions officer. They look at GPA, SAT score, and essay quality. Each factor gets a personal *weight* — maybe essays matter most. They sum the weighted scores, add a personal "bar to clear" (bias), and decide yes/no. A single neuron does the exact same thing with numbers.

## The intuition (plain English)
- A **single neuron** can only draw a straight line between two groups — fine for simple problems, useless for anything wiggly.
- Stack many neurons in a layer → the layer can detect many patterns at once.
- Stack many *layers* → each layer combines patterns from the layer below, building up to abstract concepts.
- The same building block (weighted sum + bias + activation) makes up MLPs, CNNs, RNNs, and Transformers — only the **wiring** changes.

## Mini worked example — one neuron deciding "is this email spam?"

Three features: number of links (x₁), number of "FREE!!" words (x₂), sender reputation 0–1 (x₃).

```
Inputs:    x = [5,  3,  0.2]    (5 links, 3 spam words, low reputation)
Weights:   w = [0.4, 0.6, -1.5] (more links/spam-words → more spam;
                                 high reputation → less spam)
Bias:      b = -0.5

z = (0.4·5) + (0.6·3) + (-1.5·0.2) + (-0.5)
  = 2.0 + 1.8 - 0.3 - 0.5
  = 3.0

output = sigmoid(3.0) ≈ 0.95     → 95% chance spam
```

That single neuron is already a tiny logistic-regression spam classifier.

## At-a-glance — from neuron to MLP

```
   Single neuron (1958, perceptron):
        x ──► [Σ + b] ──► activation ──► y      ← can only do straight-line splits

   Multi-layer perceptron (MLP):
        x ──► [Layer 1] ──► [Layer 2] ──► [Output]
              many neurons    many neurons    one or more
              learn edges    learn corners   final answer

   Each line between layers = one weight to learn.
```

## Why this matters
- The neuron is the **Lego brick** of every deep network you'll build.
- Understanding why a single perceptron fails on XOR explains why **depth** is non-negotiable.
- Every PyTorch `nn.Linear` you'll ever write is just a layer of these neurons in disguise.

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

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Neuron** | One math unit: weighted-sum of inputs + bias, passed through an activation |
| **Weight (w)** | The "importance" assigned to each input; learned during training |
| **Bias (b)** | A constant added to the weighted sum; lets the neuron shift its decision boundary |
| **Activation function** | A non-linear curve (sigmoid, ReLU…) applied after the sum |
| **Linear combination** | The weighted-sum step: `w₁x₁ + w₂x₂ + … + b` |
| **Perceptron** | Earliest single-neuron classifier (1958), uses a step activation |
| **Step activation** | Outputs 0 or 1 based on whether the input is above a threshold |
| **Linearly separable** | Two groups can be split by a single straight line / flat plane |
| **XOR problem** | Famous toy task no single perceptron can solve — required multilayer nets to fix |
| **MLP (Multi-Layer Perceptron)** | A stack of fully-connected layers with non-linear activations between them |
| **Hidden layer** | Any layer between the input and output |
| **Fully connected (Dense)** | Every neuron in a layer connects to every neuron in the next |
| **Forward pass** | Data flowing left-to-right through the network to make a prediction |
| **Backward pass / backprop** | Gradients flowing right-to-left to update the weights |
| **Width** | How many neurons are in a layer |
| **Depth** | How many layers the network has |
| **Universal Approximation Theorem** | A wide-enough network with one hidden layer can approximate any continuous function |
| **`nn.Linear`** | PyTorch's fully-connected layer (matrix multiply + bias) |
| **`nn.ReLU`** | PyTorch's ReLU activation as a module |
| **`nn.Sequential`** | A container that runs its sub-modules in order |
| **`autograd`** | PyTorch's automatic-differentiation engine |
| **Vanishing gradient** | Gradients shrink so much through deep nets that early layers stop learning |
| **Skip connection** | A shortcut that lets gradients bypass layers (used in ResNet) |

## Further reading
- Next: [03-activation-functions.md](03-activation-functions.md) — the curves that make depth useful
- Visual + math reference: [../architectures-and-math.md](../architectures-and-math.md)
- Linear-classifier ancestor: [../../06-machine-learning/02-classification/01-logistic-regression.md](../../06-machine-learning/02-classification/01-logistic-regression.md)
