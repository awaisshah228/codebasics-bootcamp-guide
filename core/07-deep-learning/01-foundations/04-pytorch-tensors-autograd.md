# Foundations 4 — PyTorch: Tensors and Autograd

## Lectures covered
- PyTorch Installation
- PyTorch Tensor Basics
- Autograd in PyTorch

---

## In one sentence
**PyTorch** gives you two superpowers: a NumPy-like array (the **tensor**) that runs on the GPU, and an automatic gradient engine (**autograd**) that does the calculus for you so you can train any neural net.

## Real-world analogy
Imagine a spreadsheet that (1) runs on a thousand-core supercomputer instead of one core, and (2) remembers every formula you typed so it can compute every cell's sensitivity to every input automatically. That's a tensor with autograd. You write `y = x**2 + 3`, ask "how does y change with x?", and PyTorch answers without you doing any calculus.

## The intuition (plain English)
- A **tensor** is just a multi-dimensional array — scalar (0D), vector (1D), matrix (2D), image (3D), batch of images (4D).
- Add `requires_grad=True` and PyTorch silently builds a **computation graph** as you do math on the tensor.
- Call `.backward()` and PyTorch walks the graph backwards to compute every derivative automatically.
- The **training loop** is always the same six lines: forward → loss → zero grads → backward → step → repeat.

## Mini worked example — autograd in three lines

You want the derivative of `y = x² + 3x + 1` at `x = 2`. Calculus says `dy/dx = 2x + 3 = 7`.

```python
import torch

x = torch.tensor(2.0, requires_grad=True)   # we want grad with respect to x
y = x**2 + 3*x + 1                          # forward pass; PyTorch records ops
y.backward()                                # walk graph back, compute dy/dx
print(x.grad)                               # tensor(7.)   ← matches 2(2)+3
```

That same machinery scales to a network with 100 million parameters — you never write a derivative again.

## At-a-glance — the canonical PyTorch training loop

```mermaid
flowchart LR
    A[Get a batch] --> B[Forward: model x]
    B --> C[Compute loss]
    C --> D[optimizer.zero_grad]
    D --> E[loss.backward]
    E --> F[optimizer.step]
    F --> A
```

```
for x, y in loader:                       # batch
    logits = model(x)                     # forward
    loss   = loss_fn(logits, y)           # compute loss
    optimizer.zero_grad()                 # reset old gradients
    loss.backward()                       # autograd: compute new gradients
    optimizer.step()                      # update weights
```

Memorize these six lines — every CNN, RNN, Transformer training loop in this module follows this exact shape.

## Why this matters
- Tensors + autograd are why you can prototype a new architecture in 30 lines instead of writing a calculus textbook.
- The "data ↔ device" rules (CPU vs GPU) cause 80% of beginner errors — knowing them saves hours.
- `model.train()` vs `model.eval()` and `torch.no_grad()` are tiny lines with huge effects on accuracy and memory.

---

## 1. Installation

### CPU-only (works on any laptop)
```bash
pip install torch torchvision
```

### GPU-enabled (requires CUDA drivers)
Visit https://pytorch.org → use their selector to get the right command for your CUDA version.

```bash
# example for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Verify
```python
import torch
print(torch.__version__)                         # 2.x.x
print(torch.cuda.is_available())                  # True if GPU available
print(torch.backends.mps.is_available())          # True if Apple Silicon GPU available
```

---

## 2. Tensors — the fundamental data type

A **tensor** is PyTorch's n-dimensional array (like NumPy's ndarray, but with GPU + autograd support).

```python
import torch

# from list / numpy
a = torch.tensor([1, 2, 3])                       # 1D
b = torch.tensor([[1, 2], [3, 4]])                # 2D

# special constructors
zeros = torch.zeros(3, 4)
ones  = torch.ones(2, 5)
rand  = torch.randn(3, 3)                          # standard normal
arange = torch.arange(0, 10, 2)
linspace = torch.linspace(0, 1, 5)
```

### Attributes
```python
b.shape                    # torch.Size([2, 2])
b.dtype                    # torch.int64
b.device                   # device(type='cpu')
b.ndim                     # 2
```

### Common dtypes
- `torch.float32` (default) — most layers expect this
- `torch.float64` — rarely used (memory)
- `torch.float16` / `torch.bfloat16` — mixed precision
- `torch.int64` — labels
- `torch.bool` — masks

### Move to GPU
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
b = b.to(device)
# or:  b = b.cuda()  /  b = b.mps  (Apple Silicon)
```

---

## 3. Tensor ops (mostly NumPy-compatible)

```python
a = torch.randn(3, 4)
b = torch.randn(4, 5)

a + a                         # element-wise
a * a
a @ b                         # matrix multiply
torch.matmul(a, b)            # same
a.T                           # transpose
a.sum(), a.mean(), a.std()
a.sum(dim=0), a.sum(dim=1)
a.reshape(4, 3)
a.view(4, 3)                  # like reshape (must be contiguous)
a.unsqueeze(0)                # add dim
a.squeeze()                   # remove size-1 dims
```

### Indexing (NumPy-like)
```python
a[0, 1]
a[:, 0]
a[a > 0]                       # boolean mask
```

### Concat / stack
```python
torch.cat([a, b], dim=0)
torch.stack([a, a], dim=0)
```

---

## 4. Tensor ↔ NumPy interop

```python
import numpy as np

t = torch.from_numpy(np.array([1.0, 2.0]))
arr = t.numpy()            # tensor → numpy (must be on CPU)
```

> Both share memory by default. Modifying one modifies the other. Use `.clone()` if you want a copy.

---

## 5. Autograd — the magic

PyTorch tracks operations on tensors that have `requires_grad=True`, then computes gradients automatically.

### Simple example
```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()                  # computes dy/dx
print(x.grad)                 # 2x + 3 = 7.0
```

### Multi-variable
```python
x = torch.tensor([1.0, 2.0], requires_grad=True)
y = (x ** 2).sum()
y.backward()
print(x.grad)                 # [2x_1, 2x_2] = [2, 4]
```

### How it actually works
PyTorch builds a **computation graph** as you do operations. When you call `.backward()`, it walks the graph backwards applying the chain rule.

This is the engine behind **every** training loop you'll ever write.

---

## 6. Disabling autograd (for inference)

Computing gradients costs memory + time. During evaluation:
```python
with torch.no_grad():
    y = model(x)
```

Or globally for a function:
```python
@torch.no_grad()
def predict(x):
    return model(x)
```

In modern PyTorch, also use `model.eval()` to put the model in inference mode (turns off dropout, uses batch-norm running stats).

```python
model.eval()
with torch.no_grad():
    preds = model(X_test)
```

---

## 7. The basic training loop (preview)

```python
model = MyModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()                   # reset gradients
        logits = model(x)                       # forward pass
        loss = loss_fn(logits, y)               # compute loss
        loss.backward()                         # backward pass — autograd!
        optimizer.step()                        # update parameters
```

Every training loop in PyTorch — for a CNN, RNN, Transformer, anything — has this exact shape. Memorize it.

---

## 8. DataLoader — feeding data efficiently

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

ds = MyDataset(X_train, y_train)
loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
```

DataLoader handles batching, shuffling, multi-worker loading, prefetching. Standard for any PyTorch project.

---

## 9. Saving / loading models

```python
# save state_dict (recommended)
torch.save(model.state_dict(), "model.pt")

# load
model = MyModel()
model.load_state_dict(torch.load("model.pt"))
model.eval()

# save optimizer + epoch for resuming training
torch.save({
    "model": model.state_dict(),
    "optim": optimizer.state_dict(),
    "epoch": epoch,
}, "checkpoint.pt")
```

---

## 10. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Tensor on CPU when model on GPU | `RuntimeError` | `.to(device)` everywhere |
| Forgot `optimizer.zero_grad()` | gradients accumulate | always zero before `.backward()` |
| Forgot `model.eval()` during inference | dropout / batchnorm wrong | always `model.eval() + torch.no_grad()` |
| `loss.backward()` after `model.eval()` without grad | works but wasteful | use `torch.no_grad()` |
| `.cuda()` mixed with `.to(device)` style | inconsistent | pick one style and stick |

## Self-check

- [ ] Difference between PyTorch tensor and NumPy array?
- [ ] How do you put a tensor on GPU?
- [ ] What does `requires_grad=True` enable?
- [ ] Walk through the 6 lines of a basic training loop.
- [ ] Why use `with torch.no_grad():` during inference?
- [ ] What do `model.train()` and `model.eval()` toggle?
- [ ] When to use `state_dict()` vs `torch.save(model)`?
- [ ] What's a DataLoader and why use one?

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Tensor** | PyTorch's n-dimensional array — like a NumPy array but GPU-capable and gradient-aware |
| **Scalar / Vector / Matrix / 4D tensor** | 0D / 1D / 2D / 4D arrays. Images are typically 4D: (batch, channels, height, width) |
| **dtype** | Numerical type stored in the tensor (`float32`, `int64`, `bool`, …) |
| **device** | Where the tensor lives: `"cpu"`, `"cuda"`, or `"mps"` (Apple Silicon) |
| **CUDA** | NVIDIA's GPU programming platform; required for GPU PyTorch on NVIDIA cards |
| **MPS** | Apple's Metal Performance Shaders backend for Apple-Silicon GPUs |
| **`.to(device)`** | Move a tensor or model between CPU and GPU |
| **`requires_grad`** | Flag on a tensor that tells PyTorch to track ops on it for autograd |
| **Computation graph** | Hidden DAG PyTorch builds while you do math; used during `.backward()` |
| **autograd** | PyTorch's automatic-differentiation engine — computes gradients via the chain rule |
| **`backward()`** | Walks the graph backwards and fills in `.grad` for every leaf tensor |
| **`grad`** | The accumulated gradient stored on a tensor with `requires_grad=True` |
| **Forward pass** | Running data through the model to get a prediction |
| **Backward pass** | Computing gradients of the loss with respect to every parameter |
| **Loss function** | Number measuring how wrong the prediction is (e.g., `nn.CrossEntropyLoss`) |
| **Optimizer** | Algorithm that updates weights using gradients (SGD, Adam, …) |
| **`optimizer.zero_grad()`** | Reset gradients to zero before computing new ones (they accumulate by default) |
| **`optimizer.step()`** | Apply the gradient update to every parameter |
| **`model.train()` / `model.eval()`** | Switches dropout/BatchNorm into training or inference mode |
| **`torch.no_grad()`** | Context that disables gradient tracking — saves memory during inference |
| **`nn.Module`** | Base class for any PyTorch model — gives you parameter tracking, `to()`, save/load |
| **`state_dict`** | Dictionary of all model weights — the recommended save format |
| **DataLoader** | Iterator that batches, shuffles, and parallel-loads your data |
| **Dataset** | Class with `__len__` and `__getitem__` describing how to fetch one sample |
| **Batch** | A small group of samples processed together (e.g., 32 images at once) |
| **Epoch** | One full pass through the training data |
| **Mixed precision** | Training with `float16`/`bfloat16` to save memory and speed up GPUs |

## Further reading
- Next: [../02-training/01-backprop-gradient-descent.md](../02-training/01-backprop-gradient-descent.md) — what `loss.backward()` is doing under the hood
- Architecture math: [../architectures-and-math.md](../architectures-and-math.md)
- Official PyTorch docs: https://pytorch.org/docs/stable/index.html
