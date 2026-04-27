# Foundations 4 — PyTorch: Tensors and Autograd

## Lectures covered
- PyTorch Installation
- PyTorch Tensor Basics
- Autograd in PyTorch

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
