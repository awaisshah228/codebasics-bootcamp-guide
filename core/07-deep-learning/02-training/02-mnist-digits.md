# Training 2 — Handwritten Digits Classification (MNIST)

## Lectures covered
- Handwritten Digits Classification

---

## Why MNIST is the universal first DL project

- Simple: 28×28 grayscale digits, 10 classes
- Small: 60k train + 10k test
- Fast: trains in minutes on CPU, seconds on GPU
- Standard: every framework, every textbook uses it

It's the "Hello World" for neural networks. Master it; the patterns transfer everywhere.

---

## End-to-end MNIST in PyTorch

### 1. Data loading
```python
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.ToTensor(),                        # PIL image → tensor in [0,1]
    transforms.Normalize((0.1307,), (0.3081,)),    # MNIST stats
])

train_ds = datasets.MNIST("data/", train=True, download=True, transform=transform)
test_ds  = datasets.MNIST("data/", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256)
```

### 2. The model — first an MLP
```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                          # 28×28 → 784
            nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 64),  nn.ReLU(),
            nn.Linear(64, 10),                     # 10 classes
        )

    def forward(self, x):
        return self.net(x)
```

### 3. Training loop
```python
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model = MLP().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_epoch():
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def eval_acc(loader):
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total

for epoch in range(10):
    train_epoch()
    print(f"epoch {epoch}: train_acc={eval_acc(train_loader):.4f}, test_acc={eval_acc(test_loader):.4f}")
```

A vanilla MLP gets ~98% test accuracy on MNIST. Solid baseline.

### 4. Upgrade to a CNN (preview — full CNN treatment in the vision subfolder)
```python
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64), nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.fc(self.conv(x))
```

CNN gets ~99.2% — meaningfully better than MLP because it exploits spatial structure.

### 5. Visualizing predictions
```python
import matplotlib.pyplot as plt

x, y = next(iter(test_loader))
x, y = x.to(device), y.to(device)
preds = model(x).argmax(dim=1)

fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(x[i].cpu().squeeze(), cmap="gray")
    ax.set_title(f"pred {preds[i].item()} / true {y[i].item()}")
    ax.axis("off")
plt.tight_layout()
```

### 6. Where the model struggles
Look at confusion matrix and the actual misclassified images:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

all_preds, all_targets = [], []
model.eval()
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        preds = model(x).argmax(dim=1).cpu()
        all_preds.append(preds); all_targets.append(y)
all_preds = torch.cat(all_preds); all_targets = torch.cat(all_targets)

cm = confusion_matrix(all_targets, all_preds)
sns.heatmap(cm, annot=True, fmt="d")
```

Common confusions: 4↔9, 3↔5, 7↔1. Most match human-eye difficulty.

---

## Fashion-MNIST — the natural next step

Same shape (28×28, 10 classes) but **clothing items** — much harder than digits. Drop in:
```python
datasets.FashionMNIST("data/", train=True, download=True, transform=transform)
```

Your MLP gets ~88%, CNN ~92%. Demonstrates that you have to *think* about architecture, not copy MNIST results.

---

## Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Forgot to flatten before MLP | shape error | `nn.Flatten()` first |
| No normalization | slow convergence | `transforms.Normalize` |
| Using softmax + CrossEntropy | double softmax | drop softmax (CE includes it) |
| Tested without `model.eval()` | dropout still on | always toggle |
| Tiny `num_workers` | data loading bottleneck | `num_workers=2` or 4 |

## Self-check

- [ ] Set up the MNIST DataLoader with normalization.
- [ ] Build an MLP and a CNN; compare accuracy.
- [ ] Why does the CNN beat the MLP on images?
- [ ] How do you compute test accuracy?
- [ ] What confusions are typical on MNIST?
- [ ] Switch to Fashion-MNIST without changing model code.
- [ ] Plot 10 misclassified images.
- [ ] Write the training loop from memory in 15 lines.
