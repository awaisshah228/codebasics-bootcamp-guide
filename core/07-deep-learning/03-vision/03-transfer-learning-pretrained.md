# Vision 3 — Transfer Learning and Pre-trained Models

## Lectures covered
- Transfer Learning
- Pre-trained Models: ResNet, EfficientNet, MobileNet
- Model Training using Transfer Learning

---

## 1. Why transfer learning is non-negotiable in 2025

Training a CNN from scratch on a few thousand images gives mediocre accuracy. Taking a model **pre-trained on ImageNet** (1.4M images, 1000 classes) and fine-tuning it on your few thousand → near state-of-the-art.

The pre-trained model already learned:
- Edges
- Textures
- Object parts
- High-level patterns

You're just *adapting* the last layer(s) to your specific classes.

> **For 99% of vision projects, you should not be training from scratch.**

---

## 2. Two flavors of transfer learning

### Feature extraction
- **Freeze** the pre-trained layers (no gradient updates)
- Replace + train only the final classifier head
- Fast, works with small data

### Fine-tuning
- Replace the head AND **unfreeze** some / all of the backbone
- Use a small learning rate so you don't destroy the pre-trained weights
- More accurate, slower, needs more data

A common middle ground: **two-phase training** — first feature-extract, then unfreeze and fine-tune at a small LR.

---

## 3. Loading a pre-trained model in PyTorch

```python
import torch
import torch.nn as nn
from torchvision import models

# ResNet-50, pre-trained on ImageNet
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# replace the final layer for your number of classes
n_classes = 4
model.fc = nn.Linear(model.fc.in_features, n_classes)
```

For different architectures:
```python
models.efficientnet_b0(weights=...)              # EfficientNet
models.mobilenet_v3_large(weights=...)            # MobileNet
models.vit_b_16(weights=...)                      # Vision Transformer
models.convnext_tiny(weights=...)                 # ConvNeXt
```

---

## 4. Freezing layers

```python
# freeze all backbone params
for p in model.parameters():
    p.requires_grad = False

# unfreeze just the new head
n_classes = 4
model.fc = nn.Linear(model.fc.in_features, n_classes)        # new layer auto requires_grad=True

# only optimize the head
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
```

For fine-tuning later:
```python
# unfreeze everything
for p in model.parameters():
    p.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)        # tiny LR
```

---

## 5. Two-phase fine-tuning recipe

```python
# Phase 1: head only
for p in model.parameters(): p.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, n_classes)
optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-3)
train_for(epochs=5)

# Phase 2: full fine-tune
for p in model.parameters(): p.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train_for(epochs=15)
```

Often gives the best of both worlds. Used in the car-damage-detection project.

---

## 6. Layer-wise learning rates (advanced)

Lower LR for early layers (general features), higher for later layers (task-specific):

```python
optimizer = torch.optim.Adam([
    {"params": model.layer1.parameters(), "lr": 1e-6},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 1e-4},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(),     "lr": 1e-3},
])
```

---

## 7. Image preprocessing — must match the pre-training

ImageNet stats:
```python
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

These are the per-channel means and stds of ImageNet. Pre-trained models expect this normalization.

Most pre-trained models expect 224×224 input (or 299×299 for some).

```python
from torchvision import transforms
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

For the new `weights` API, the transform comes built-in:
```python
weights = models.ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()
```

---

## 8. Common pre-trained models compared

| Model | Params | ImageNet top-1 | When pick |
|---|---|---|---|
| **ResNet-50** | 25M | 80.9% | Solid baseline; widely used |
| **ResNet-152** | 60M | 82.4% | Deeper if you have data |
| **EfficientNet-B0** | 5M | 77.7% | Mobile-friendly; small |
| **EfficientNet-B7** | 66M | 84.3% | High-accuracy CPU/GPU |
| **MobileNetV3** | 5M | 74.0% | Mobile / edge deployment |
| **ViT-B/16** | 86M | 81.1% | Modern; benefits from more data |
| **ConvNeXt-Tiny** | 28M | 82.1% | Great accuracy/efficiency tradeoff |

For a portfolio project: **EfficientNet-B0** or **ResNet-50** is a great default.

---

## 9. timm — the modern alternative

`timm` (PyTorch Image Models) has hundreds of pre-trained models, often state-of-the-art:

```bash
pip install timm
```

```python
import timm
model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=4)
```

For competitions / research, `timm` is the standard.

---

## 10. End-to-end transfer learning recipe

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# Data
train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_ds = datasets.ImageFolder("data/train", transform=train_tfm)
val_ds   = datasets.ImageFolder("data/val", transform=val_tfm)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=64, num_workers=4)

# Model
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))

# Phase 1: feature extraction
for p in model.features.parameters(): p.requires_grad = False
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
train(model, train_loader, val_loader, epochs=5)

# Phase 2: fine-tune
for p in model.parameters(): p.requires_grad = True
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
train(model, train_loader, val_loader, epochs=15)
```

This is exactly the recipe for the car-damage-detection project.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Wrong normalization stats | poor accuracy | use ImageNet stats for transfer learning |
| Forgot to replace final layer | trains 1000-class output | replace `model.fc` (or equivalent) |
| LR too high during fine-tune | destroys pre-trained weights | use 1e-4 or smaller |
| Unfroze everything from epoch 0 | catastrophic forgetting | two-phase: freeze, then unfreeze |
| Used 224×224 model on tiny images | can't extract features | upscale or pick a model trained at smaller resolution |

## Self-check

- [ ] Why is transfer learning the default in 2025?
- [ ] Two flavors of transfer learning — when use each?
- [ ] What's the typical input size + normalization for an ImageNet model?
- [ ] Walk through the two-phase fine-tune recipe.
- [ ] What's `timm` and why is it useful?
- [ ] Pick a pre-trained model for a small mobile-deployed app.
- [ ] Pick one for a high-accuracy server-side classifier.
- [ ] Why use a tiny LR during fine-tune?
