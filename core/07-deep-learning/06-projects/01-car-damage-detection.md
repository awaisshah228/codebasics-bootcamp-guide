# Project — Car Damage Detection (CNN)

## Domain
Insurance + automotive. After an accident, customers upload photos of their car. The system **automatically classifies** the damage so claims can be triaged faster.

## ML formulation
- **Type**: image classification (multi-class)
- **Classes**: typically 4 — `no_damage`, `scratch`, `dent`, `severe`
- **Metric**: macro-F1, plus per-class precision/recall (severe damage matters more than missing a tiny scratch)
- **Architecture**: pre-trained CNN (ResNet / EfficientNet) + transfer learning

---

## Why this is a great DL portfolio project

1. **Real-world domain** (automotive / insurance — both hire ML)
2. **Visual demo** — recruiters love a Streamlit app where they upload a car photo
3. **Full stack** — CNN + transfer learning + augmentation + Streamlit + FastAPI
4. **Production-relevant** — insurers actually use systems like this

---

## Walkthrough

### 1. Dataset setup
Find or construct a labeled dataset. Options:
- Existing Kaggle datasets ("car damage detection")
- Codebasics' provided dataset
- Hand-labeled (if you collect images)

```
data/
├── train/
│   ├── no_damage/      <- 800 images
│   ├── scratch/         <- 800
│   ├── dent/            <- 800
│   └── severe/          <- 800
└── val/
    ├── no_damage/      <- 200
    ├── scratch/         <- 200
    ├── dent/            <- 200
    └── severe/          <- 200
```

`torchvision.datasets.ImageFolder` reads this layout natively.

### 2. EDA + class balance check
```python
from collections import Counter
from torchvision import datasets

ds = datasets.ImageFolder("data/train")
print(Counter([y for _, y in ds.samples]))    # ensure balance
```

If imbalanced: use `WeightedRandomSampler` or `class_weight` in loss.

### 3. Augmentation pipeline
```python
from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
```

### 4. Model — EfficientNet-B0 transfer learning
```python
import torch.nn as nn
from torchvision import models

CLASSES = ["no_damage", "scratch", "dent", "severe"]

def build_model(n_classes=len(CLASSES)):
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = models.efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model
```

### 5. Two-phase training
```python
import torch.optim as optim
import torch.nn as nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = build_model().to(device)
loss_fn = nn.CrossEntropyLoss()

# Phase 1: feature extraction
for p in model.features.parameters(): p.requires_grad = False
opt = optim.Adam(model.classifier.parameters(), lr=1e-3)
train_loop(model, train_loader, val_loader, opt, loss_fn, epochs=5)

# Phase 2: full fine-tune
for p in model.parameters(): p.requires_grad = True
opt = optim.Adam(model.parameters(), lr=1e-5)
sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15)
train_loop(model, train_loader, val_loader, opt, loss_fn, epochs=15, scheduler=sched)
```

### 6. Hyperparameter tuning with Optuna (optional but nice)
- LR (log)
- Augmentation strength
- Dropout in classifier head
- Batch size

### 7. Evaluation
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_preds, all_y = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(dim=1).cpu()
        all_preds.append(preds); all_y.append(y)
    return torch.cat(all_preds), torch.cat(all_y)

preds, y_true = evaluate(model, val_loader)
print(classification_report(y_true, preds, target_names=CLASSES))
sns.heatmap(confusion_matrix(y_true, preds), annot=True, xticklabels=CLASSES, yticklabels=CLASSES)
```

### 8. Error analysis
Look at misclassified images:
- Worst confused pair → maybe scratches + dents — visually similar
- Recommend more data for that pair
- Or class-specific rules (severity score from area)

### 9. Deployment

#### FastAPI server (saves model.pt + serves /predict)
See `05-deployment/02-fastapi.md` — full template there.

#### Streamlit UI
See `05-deployment/01-streamlit.md`.

#### Hosting
- **HuggingFace Spaces** — free, fastest path
- **Render** — also free
- Both: push code + model to GitHub → connect → live URL in 5 min

### 10. Demo polish
- Demo gif / Loom recording
- README with screenshot
- 3 example images bundled (so reviewers don't need their own)
- Performance numbers in README ("macro-F1 0.91; per-class table below")

---

## Repo structure
```
car-damage-detection/
├── data/                            # not committed (huge)
├── notebooks/
│   ├── 01-data-overview.ipynb
│   ├── 02-baseline-training.ipynb
│   ├── 03-fine-tune.ipynb
│   └── 04-error-analysis.ipynb
├── src/
│   ├── train.py
│   ├── server.py                    # FastAPI
│   └── app.py                       # Streamlit
├── models/
│   └── model.pt                     # (or hosted on HuggingFace Hub)
├── examples/                        # 3 sample images for the README
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Stretch goals

- **Object detection / bounding box** instead of classification (using YOLOv8 or Detectron2)
- **Segmentation** of damage area (Mask R-CNN, U-Net)
- **Cost estimation** model that takes the damage class + car make/model → predicted repair cost
- **Mobile** deployment (TorchScript / ONNX → CoreML / TFLite)

These are great extensions for follow-up posts and demonstrate progression.

---

## Self-check

- [ ] Did I use a pre-trained model + transfer learning (not from scratch)?
- [ ] Two-phase training: feature extraction then fine-tune?
- [ ] Augmentation appropriate for the domain?
- [ ] Per-class metrics + confusion matrix in the README?
- [ ] Streamlit demo deployed publicly?
- [ ] Error analysis section included?
- [ ] README with screenshots, demo link, and per-class numbers?
- [ ] LinkedIn post with the demo link?
