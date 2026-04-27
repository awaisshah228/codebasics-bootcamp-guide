# Vision 2 — Data Augmentation

## Lectures covered
- Data Augmentation

---

## 1. Why augment

You have 5,000 cat/dog images. The model sees the same images 50 times across 50 epochs → memorizes them. Augmentation creates *artificial diversity* by transforming each image differently each time it's seen.

Effects:
- **Reduces overfitting** (acts as regularization)
- **Improves generalization** (model becomes invariant to nuisance variations)
- **Effectively grows the dataset** without collecting new data

---

## 2. The standard `torchvision.transforms` pipeline

```python
from torchvision import transforms

train_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**Train transforms include random augmentations.** Validation transforms are deterministic — every run on the same image gives the same result.

---

## 3. The augmentation menu

### Geometric
| Transform | What |
|---|---|
| `RandomCrop` | crop a random patch |
| `RandomResizedCrop` | random crop + resize (aspect ratio jitter too) |
| `RandomHorizontalFlip` | mirror — common, useful |
| `RandomVerticalFlip` | rare; only when up/down is symmetric (e.g., satellite imagery) |
| `RandomRotation` | small rotations (5–20°) |
| `RandomAffine` | shear, translate, rotate, scale combined |
| `RandomPerspective` | 3D-look distortion |

### Color
| Transform | What |
|---|---|
| `ColorJitter` | brightness, contrast, saturation, hue |
| `Grayscale(p=0.1)` | random grayscale conversion |
| `RandomAdjustSharpness` | sharpen/blur |

### Mask / occlusion
| Transform | What |
|---|---|
| `RandomErasing` | replace a patch with noise — simulates occlusion |
| `Cutout` | similar; replaces with a fixed color |

### Stronger / data-mixing (huge accuracy gains in modern papers)
| Method | What |
|---|---|
| **Mixup** | linear combination of two images + their labels |
| **CutMix** | paste a patch of one image onto another, mix labels by area |
| **AugMix** | combines multiple augmentations |
| **AutoAugment / RandAugment** | learned/parameterized augmentation policies |

---

## 4. Mixup / CutMix in PyTorch

```python
import torch
import numpy as np

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam

# in training step:
mixed_x, y_a, y_b, lam = mixup(x, y)
logits = model(mixed_x)
loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
```

For modern image classification, mixup + cutmix together can add 1–3pp accuracy on top of standard augmentation.

---

## 5. Domain-specific augmentation

| Domain | Useful transforms |
|---|---|
| Natural photos | flip, crop, color jitter — standard recipe |
| Medical imaging | careful — might lose diagnostic features; modest rotation/zoom |
| Satellite | both flips, 90° rotations |
| Faces | NO horizontal flip if asymmetric features matter (e.g., scars) |
| Text in images (OCR) | NO flip (mirrored text isn't text) |
| Time-series (1D) | scaling, time-warping, jitter |
| Audio | time-stretch, pitch-shift, SpecAugment |

> **Augmentation is a domain decision.** Don't blindly apply the standard recipe.

---

## 6. Modern alternative — albumentations

Faster + more transforms than torchvision:

```bash
pip install albumentations
```

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

tfm = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(p=0.3),
    A.OneOf([A.GaussianBlur(), A.GaussNoise()], p=0.2),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

augmented = tfm(image=numpy_image)["image"]
```

Especially nice for object detection / segmentation where bounding boxes / masks must transform alongside the image.

---

## 7. Visualizing your augmentations

ALWAYS look at a few augmented samples before training. Surprisingly often the augmentation is too aggressive and destroys the label.

```python
import matplotlib.pyplot as plt

x, y = next(iter(train_loader))
fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i, ax in enumerate(axes):
    ax.imshow(x[i].permute(1, 2, 0).numpy().clip(0, 1))
    ax.set_title(class_names[y[i]])
    ax.axis("off")
```

If a "cat" image looks unrecognizable due to crop + jitter — back off the strength.

---

## 8. Test-time augmentation (TTA)

For the last 0.5–1pp of accuracy: apply augmentation at test time, average predictions.

```python
@torch.no_grad()
def predict_tta(model, image, n=8):
    preds = []
    for _ in range(n):
        x = train_tfm(image).unsqueeze(0).to(device)
        preds.append(F.softmax(model(x), dim=-1))
    return torch.stack(preds).mean(dim=0)
```

Good for competitions and final predictions; not worth it for real-time inference.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Augmenting validation data | unreliable val metric | val_tfm should be deterministic |
| Too aggressive augmentation | destroys label info | visualize before training |
| Wrong normalization stats | poor accuracy | use ImageNet stats for transfer learning, dataset stats for from-scratch |
| Forgot `Normalize` | different from training distribution | always normalize |
| Flipping when label depends on orientation | wrong labels | think about the domain |

## Self-check

- [ ] Why does augmentation help generalization?
- [ ] Default recipe for natural images?
- [ ] When is `RandomVerticalFlip` appropriate? When NOT?
- [ ] What does `Normalize` actually do?
- [ ] What's mixup, in one sentence?
- [ ] Why use albumentations over torchvision.transforms?
- [ ] What's TTA and when is it worth using?
- [ ] Walk through deciding the augmentation set for a medical X-ray classifier.
