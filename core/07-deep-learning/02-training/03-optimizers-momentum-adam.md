# Training 3 — Optimizers: Momentum, Adam, LR Schedules

## Lectures covered
- Model Optimization
- Gradient Descent with Momentum
- Adam Optimizer

---

## 1. Why "vanilla" SGD isn't always enough

Plain SGD has problems:
- Bounces around in narrow ravines (loss landscapes are usually anisotropic)
- Slow on plateaus (small gradients)
- Sensitive to learning rate

Modern optimizers fix some of these.

---

## 2. SGD with Momentum

Like a ball rolling — it accumulates velocity, smooths out noise, accelerates in consistent directions.

$$v_t = \beta v_{t-1} + \nabla L$$
$$\theta_t = \theta_{t-1} - \eta v_t$$

- $\beta$ (momentum coefficient): typically 0.9
- Effectively averages recent gradients

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### When to use SGD+momentum
- Computer vision (ResNet, YOLO are typically trained with SGD+momentum)
- When you have time to tune LR carefully — usually slightly better final accuracy than Adam

---

## 3. Adam — adaptive learning rates per parameter

Combines momentum + per-parameter learning rate scaling based on the **squared gradient** history.

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

(Hat = bias-corrected.)

In English: maintain a running mean (momentum) AND a running squared mean (per-parameter scale). Use the ratio.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### Why Adam is the default
- Less sensitive to LR choice
- Often converges faster than SGD
- Good default for almost everything

### When NOT to use Adam
- Final accuracy on vision tasks sometimes lower than SGD+momentum
- Some papers report Adam generalizes worse than SGD

### AdamW — Adam + correct weight decay
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
```

The default for modern Transformers / LLMs.

---

## 4. Comparison of common optimizers

| Optimizer | Pros | Cons | Default for |
|---|---|---|---|
| SGD | simple | slow, sensitive | rarely used alone |
| SGD+momentum | strong final accuracy | tune LR | CNN training |
| RMSprop | per-param LR, simple | older | older RNNs |
| Adam | fast, robust, low-tune | sometimes generalizes worse | default for almost everything |
| AdamW | Adam + correct weight decay | ~same speed as Adam | Transformers, LLMs |

---

## 5. Learning rate scheduling

Even good optimizers benefit from **decaying** the LR over training:
- Start large to make progress
- Decay to refine

### StepLR — drop LR every N epochs
```python
from torch.optim.lr_scheduler import StepLR
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# every 10 epochs, LR *= 0.5
```

### ExponentialLR
```python
ExponentialLR(optimizer, gamma=0.95)             # LR *= 0.95 every epoch
```

### CosineAnnealingLR — smooth decay
```python
from torch.optim.lr_scheduler import CosineAnnealingLR
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
```

The default for many modern Transformer trainings.

### OneCycleLR — go up then down
```python
from torch.optim.lr_scheduler import OneCycleLR
scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=epochs, steps_per_epoch=len(loader))
```

Highly effective for vision; faster convergence in fewer epochs.

### ReduceLROnPlateau — auto-decay on validation plateau
```python
from torch.optim.lr_scheduler import ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

# in training loop:
scheduler.step(val_loss)
```

### Scheduler call sites — which step where
- Most schedulers: `scheduler.step()` once per epoch
- OneCycle: `scheduler.step()` once per **batch**
- ReduceLROnPlateau: `scheduler.step(val_loss)` once per epoch

Read each scheduler's docs to be sure.

---

## 6. Warmup — the modern pattern

Big LRs early on can blow up. **Linear warmup** for the first few hundred steps, then your normal schedule.

```python
# manual warmup
warmup_steps = 500
for step, (x, y) in enumerate(loader):
    lr = base_lr * min(1.0, (step + 1) / warmup_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr
    ...
```

Or use `torch.optim.lr_scheduler.LambdaLR` with a warmup function.

---

## 7. Mixed-precision training (free 2x speedup on GPU)

```python
scaler = torch.cuda.amp.GradScaler()

for x, y in loader:
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        logits = model(x)
        loss = loss_fn(logits, y)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

- 2× faster, ~half memory
- Required for training large models on consumer GPUs
- Built into PyTorch

---

## 8. Picking optimizer + LR for new projects

| Domain | Default starting point |
|---|---|
| MLP on tabular | Adam, lr=1e-3 |
| CNN on images | SGD+momentum, lr=0.1 with cosine; OR AdamW lr=1e-3 |
| Transformer / LLM | AdamW, lr=2e-5 (fine-tune), 5e-4 (from scratch) |
| Reinforcement learning | Adam, lr=3e-4 |
| Anything weird | AdamW, lr=1e-3 |

If unsure: start with AdamW, lr=1e-3, plot loss, adjust.

---

## 9. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Calling `scheduler.step()` before `optimizer.step()` | warning + wrong LR | call after optimizer.step() |
| Not zeroing gradients | accumulation | `optimizer.zero_grad()` |
| Using Adam's defaults on unstable training | divergence | drop LR by 10x |
| Forgetting `weight_decay` for AdamW | over-regularization absent | tune weight_decay |
| Mixed-precision without GradScaler | NaN losses | always use `GradScaler` for FP16 |

## Self-check

- [ ] How does momentum modify plain SGD?
- [ ] What two running statistics does Adam maintain?
- [ ] When prefer SGD+momentum over Adam?
- [ ] What's the difference between Adam and AdamW?
- [ ] When use cosine annealing vs step LR?
- [ ] What's warmup and why does it help?
- [ ] How do you enable mixed precision in PyTorch?
- [ ] Show the right call order for `optimizer` and `scheduler`.
