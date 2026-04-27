# Deployment 1 — Streamlit App

## Lectures covered
- Streamlit App

---

## Why Streamlit for DL demos

A Streamlit app is the **fastest way** to put a model behind a UI. No HTML, no JS, no React. Just Python.

For bootcamp portfolio projects, Streamlit + a hosted demo on Streamlit Cloud is the **single highest-ROI deployment effort**. Recruiters click the link, play with your model, and remember you.

---

## 1. Install
```bash
pip install streamlit
streamlit hello
```

## 2. Hello-world app
```python
# app.py
import streamlit as st

st.title("Image Classifier")
st.write("Upload an image, get a prediction.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "png"])
if uploaded:
    st.image(uploaded, caption="Your image")
    st.write("Prediction: ...")
```

Run:
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## 3. Loading a PyTorch model — `@st.cache_resource`

You don't want to reload your 100MB model on every interaction. Cache it:

```python
import torch
from torchvision import models, transforms

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(num_classes=4)
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()
```

`@st.cache_resource` runs once per session. `@st.cache_data` is for cheap data caching.

---

## 4. Image classification app — full

```python
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

CLASSES = ["No Damage", "Scratch", "Dent", "Severe"]

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(num_classes=len(CLASSES))
    model.load_state_dict(torch.load("model.pt", map_location="cpu"))
    model.eval()
    return model

tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

st.title("🚗 Car Damage Detector")
st.write("Upload a photo of a car; the model predicts damage class and confidence.")

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    model = load_model()
    x = tfm(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=-1)[0]

    pred_class = CLASSES[probs.argmax().item()]
    st.success(f"Prediction: **{pred_class}**")

    st.subheader("Confidence per class")
    for cls, p in zip(CLASSES, probs):
        st.progress(p.item(), text=f"{cls}: {p:.2%}")
```

That's a complete app for the car-damage-detection project. ~40 lines.

---

## 5. Layout primitives

```python
st.title("...")
st.header("...")
st.subheader("...")
st.write(...)              # auto-formats for any type
st.markdown("...")
st.code("...", language="python")

# inputs
st.text_input("Name")
st.text_area("Description")
st.number_input("Age", min_value=0, max_value=120)
st.slider("Threshold", 0.0, 1.0, 0.5)
st.selectbox("Choose", ["A", "B", "C"])
st.multiselect("Pick many", ["X", "Y", "Z"])
st.checkbox("Agree?")
st.radio("Pick one", ["yes", "no"])
st.date_input("Date")
st.file_uploader("Upload")

# output
st.success("..."); st.warning("..."); st.error("..."); st.info("...")
st.metric("Accuracy", "92%", "+3%")
st.progress(0.7)
st.image(...); st.video(...); st.audio(...)
st.dataframe(df); st.table(df)            # interactive vs static
st.line_chart(df); st.bar_chart(df)       # quick charts

# layout
col1, col2 = st.columns(2)
col1.metric("Train acc", "98%")
col2.metric("Val acc", "94%")

with st.expander("Show details"):
    st.write("...")

with st.sidebar:
    st.write("settings")
```

---

## 6. Multi-page apps

```
my_app/
├── streamlit_app.py        # main page
└── pages/
    ├── 1_model_info.py
    └── 2_about.py
```

Streamlit auto-discovers `pages/` and adds them to a sidebar nav.

---

## 7. State management with `st.session_state`

```python
if "counter" not in st.session_state:
    st.session_state.counter = 0

if st.button("Click"):
    st.session_state.counter += 1
st.write(st.session_state.counter)
```

---

## 8. Hosting — Streamlit Cloud (free)

1. Push your repo to GitHub (must include `requirements.txt`)
2. Go to https://streamlit.io/cloud
3. Connect repo + branch + file path
4. Click Deploy → live URL in ~2 minutes

For larger models, secrets (API keys), or higher traffic: see Streamlit Cloud's paid tiers, or self-host on Render / Railway.

---

## 9. Performance tips

- `@st.cache_resource` for models, DB connections (heavy, persistent objects)
- `@st.cache_data` for cheap pure functions
- For huge models (LLMs): hosted inference (HuggingFace Inference, OpenAI/Anthropic API) instead of loading locally
- For real-time / many concurrent users: use FastAPI backend + Streamlit calling it (next file)

---

## 10. Production-y patterns

| Need | How |
|---|---|
| Auth | `st.secrets` for keys; for user auth use `streamlit-authenticator` or external Auth0 |
| Logging | regular Python logging |
| Background jobs | not Streamlit's strength — use FastAPI + Celery |
| File uploads at scale | upload to S3 / blob storage, not on disk |
| Custom domain | Streamlit Cloud supports it on paid plans; Vercel + iframe is a workaround |

For real production, **Streamlit is best for internal tools / demos, not customer-facing scaled web apps.** Pair with FastAPI when needed.

---

## 11. Common pitfalls

| Mistake | Effect | Fix |
|---|---|---|
| Model reloads each interaction | slow | `@st.cache_resource` |
| State lost on rerun | unexpected behavior | use `st.session_state` |
| Pushing model to GitHub | huge repo | use Git LFS, or host model on HuggingFace Hub / S3 |
| Hardcoded API keys | security leak | use `st.secrets` and `.streamlit/secrets.toml` |
| `st.image()` huge images | slow page | resize before display |

## Self-check

- [ ] Build a "hello" Streamlit app in 5 lines.
- [ ] Why use `@st.cache_resource` for models?
- [ ] How do you deploy to Streamlit Cloud?
- [ ] What's `st.session_state` and when use it?
- [ ] How do you store an API key safely?
- [ ] Build a multi-page app — show me the folder structure.
- [ ] When prefer FastAPI over Streamlit for serving a model?
- [ ] Walk through a Streamlit demo for an image classifier.
