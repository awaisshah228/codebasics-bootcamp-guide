# Module 7 — Deep Learning (Beginner → Advanced)

> **Status**: 🔒 Locked
> **Project**: Car Damage Detection using CNN
> **Framework**: PyTorch (per the brochure)

## Why this module exists

Classical ML (Module 6) handles tabular data brilliantly. **Deep learning is essential when**:
- Inputs are unstructured (images, audio, text)
- Patterns are highly non-linear and require representation learning
- You have lots of data
- Pre-trained models can be transfer-learned for your task

The bootcamp covers the **full DL stack**: neural network basics → CNN (vision) → RNN/LSTM (sequence) → Transformers + BERT.

## Folder layout

```
07-deep-learning/
├── README.md
├── 01-foundations/
│   ├── README.md
│   ├── 01-intro-applications.md
│   ├── 02-neuron-perceptron-mlp.md
│   ├── 03-activation-functions.md
│   └── 04-pytorch-tensors-autograd.md
├── 02-training/
│   ├── README.md
│   ├── 01-backprop-gradient-descent.md
│   ├── 02-mnist-digits.md
│   ├── 03-optimizers-momentum-adam.md
│   ├── 04-regularization-dropout-batchnorm.md
│   └── 05-hyperparameter-optuna.md
├── 03-vision/
│   ├── README.md
│   ├── 01-cnn.md
│   ├── 02-data-augmentation.md
│   └── 03-transfer-learning-pretrained.md
├── 04-sequence/
│   ├── README.md
│   ├── 01-rnn.md
│   ├── 02-lstm.md
│   ├── 03-transformer-architecture.md
│   ├── 04-attention.md
│   └── 05-bert-huggingface.md
├── 05-deployment/
│   ├── README.md
│   ├── 01-streamlit.md
│   └── 02-fastapi.md
└── 06-projects/
    ├── README.md
    └── 01-car-damage-detection.md
```

## Curriculum (verbatim from brochure)

### Foundations
- Introduction to Deep Learning
- Neural Networks · Deep Learning vs Statistical ML
- Neural Network Architectures · Applications of Deep Learning
- PyTorch vs TensorFlow · GPU, TPU
- Neuron, Perceptron and MLP
- Activation Functions: Sigmoid, ReLU, Tanh, SoftMax
- PyTorch Installation · PyTorch Tensor Basics
- Autograd in PyTorch

### Training
- Training through Backpropagation
- Gradient Descent (theoretical foundation + PyTorch implementation)
- Batch GD vs Mini Batch GD vs SGD
- Handwritten Digits Classification
- Model Optimization
- Gradient Descent with Momentum · Adam Optimizer
- Regularization · Dropout · Batch Normalization
- Hyperparameter Tuning · Optuna

### Vision
- CNN · CIFAR10 Image Classification using CNN
- Data Augmentation
- Transfer Learning · Pre-trained Models (ResNet, EfficientNet, MobileNet)

### Sequence + Transformers
- RNN · Vanishing Gradient Problem · LSTM
- Transformer Architecture · Word Embeddings · Attention Mechanism
- Hugging Face: BERT Basics · Model Training with CNN · Transfer Learning
- Streamlit App · FastAPI Server

### Project
- Car Damage Detection using CNN

## Module-level goal

After this module:
- Build, train, and debug PyTorch models for any tabular / vision / sequence task
- Use transfer learning from torchvision pre-trained models
- Understand attention + Transformer architecture deeply enough to read papers
- Fine-tune a HuggingFace BERT for a downstream classification task
- Deploy a model with Streamlit + FastAPI

## Module-level self-check

- [ ] Walk through one step of backpropagation by hand on a 2-layer net
- [ ] Why does ReLU mostly fix vanishing gradients?
- [ ] Difference between batch norm and layer norm — when does each shine?
- [ ] What does dropout = 0.3 do during training vs inference?
- [ ] Why are CNNs translation-equivariant?
- [ ] Adam vs SGD — when use which?
- [ ] What's transfer learning and why does it work so well?
- [ ] Walk through self-attention for one token
- [ ] Why is `√d` in the attention formula?
- [ ] BERT vs GPT — encoder-only vs decoder-only — when each?
