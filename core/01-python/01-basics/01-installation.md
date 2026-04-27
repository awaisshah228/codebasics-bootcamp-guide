# Section 3 — Python Installation & Environment

## Lectures covered
- Peter Pandey's Journey and Need to Learn Python
- Python Installation — Windows
- Python Installation — Linux
- Python Installation — Mac
- How to Download Code and Get Help

---

## 1. Why Python (5-second version)

Three reasons it dominates data work:
1. **Readable syntax** — closer to pseudocode than any other production language
2. **Ecosystem** — pandas, NumPy, scikit-learn, PyTorch, FastAPI, Streamlit, LangChain. All Python, all interoperable.
3. **Community** — every error message has a Stack Overflow answer

It's not the fastest language, but for anything that's "slow because of Python" you can drop down to NumPy/Cython/Rust at the bottleneck only.

---

## 2. Installation by OS

### Windows
```powershell
# Option A: official installer
# Download from https://www.python.org/downloads/
# Tick "Add Python to PATH" during install.

python --version
pip --version
```

If `python` opens the Microsoft Store, you forgot the PATH checkbox. Re-run installer with the checkbox.

### macOS
```bash
# Option A: Homebrew (recommended)
brew install python@3.12

# Option B: pyenv (best for managing multiple versions)
brew install pyenv
pyenv install 3.12.0
pyenv global 3.12.0

python3 --version
pip3 --version
```

> macOS has a system Python — never modify it. Always use a brew/pyenv install.

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
python3 --version
```

For RHEL/Fedora: use `dnf install python3 python3-pip`.

---

## 3. Virtual environments — non-negotiable

Never `pip install` into system Python. Always:

```bash
# create
python3 -m venv .venv

# activate
source .venv/bin/activate     # Mac/Linux
.venv\Scripts\activate        # Windows PowerShell

# verify
which python   # should point inside .venv/

# install packages
pip install pandas numpy matplotlib seaborn jupyterlab
```

Modern alternative: **uv** (fast, written in Rust):
```bash
pip install uv
uv venv
uv pip install pandas numpy
```

For data-science specific: **conda** / **miniconda**.

---

## 4. IDE choice

Two recommended for this bootcamp:

| IDE | When to use |
|---|---|
| **PyCharm Community** (free) | When you want a heavyweight IDE with refactoring, debugging UI, run configs |
| **VS Code** | When you want lightweight + Jupyter-in-editor + works for everything else (JS, Markdown, etc.) |

Codebasics demos PyCharm in the early lectures (PyCharm install is its own lecture in Section 5).

### VS Code minimal setup
- Install: VS Code from https://code.visualstudio.com
- Extensions: **Python** (Microsoft), **Jupyter**, **Pylance**
- Settings: enable "Format on Save" with Black or Ruff

---

## 5. Jupyter Notebook / JupyterLab

Most data-science work in the bootcamp is in notebooks.

```bash
pip install jupyterlab notebook
jupyter lab        # opens browser at localhost:8888
```

In VS Code, you can open `.ipynb` files directly and run cells without launching a separate Jupyter server.

### Notebook hygiene
- Restart-and-run-all before sharing — proves the notebook actually executes top-to-bottom
- Clear all outputs before committing to git (or use `nbstripout`)
- Use markdown cells liberally — your future self / interviewer reads them

---

## 6. Downloading code & getting help

### Codebasics' code
Each lecture's code is in their public GitHub: https://github.com/codebasics

```bash
git clone https://github.com/codebasics/py.git
```

### Help loop (the order I should try)
1. **Read the error message** — Python errors are unusually clear; the *last line* names the problem
2. **Re-watch the relevant 3 minutes** of the lecture
3. **Read the docs** — https://docs.python.org/3/
4. **Search Discord #python channel** — your question was probably asked
5. **Post in Discord** with: code, error, what you tried

---

## 7. Folder convention I'll use for this bootcamp

```
~/Documents/data-scient-bootcamp/
└── practice/
    └── python/
        ├── section-04-variables/
        │   └── notebook.ipynb
        ├── section-05-lists-loops/
        ├── ...
```

Mirroring the bootcamp's section structure makes it easy to find old work later.

---

## Self-check

- [ ] `python --version` prints 3.10 or higher
- [ ] I know what a venv is and can create / activate one
- [ ] I have JupyterLab or VS Code + Jupyter extension working
- [ ] I've cloned the Codebasics public repo and run one of their notebooks
- [ ] I know where to ask for help (Discord, docs, error message)
