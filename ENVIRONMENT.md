# ENVIRONMENT

This document describes the **canonical environment** for running
`noise-sched` + `external/ablation-harness` so that:

- Results (especially FID) are **comparable across machines**.
- CPU and CUDA runs behave the same up to small numerical noise.
- New environments (Colab, new laptop, cluster) can be bootstrapped in a repeatable way.


The main steps:

1. **Pin code:** same repo commit + same submodule commit.
2. **Pin key libraries:** especially `torch` / `torchvision`.
3. **Ensure a single source of truth for `ablation-harness`.**

---

## 1. Canonical stack

### 1.1. Languages and tools

- Python: **3.11.x**
- Git: any reasonably recent version (2.40+ recommended)


### 1.2. Core Python packages

Pinned versions (in `env/requirements-base.txt`):


- `numpy==1.26.4`
- `omegaconf==2.3.0`
- `scikit-learn==1.4.2`
- `matplotlib==3.8.4`
- `tensorboard==2.17.0`
- `wandb==0.17.4`

Dev/testing:

- `pytest==8.2.1`
- `pytest-cov==5.0.0`


### 1.3. PyTorch (CPU vs CUDA builds)

We pin **version numbers**, then choose CPU or CUDA wheels depending on the machine.


Canonical versions (used everywhere):

- `torch==2.3.1`
- `torchvision==0.18.1`
- `torchaudio==2.3.1`


Install command examples:

- **CPU-only:**

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
--index-url https://download.pytorch.org/whl/cpu
```

- **CUDA-only:**

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
--index-url https://download.pytorch.org/whl/cu121
```

## 2. Repository layout & installation

### 2.1. Clone with submodules
Always clone noise-sched with submodules:


```bash
git clone --recurse-submodules https://github.com/l-mid/noise-sched.git
cd noise-sched

# Pin to a specific commit (example: E3 / FID update branch)
git checkout 1dc60460d6358fbbde5afb9fa36c89a11610cf34

# Ensure submodules match this commit
git submodule update --init --recursive

This guarantees that external/ablation-harness is at the exact
revision expected by the main repo.
```

### 2.2. Local environment (Windows / Linux / macOS, CPU)

From repo root:
Set up .venv (on windows):

```powershell
# 0) (Optional) delete the old env 
Remove-Item .venv -Recurse -Force   # or delete via Explorer

# 1) New venv explicitly with 3.11
py -3.11 -m venv .venv

# 2) Activate it
.\.venv\Scripts\Activate.ps1

# 3) Sanity check
python -VV     # should say Python 3.11.x

# 4) Upgrade pip/wheel
python -m pip install --upgrade pip wheel

# 5) Base dependencies
pip install -r env/requirements-base.txt

# 6) Install PyTorch CPU, pinned to the “Colab” version
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 `
--index-url https://download.pytorch.org/whl/cpu

# 7) Install ablation-harness from the submodule (editable, dev extras)
pip install -e external/ablation-harness[dev]

# 8) Install the main project (editable)
pip install -e .
```

There should be no other ablation-harness install (e.g. from PyPI).

All imports of ablation_harness should come from `external/ablation-harness/src`.


### 2.3. Colab environment (CUDA)
Notebook snippet (from within Colab):

```python
# 1) Mount Drive & clone
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive

!git clone --recurse-submodules https://github.com/l-mid/noise-sched.git
%cd noise-sched

# Pin to desired commit
!git checkout <latest commit sha>
!git submodule update --init --recursive
```

```python
# 2) Install dependencies
!pip install --upgrade pip

# Base deps
!pip install -r env/requirements-base.txt

# Torch CUDA build (pinned versions)
!pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Harness + project (editable)
!pip install -e external/ablation-harness[dev]
!pip install -e .
```

```python
# 3) (Optional) W&B setup
import wandb, os
from google.colab import userdata
os.environ["WANDB_API_KEY"] = userdata.get("WANDB_API_KEY")
wandb.login()
```




