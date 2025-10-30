# Repro details (Oct 29, 2025)

Python: 3.11.x
CUDA/cuDNN: <fill>
GPU driver: <fill>
OS: <fill>

Harness:
- Source: https://github.com/<you>/ablation-harness
- Pin:
  `git submodule status` shows `<abcdef1> external/ablation-harness`.

Install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e external/ablation-harness  # installs the submodule if edited