# Repro details (Oct 29, 2025)

Python: 3.11.x
CUDA/cuDNN: Google Colab: T4 GPU
GPU driver: Windows 12th-gen i7-1255U
Local: CPU-only (Windows 12th-gen i7-1255U, ~24 GB RAM, Intel Iris Xe iGPU)
Torch: CPU build (pip torch/vision/audio from pytorch.org cpu index)
Colab: GPU runtime (T4/L4/A100), CUDA torch preinstalled or installed as per PyTorch index-url cuXX

Harness:
- Source: https://github.com/L-mid/ablation-harness
- Pin:
  `git submodule status` shows `<abcdef1> external/ablation-harness`.

Install:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -e external/ablation-harness  # installs the submodule if edited




"""
For CUDA on Colab (later):

!nvidia-smi
!pip -q install -U pip
!git clone --recurse-submodules https://github.com/L-mid/noise-sched
%cd noise-sched

# Colab images usually come with CUDA torch; if not:
# !pip -q install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

!pip -q install -e external/ablation-harness
!pip -q install -e .

import torch; print("CUDA available:", torch.cuda.is_available())
!python scripts/run_baseline.py --config configs/cifar10/baseline.yaml --seed 1337  # device auto=CUDA
"""
