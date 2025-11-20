"""
Quick FID vs NFE sweep from an existing diffusion checkpoint.

This uses:
  - your CIFAR10 U-Net (unet_cifar32)
  - your diffusion schedule helpers (get_beta_schedule, precompute_q)
  - your existing FID helper (_fid_for_generated)
and DOES NOT retrain anything.

It loads the checkpoint's *model* weights (non-EMA) and runs FID at
one or more NFE values.

Example:

python -m tools.eval_fid_from_ckpt \
  --ckpt docs/assets/E3/noise-sched-e3a/ckpts/last.pt \
  --fid-stats external/ablation-harness/stats/cifar10_inception_train.npz \
  --nfe 10 20 50 \
  --n-samples 32 \
  --batch-size 64 \
  --beta-schedule linear \
  --save-grid 


Fid stats: 
Normal (n samples = 32):  
imgs mean/std: 0.49897313117980957 0.3493768274784088
[summary] NFE=  10  FID=195.725134
[summary] NFE=  20  FID=203.987357
[summary] NFE=  50  FID=203.081029




"""


from __future__ import annotations

import argparse
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torchvision.utils import save_image

# --- imports from your codebase ---
from ablation_harness.eval.generative import (
    _fid_for_generated,
    _sample,
)
from ablation_harness.tasks.diffusion.schedule import (
    get_beta_schedule,
    precompute_q,
)
from ablation_harness.tasks.diffusion.models.unet_cifar32 import (
    build_unet_model,
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_model_for_ckpt(device: torch.device) -> torch.nn.Module:
    """
    Rebuild the same model architecture used during training.

    We mimic train._run_diffusion, which does:

        from .tasks.diffusion.models.unet_cifar32 import build_unet_model
        model = build_unet_model(rt).to(device)

    Since build_unet_model() expects a RuntimeConfig-like object,
    we pass a minimal stub with the fields it actually uses.

    If build_unet_model requires additional fields (e.g. dropout),
    add them to rt_stub here.
    """
    rt_stub = SimpleNamespace(
        dataset="cifar10",      # if your builder asserts this
        model_name="unet_cifar32",   # in case it checks rt.model
    )
    model = build_unet_model(rt_stub).to(device)
    model.eval()
    return model


def build_q(beta_schedule: str, K: int, device: torch.device):
    """Rebuild diffusion schedule q exactly like in train._run_diffusion."""
    betas = get_beta_schedule(beta_schedule, K, device=device)
    q = precompute_q(betas)
    return q


def load_model_weights(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    """
    Load the model weights from a checkpoint created by train._run_diffusion.

    We expect the checkpoint dict to contain a 'model' key with a state_dict.

    If your checkpoint structure differs, print the keys and adjust.
    """
    print(f"[eval] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[eval] Checkpoint keys: {list(ckpt.keys())}")

    if "model" not in ckpt:
        raise KeyError(
            "Expected key 'model' in checkpoint. "
            "Print ckpt.keys() and adapt load_model_weights() if needed."
        )

    model = build_model_for_ckpt(device)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print("[eval] Warning: missing / unexpected keys when loading state_dict:")
        print("  missing   :", missing)
        print("  unexpected:", unexpected)

    model.eval()
    return model


# ---------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("eval_fid_nfe_sweep")

    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (best_val.pt or last.pt).")
    p.add_argument(
        "--fid-stats",
        required=True,
        help="Path to CIFAR10 FID stats .npz (from make_cifar10_fid_stats.py).",
    )
    p.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        required=True,
        help="One or more NFE values to evaluate, e.g. --nfe 10 20 50",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of generated samples for FID. "
             "Use something like 256 for quick debugging.",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for sample generation.",
    )
    p.add_argument(
        "--beta-schedule",
        type=str,
        default="linear",
        help="Beta schedule name (matches get_beta_schedule; e.g. linear, cosine).",
    )
    p.add_argument(
        "--K",
        type=int,
        default=1000,
        help="Number of diffusion steps used to build q (K in train.py).",
    )
    p.add_argument(
        "--sampler",
        type=str,
        default="ddpm",
        choices=("ddpm", "ddim"),
        help="Sampler name (must match what _sample() expects).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for generation (incremented per batch inside _fid_for_generated).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (e.g. 'cuda', 'cuda:0', 'cpu'). "
             "Defaults to CUDA if available, else CPU.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save grids / summary. "
             "Defaults to <ckpt_dir>/eval_nfe_sweep.",
    )
    p.add_argument(
        "--save-grid",
        action="store_true",
        help="If set, save a sample grid for each NFE.",
    )
    p.add_argument(
        "--grid-samples",
        type=int,
        default=64,
        help="Number of images for each grid (must be a perfect square).",
    )

    return p.parse_args()


def main():
    args = parse_args()

    ckpt_path = Path(args.ckpt)
    fid_stats_path = Path(args.fid_stats)

    # device
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # out dir
    out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else ckpt_path.parent / "eval_nfe_sweep"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] device       = {device}")
    print(f"[eval] ckpt         = {ckpt_path}")
    print(f"[eval] fid_stats    = {fid_stats_path}")
    print(f"[eval] out_dir      = {out_dir}")
    print(f"[eval] beta_schedule= {args.beta_schedule}, K={args.K}")

    # build q and model
    q = build_q(args.beta_schedule, args.K, device)
    model = load_model_weights(ckpt_path, device)

    # main sweep
    results = []
    for nfe in args.nfe:
        print(f"\n[eval] ===== NFE = {nfe} =====")
        fid_val = _fid_for_generated(
            model_ema=model,          # using plain model weights here
            q=q,
            device=device,
            n_samples=int(args.n_samples),
            sampler=args.sampler,
            nfe=int(nfe),
            fid_stats_path=str(fid_stats_path),
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
        print(f"[eval] NFE={nfe:4d}  FID={fid_val:.6f}")
        results.append((nfe, fid_val))

        # optional grid
        if args.save_grid:
            grid_n = int(math.sqrt(args.grid_samples))
            if grid_n * grid_n != args.grid_samples:
                raise ValueError(f"grid_samples={args.grid_samples} is not a perfect square.")

            with torch.no_grad():
                imgs = _sample(
                    model=model,
                    shape=(args.grid_samples, 3, 32, 32),  # CIFAR10
                    q=q,
                    sampler=args.sampler,
                    nfe=int(nfe),
                    seed=int(args.seed),
                    device=device,
                )
                # [-1,1] -> [0,1]
                imgs = (imgs.clamp(-1, 1) + 1.0) / 2.0

            grid_path = out_dir / f"grid_nfe{nfe}.png"
            save_image(imgs[: grid_n * grid_n], grid_path, nrow=grid_n)
            print(f"[eval] Saved grid to {grid_path}")

    # Save a tiny CSV with the sweep
    csv_path = out_dir / "nfe_fid_sweep.csv"
    np.savetxt(
        csv_path,
        np.array(results, dtype=float),
        fmt="%.6f",
        delimiter=",",
        header="nfe,fid",
        comments="",
    )
    print(f"\n[eval] Saved NFE↔FID sweep to {csv_path}")
    for nfe, fid_val in results:
        print(f"[summary] NFE={int(nfe):4d}  FID={fid_val:.6f}")


if __name__ == "__main__":
    main()