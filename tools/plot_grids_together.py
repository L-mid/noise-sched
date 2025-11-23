"""
Usage:
    Basic example (E6-style 4-way grid):

        python tools/plot_grids_together.py \
            --ckpt-linear docs/assets/E6/ckpts/E1_linear.pt \
            --ckpt-cosine docs/assets/E6/ckpts/E2_cosine.pt \
            --out-dir docs/assets/E6/e6_plots \
            --out-name e6_linear_cosine_ddpm_ddim.png

    This will:
      - Load EMA weights from each checkpoint if available (fallback: raw model).
      - Build beta schedules:
            schedule_linear  -> for the linear model (default: "linear")
            schedule_cosine  -> for the cosine model (default: "cosine")
      - Construct 4 samplers:
            (linear, DDPM), (linear, DDIM),
            (cosine, DDPM), (cosine, DDIM)
      - Sample a shared batch from each and save a 2x2 grid figure.

    Optional flags:

        --schedule-linear linear        # beta schedule name for the "linear" model
        --schedule-cosine cosine        # beta schedule name for the "cosine" model
        --no-ema                        # disable EMA usage, use raw model weights
        --K 1000                        # number of diffusion steps
        --nfe 50                        # sampler NFE
        --batch-size 36                 # samples per grid
        --img-size 32                   # image side length
        --seed 1077                     # random seed for sampling
        --device cuda                   # or "cpu"
        --title-linear-ddpm  "..."      # override panel titles
        --title-linear-ddim  "..."
        --title-cosine-ddpm  "..."
        --title-cosine-ddim  "..."

current:
    python tools/plot_grids_together.py \
        --ckpt-linear docs/assets/E6/e1_data/last.pt \
        --ckpt-cosine docs/assets/E6/e2_data/last.pt \
        --out-dir docs/assets/E6/e6_plots \
        --out-name e6_linear_cosine_ddpm_ddim.png \
        --K 1000 \
        --nfe 50 \
        --batch-size 36 \
        --img-size 32 \
        --seed 1077 \
        --device cuda \
        --title-linear-ddpm  "E1-linear — DDPM" \
        --title-linear-ddim  "E1-linear — DDIM" \
        --title-cosine-ddpm  "E2-cosine — DDPM" \
        --title-cosine-ddim  "E2-cosine — DDIM"
        
"""

import argparse
import os

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from ablation_harness.tasks.diffusion.schedule import get_beta_schedule, precompute_q  # adjust import path
from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler  # adjust import path
from ablation_harness.tasks.diffusion.samplers.ddim import DDIMSampler  # adjust import path
from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32


def _pick_ema_state(ckpt):
    """
    Try to extract EMA weights from a checkpoint in a few common formats.

    Returns:
        state_dict or None
    """
    ema = ckpt.get("ema", None)
    if ema is None:
        return None

    # If it's already a flat state_dict
    if isinstance(ema, dict):
        # Common patterns:
        #  - ckpt["ema"] is directly the state_dict
        #  - ckpt["ema"]["state_dict"]
        #  - ckpt["ema"]["shadow_params"] / ["params"]
        if "weight" in ema or any(k.startswith("module.") for k in ema.keys()):
            return ema

        for key in ("state_dict", "shadow_params", "params"):
            if key in ema and isinstance(ema[key], dict):
                return ema[key]

    # If it's actually an nn.Module-like object
    if hasattr(ema, "state_dict"):
        return ema.state_dict()

    return None


def load_model_from_ckpt(ckpt_path, device, use_ema=True):
    """
    Load UNetCifar32 from checkpoint.

    Prefers EMA weights if `use_ema` and the EMA state_dict is compatible with the model.
    Falls back cleanly to the raw model weights otherwise.
    """
    ckpt = torch.load(ckpt_path, map_location=device)

    model = UNetCifar32().to(device)

    # Try EMA first
    if use_ema and "ema" in ckpt:
        try:
            print(f"[load_model_from_ckpt] Trying EMA weights from: {ckpt_path}")
            model.load_state_dict(ckpt["ema"])
            print(f"[load_model_from_ckpt] Using EMA weights.")
            model.eval()
            return model
        except Exception as e:
            print(
                f"[load_model_from_ckpt] EMA state_dict not compatible with UNetCifar32 "
                f"(error: {e!r}). Falling back to raw model weights."
            )

    # Fallback: raw model weights
    print(f"[load_model_from_ckpt] Using raw model weights from: {ckpt_path}")
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def make_sampler(sampler_kind, beta_schedule_kind, K, nfe, device):
    """
    Instantiate a sampler (DDPM or DDIM) for a given beta schedule.

    Args:
        sampler_kind: "ddpm" or "ddim"
        beta_schedule_kind: schedule name string, e.g. "linear", "cosine", "cosine_match_linear"
        K: number of diffusion steps
        nfe: sampler function evaluations
        device: torch.device

    Returns:
        Configured sampler with q/nfe/device attached.
    """
    sampler_kind = sampler_kind.lower()
    betas = get_beta_schedule(beta_schedule_kind, K=K, device=device)
    q = precompute_q(betas)

    if sampler_kind == "ddpm":
        sampler = DDPMSampler(q, nfe=nfe)
    elif sampler_kind == "ddim":
        sampler = DDIMSampler(q, nfe=nfe)
    else:
        raise ValueError(f"Unknown sampler_kind '{sampler_kind}'. Expected 'ddpm' or 'ddim'.")

    sampler.q = q
    sampler.nfe = nfe
    sampler.device = device
    return sampler


@torch.no_grad()
def sample_grid(model, sampler, batch_size=36, img_size=32, seed=0, device="cuda"):
    """Run sampling job and return a torchvision grid tensor."""
    shape = (batch_size, 3, img_size, img_size)
    xs = sampler.sample(model, shape, seed=seed)  # outputs in [-1, 1]
    xs = (xs + 1) / 2.0  # map to [0, 1]
    grid = make_grid(xs, nrow=int(batch_size ** 0.5), padding=2)
    return grid


def plot_four_grids(
    ckpt_linear,
    ckpt_cosine,
    schedule_linear="linear",
    schedule_cosine="cosine",
    K=1000,
    nfe=50,
    batch_size=36,
    img_size=32,
    seed=123,
    device="cuda",
    use_ema=True,
    out_path="e6_grids.png",
    title_linear_ddpm="Linear β — DDPM",
    title_linear_ddim="Linear β — DDIM",
    title_cosine_ddpm="Cosine β — DDPM",
    title_cosine_ddim="Cosine β — DDIM",
):
    """
    Build a 2×2 grid:

        [ Linear+DDPM   |  Linear+DDIM  ]
        [ Cosine+DDPM   |  Cosine+DDIM  ]
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load models ----
    model_lin = load_model_from_ckpt(ckpt_linear, device, use_ema=use_ema)
    model_cos = load_model_from_ckpt(ckpt_cosine, device, use_ema=use_ema)

    # ---- Samplers ----
    sampler_lin_ddpm = make_sampler("ddpm", schedule_linear, K=K, nfe=nfe, device=device)
    sampler_lin_ddim = make_sampler("ddim", schedule_linear, K=K, nfe=nfe, device=device)
    sampler_cos_ddpm = make_sampler("ddpm", schedule_cosine, K=K, nfe=nfe, device=device)
    sampler_cos_ddim = make_sampler("ddim", schedule_cosine, K=K, nfe=nfe, device=device)

    # ---- Grids (same seed across all four for comparability) ----
    grid_lin_ddpm = sample_grid(model_lin, sampler_lin_ddpm, batch_size, img_size, seed, device)
    grid_lin_ddim = sample_grid(model_lin, sampler_lin_ddim, batch_size, img_size, seed, device)
    grid_cos_ddpm = sample_grid(model_cos, sampler_cos_ddpm, batch_size, img_size, seed, device)
    grid_cos_ddim = sample_grid(model_cos, sampler_cos_ddim, batch_size, img_size, seed, device)

    # ---- Plot 2×2 figure ----
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    titles = [
        title_linear_ddpm,
        title_linear_ddim,
        title_cosine_ddpm,
        title_cosine_ddim,
    ]
    grids = [
        grid_lin_ddpm,
        grid_lin_ddim,
        grid_cos_ddpm,
        grid_cos_ddim,
    ]

    for ax, title, grid in zip(axes.flatten(), titles, grids):
        img = grid.permute(1, 2, 0).cpu().numpy()
        img = img.clip(0, 1)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 4-way sample grids: (linear vs cosine) × (DDPM vs DDIM)."
    )

    parser.add_argument(
        "--ckpt-linear",
        type=str,
        required=True,
        help="Path to checkpoint for the linear-β trained model (E1).",
    )
    parser.add_argument(
        "--ckpt-cosine",
        type=str,
        required=True,
        help="Path to checkpoint for the cosine-β trained model (E2).",
    )

    parser.add_argument(
        "--schedule-linear",
        type=str,
        default="linear",
        help="Beta schedule name for the 'linear' model (must match training; default: 'linear').",
    )
    parser.add_argument(
        "--schedule-cosine",
        type=str,
        default="cosine",
        help="Beta schedule name for the 'cosine' model (must match training; default: 'cosine').",
    )

    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Directory to write the output figure into (will be created if needed).",
    )
    parser.add_argument(
        "--out-name",
        type=str,
        default="e6_linear_cosine_ddpm_ddim.png",
        help="Filename for the output figure.",
    )

    parser.add_argument(
        "--K",
        type=int,
        default=1000,
        help="Number of diffusion steps (must match training).",
    )
    parser.add_argument(
        "--nfe",
        type=int,
        default=50,
        help="Number of sampler function evaluations (NFE).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=36,
        help="Number of samples in each grid.",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=32,
        help="Image side length (assumes square).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1077,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device to use ("cuda" or "cpu").',
    )

    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use raw model weights instead of EMA weights.",
    )

    # Panel titles (CLI-labelable)
    parser.add_argument(
        "--title-linear-ddpm",
        type=str,
        default="Linear β — DDPM",
        help="Title for the (linear, DDPM) panel.",
    )
    parser.add_argument(
        "--title-linear-ddim",
        type=str,
        default="Linear β — DDIM",
        help="Title for the (linear, DDIM) panel.",
    )
    parser.add_argument(
        "--title-cosine-ddpm",
        type=str,
        default="Cosine β — DDPM",
        help="Title for the (cosine, DDPM) panel.",
    )
    parser.add_argument(
        "--title-cosine-ddim",
        type=str,
        default="Cosine β — DDIM",
        help="Title for the (cosine, DDIM) panel.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    plot_four_grids(
        ckpt_linear=args.ckpt_linear,
        ckpt_cosine=args.ckpt_cosine,
        schedule_linear=args.schedule_linear,
        schedule_cosine=args.schedule_cosine,
        K=args.K,
        nfe=args.nfe,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        device=args.device,
        use_ema=not args.no_ema,
        out_path=out_path,
        title_linear_ddpm=args.title_linear_ddpm,
        title_linear_ddim=args.title_linear_ddim,
        title_cosine_ddpm=args.title_cosine_ddpm,
        title_cosine_ddim=args.title_cosine_ddim,
    )