"""
Usage:
    Basic example:

        python tools/plot_grids_together.py \
            --ckpt-linear docs/assets/E5/ckpts/E1_linear.pt \
            --ckpt-cosine docs/assets/E5/ckpts/E2_cosine.pt \
            --ckpt-cosine-match docs/assets/E5/ckpts/E5_matched.pt \
            --out-dir docs/assets/E5/e5_plots

            
    Optional flags:

        --K 1000                # number of diffusion steps
        --nfe 50                # sampler NFE
        --batch-size 36         # samples per grid
        --img-size 32           # image side length
        --seed 1077             # random seed for sampling
        --device cuda           # or "cpu"
        --out-name e5_E1_E2_E5_grids.png
"""

import argparse
import os

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from ablation_harness.tasks.diffusion.schedule import get_beta_schedule, precompute_q  # adjust import path
from ablation_harness.tasks.diffusion.samplers.ddpm import DDPMSampler  # adjust import path
from ablation_harness.tasks.diffusion.models.unet_cifar32 import UNetCifar32


def load_model_from_ckpt(ckpt_path, device):
    """Load UNetCifar32 from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = UNetCifar32()
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def make_sampler(kind, K, nfe, device):
    """Instantiate DDPM sampler for a given beta schedule."""
    betas = get_beta_schedule(kind, K=K, device=device)
    q = precompute_q(betas)

    sampler = DDPMSampler(q, nfe=nfe)
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


def plot_three_grids(
    ckpt_linear,
    ckpt_cosine,
    ckpt_cosine_match,
    K=1000,
    nfe=50,
    batch_size=36,
    img_size=32,
    seed=123,
    device="cuda",
    out_path="e5_grids.png",
):
    """Load E1/E2/E5 models, sample, and save a stacked 3×1 grid figure."""
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- Load models ----
    model_lin = load_model_from_ckpt(ckpt_linear, device)
    model_cos = load_model_from_ckpt(ckpt_cosine, device)
    model_cosm = load_model_from_ckpt(ckpt_cosine_match, device)

    # ---- Samplers ----
    sampler_lin = make_sampler("linear", K=K, nfe=nfe, device=device)
    sampler_cos = make_sampler("cosine", K=K, nfe=nfe, device=device)
    sampler_cosm = make_sampler("cosine_match_linear", K=K, nfe=nfe, device=device)

    # ---- Grids (same seed for comparability) ----
    grid_lin = sample_grid(model_lin, sampler_lin, batch_size, img_size, seed, device)
    grid_cos = sample_grid(model_cos, sampler_cos, batch_size, img_size, seed, device)
    grid_cosm = sample_grid(model_cosm, sampler_cosm, batch_size, img_size, seed, device)

    # ---- Plot stacked figure ----
    fig, axes = plt.subplots(3, 1, figsize=(6, 9))

    titles = ["E1 — Linear β", "E2 — Cosine β", "E5 — Cosine (Σβ matched)"]
    grids = [grid_lin, grid_cos, grid_cosm]

    for ax, title, grid in zip(axes, titles, grids):
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
        description="Plot E1/E2/E5 sample grids together (linear vs cosine vs cosine_match_linear)."
    )
    parser.add_argument(
        "--ckpt-linear",
        type=str,
        required=True,
        help="Path to E1 (linear β) checkpoint.",
    )
    parser.add_argument(
        "--ckpt-cosine",
        type=str,
        required=True,
        help="Path to E2 (cosine β) checkpoint.",
    )
    parser.add_argument(
        "--ckpt-cosine-match",
        type=str,
        required=True,
        help="Path to E5 (cosine Σβ matched) checkpoint.",
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
        default="e5_E1_E2_E5_grids.png",
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, args.out_name)

    plot_three_grids(
        ckpt_linear=args.ckpt_linear,
        ckpt_cosine=args.ckpt_cosine,
        ckpt_cosine_match=args.ckpt_cosine_match,
        K=args.K,
        nfe=args.nfe,
        batch_size=args.batch_size,
        img_size=args.img_size,
        seed=args.seed,
        device=args.device,
        out_path=out_path,
    )
