"""

Takes the three schedules so far and plots them in an overlay (log 10).
Σβ(t) normalized cumulative plot, log10 SNR(t) overlay.

Can also record jsonl.


Usage:
    python tools/plot_and_log_schedules.py \
    --K 1000 \
    --out-prefix docs/assets/E5/e5_plots \
    --json-path docs/assets/E5/e5_plots/e5_sigma_beta_snr.json \
    --csv-path docs/assets/E5/e5_plots/e5_sigma_beta_snr.csv
"""

import argparse
import json
import csv

import torch
import matplotlib.pyplot as plt

from ablation_harness.tasks.diffusion.schedule import get_beta_schedule, precompute_q 


def compute_schedules(K: int, device: str = "cpu"):
    kinds = ["linear", "cosine", "cosine_match_linear"]
    labels = {
        "linear": "Linear",
        "cosine": "Cosine",
        "cosine_match_linear": "Cosine (Σβ matched)",
    }

    betas_dict = {}
    q_dict = {}
    sum_beta = {}

    for kind in kinds:
        betas = get_beta_schedule(kind, K=K, device=device)
        betas_dict[kind] = betas
        q_dict[kind] = precompute_q(betas)
        sum_beta[kind] = float(betas.sum().item())

    return kinds, labels, betas_dict, q_dict, sum_beta


def plot_sigma_beta_and_snr(
    kinds,
    labels,
    betas_dict,
    q_dict,
    K: int,
    out_prefix: str = "e5_schedules",
):
    t = torch.arange(K, dtype=torch.float32)
    t_norm = t / (K - 1)

    # ----- Σβ cumulative comparison -----
    plt.figure(figsize=(6, 4))
    for kind in kinds:
        betas = betas_dict[kind]
        total = betas.sum()
        cumsum = betas.cumsum(0) / (total + 1e-12)
        plt.plot(t_norm.numpy(), cumsum.numpy(), label=labels[kind])

    plt.xlabel("Normalized timestep t/T")
    plt.ylabel("Cumulative Σβ(t) / Σβ(T)")
    plt.title("Cumulative noise mass Σβ(t)")
    plt.legend()
    plt.tight_layout()
    sigma_path = f"{out_prefix}_sigma_beta.png"
    plt.savefig(sigma_path, dpi=200)
    print(f"[plot] saved {sigma_path}")

    # ----- SNR(t) overlay (log10) -----
    plt.figure(figsize=(6, 4))
    for kind in kinds:
        alpha_bar = q_dict[kind]["alpha_bar"]
        snr = alpha_bar / (1.0 - alpha_bar + 1e-12)
        snr_log = torch.log10(snr + 1e-12)
        plt.plot(t_norm.numpy(), snr_log.numpy(), label=labels[kind])

    plt.xlabel("Normalized timestep t/T")
    plt.ylabel("log10 SNR(t)")
    plt.title("SNR(t) for each β schedule")
    plt.legend()
    plt.tight_layout()
    snr_plot_path = f"{out_prefix}_snr_log10.png"
    plt.savefig(snr_plot_path, dpi=200)
    print(f"[plot] saved {snr_plot_path}")


def log_sigma_beta_and_snr(
    kinds,
    betas_dict,
    q_dict,
    sum_beta,
    K: int,
    json_path: str = "e5_sigma_beta_snr.json",
    csv_path: str = "e5_sigma_beta_snr.csv",
):
    # SNR arrays as Python lists for JSON / CSV
    snr_dict = {}
    for kind in kinds:
        alpha_bar = q_dict[kind]["alpha_bar"]
        snr = (alpha_bar / (1.0 - alpha_bar + 1e-12)).cpu().numpy()
        snr_dict[kind] = snr.tolist()

    timesteps = list(range(K))
    t_norm = [i / (K - 1) if K > 1 else 0.0 for i in timesteps]

    # ----- JSON blob -----
    json_obj = {
        "K": K,
        # explicitly match the prereg wording
        "sum_beta_linear": sum_beta["linear"],
        "sum_beta_cosine": sum_beta["cosine"],
        "sum_beta_cosine_match_linear": sum_beta["cosine_match_linear"],
        "snr": {
            "t": timesteps,
            "t_norm": t_norm,
            "snr_linear": snr_dict["linear"],
            "snr_cosine": snr_dict["cosine"],
            "snr_cosine_match_linear": snr_dict["cosine_match_linear"],
        },
    }

    with open(json_path, "w") as f:
        json.dump(json_obj, f, indent=2)
    print(f"[log] wrote JSON to {json_path}")

    # ----- CSV table -----
    # columns: t, t_norm, snr_linear, snr_cosine, snr_cosine_match_linear
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "t",
                "t_norm",
                "snr_linear",
                "snr_cosine",
                "snr_cosine_match_linear",
            ]
        )
        for i in range(K):
            writer.writerow(
                [
                    timesteps[i],
                    t_norm[i],
                    snr_dict["linear"][i],
                    snr_dict["cosine"][i],
                    snr_dict["cosine_match_linear"][i],
                ]
            )
    print(f"[log] wrote CSV to {csv_path}")

    # quick sanity print
    print(
        "[sanity] Σβ:",
        f"linear={sum_beta['linear']:.6f}, "
        f"cosine={sum_beta['cosine']:.6f}, "
        f"cosine_match_linear={sum_beta['cosine_match_linear']:.6f}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--K", type=int, default=1000, help="Number of diffusion steps")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for schedule tensors (cpu is fine here)",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="e5_schedules",
        help="Prefix for plot filenames",
    )
    parser.add_argument(
        "--json-path",
        type=str,
        default="e5_sigma_beta_snr.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="e5_sigma_beta_snr.csv",
        help="Output CSV file path",
    )
    args = parser.parse_args()

    kinds, labels, betas_dict, q_dict, sum_beta = compute_schedules(
        args.K, device=args.device
    )

    plot_sigma_beta_and_snr(
        kinds,
        labels,
        betas_dict,
        q_dict,
        K=args.K,
        out_prefix=args.out_prefix,
    )

    log_sigma_beta_and_snr(
        kinds,
        betas_dict,
        q_dict,
        sum_beta,
        K=args.K,
        json_path=args.json_path,
        csv_path=args.csv_path,
    )


if __name__ == "__main__":
    main()