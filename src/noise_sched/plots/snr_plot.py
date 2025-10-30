import argparse
import numpy as np
import matplotlib.pyplot as plt
from noise_sched.schedules.registry import build    # find this
import importlib  # ensure cosine is imported to populate registry

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cosine_beta")
    p.add_argument("--T", type=int, default=1000)
    args = p.parse_args()

    # Ensure schedules are imported (so decorators run)
    importlib.import_module("noise_sched.schedules.cosine")

    betas = build(args.name, T=args.T)
    alphas = 1.0 - betas
    alpha_bar = np.cumprod(alphas)
    # SNR_t = alpha_bar / (1 - alpha_bar)
    snr = alpha_bar / np.clip(1.0 - alpha_bar, 1e-12, None)

    plt.figure()
    plt.plot(np.arange(args.T), snr)
    plt.title(f"SNR vs t — {args.name} (T={args.T})")
    plt.xlabel("t")
    plt.ylabel("SNR(t)")
    plt.tight_layout()
    plt.savefig("plots/snr.png", dpi=200)
    print("Saved plots/snr.png")

if __name__ == "__main__":
    main()

    