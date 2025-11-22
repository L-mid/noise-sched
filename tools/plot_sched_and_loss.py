"""
Plots snr cosine plot and loss (used for E4).

Useage:
# Loss vs steps (one or many)
python tools/plot_sched_and_loss.py --loss /mnt/data/loss.jsonl --outdir /mnt/data

# SNR(t) using your schedule.py
python tools/plot_sched_and_loss.py --schedule /mnt/data/schedule.py --outdir /mnt/data

# Both, and you can add more loss files for overlays
python tools/plot_sched_and_loss.py \
  --loss /mnt/data/loss.jsonl /path/to/another/loss.jsonl \
  --schedule /mnt/data/schedule.py \
  --outdir /mnt/data


Current:
    python tools/plot_sched_and_loss.py \
  --loss docs/assets/E4/data/loss.jsonl /path/to/another/loss.jsonl \
  --schedule external/ablation-harness/tasks/diffusion/schedule.py \
  --outdir /E4/E4_plots 

"""

import argparse, json, math, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------- JSONL helpers ----------
"""Reads jsonl."""
def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def infer_step_key(sample_row: dict):
    """Can infer a variety of step keys from loss.jsonl."""
    for k in ["step", "global_step", "train_step", "iteration", "iter", "steps", "_i"]:
        if k in sample_row and isinstance(sample_row[k], (int, float)):
            return k
    for k, v in sample_row.items():
        if isinstance(v, (int, float)) and "time" not in k.lower():
            return k
    return None


def extract_loss_df(loss_jsonl: Path) -> pd.DataFrame:
    """Finds the loss key and extracts it."""
    rows = read_jsonl(loss_jsonl)
    if not rows:
        return pd.DataFrame(columns=["step","loss","source"])
    step_key = infer_step_key(rows[0]) or "step"
    loss_keys = ["loss", "train/loss", "train_loss", "objective", "l_total"]
    data = []
    for r in rows:
        step = r.get(step_key, None)
        lval = None
        for lk in loss_keys:
            if lk in r:
                lval = r[lk]; break
        if lval is None:
            for k, v in r.items():
                if isinstance(v, (int, float)) and "loss" in k.lower():
                    lval = v; break
        if step is not None and lval is not None:
            data.append({"step": float(step), "loss": float(lval)})
    df = pd.DataFrame(data).sort_values("step").reset_index(drop=True)
    df["source"] = loss_jsonl.name
    return df


# ---------- schedule loading ----------
def load_schedule_module(schedule_py: Path):
    spec = importlib.util.spec_from_file_location("user_schedule", str(schedule_py))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def compute_snr_cosine(mod=None, num_steps: int = 1000):
    """Fallback. Plots cosine natively."""
    t = np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    alpha_bar = None
    # Prefer functions from user's schedule.py if present
    if mod is not None and hasattr(mod, "alpha_bar_cosine"):
        alpha_bar = np.array([mod.alpha_bar_cosine(tt) for tt in t], dtype=np.float64)
    elif mod is not None and hasattr(mod, "make_beta_schedule"):
        betas = mod.make_beta_schedule(schedule="cosine", n_timestep=num_steps)  # type: ignore
        alphas = 1.0 - np.array(betas, dtype=np.float64)
        alpha_bar = np.cumprod(alphas)
    else:
        # Nichol & Dhariwal cosine ᾱ(t) fallback
        s = 0.008
        f = lambda x: np.cos((x + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = np.clip(f(t) / f(0.0), 1e-12, 1.0)
    snr = alpha_bar / np.clip(1.0 - alpha_bar, 1e-12, None)
    return t, snr



# ---------- plotting ----------
def plot_loss(loss_paths, outdir: Path):
    """Simple loss plotter."""
    dfs = []
    for p in loss_paths:
        p = Path(p)
        if p.exists():
            dfs.append(extract_loss_df(p))
    if not dfs:
        print("No loss.jsonl files found; skipping loss plot.")
        return None
    df = pd.concat(dfs, ignore_index=True)
    plt.figure(figsize=(7,4.5))
    for src, sub in df.groupby("source"):
        sub = sub.sort_values("step")
        plt.plot(sub["step"].values, sub["loss"].values, label=src)  # default colors only
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Steps")
    plt.legend()
    out = outdir / "loss_vs_steps.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"Saved: {out}")
    return out

def plot_snr_cosine(schedule_py: Path, outdir: Path):
    """Simple cosine schedule plotter."""
    mod = None
    if schedule_py and Path(schedule_py).exists():
        mod = load_schedule_module(Path(schedule_py))
    t, snr = compute_snr_cosine(mod, num_steps=1000)
    plt.figure(figsize=(7,4.5))
    plt.plot(t, snr)  # default color
    plt.xlabel("Normalized time t")
    plt.ylabel("SNR(t)")
    plt.title("Cosine Schedule SNR(t)")
    out = outdir / "snr_cosine.png"
    plt.tight_layout(); plt.savefig(out, dpi=160); plt.close()
    print(f"Saved: {out}")
    return out


def main():
    """Orchestrator."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", nargs="*", default=[], help="One or more loss.jsonl files.")
    ap.add_argument("--schedule", type=str, default="", help="Path to schedule.py for cosine SNR.")
    ap.add_argument("--outdir", type=str, default=".", help="Output directory for plots.")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.loss:
        plot_loss(args.loss, outdir)
    if args.schedule:
        plot_snr_cosine(Path(args.schedule), outdir)
    else:
        # still generate using fallback if user wants it
        plot_snr_cosine(Path(args.schedule), outdir)

if __name__ == "__main__":
    main()


