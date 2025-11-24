"""
This plot will overlay the loss and fid from loss.jsonl + results.jsonl (if desired)

python tools/plot_loss_fid_overlay.py \
  runs/E6/loss.jsonl \
  runs/E6/results.jsonl \
  --out docs/assets/E6/loss_fid_overlay.png


"""



import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_loss_and_fid(loss_path: str):
    """Parse loss.jsonl for train loss and any fid-like metrics."""
    steps_loss = []
    losses = []
    steps_fid = []
    fids = []

    with open(loss_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)

            # step index
            step = rec.get("_i") or rec.get("global_step") or rec.get("step")
            out = rec.get("out", rec)

            # ---- loss ----
            for key in ("train/loss", "loss", "train_loss"):
                if key in out:
                    if step is not None:
                        steps_loss.append(step)
                        losses.append(out[key])
                    break

            # ---- intermittent FID(s) ----
            for k, v in out.items():
                if "fid" in k.lower():
                    if step is not None:
                        steps_fid.append(step)
                        fids.append(v)
                    break

    return (steps_loss, losses), (steps_fid, fids)


def load_final_fid(results_path: str | None, default_step=None):
    """Parse results.jsonl for final FID (take the last FID it sees)."""
    if results_path is None:
        return None, None

    final_step = None
    final_fid = None

    with open(results_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out = rec.get("out") or rec.get("metrics") or rec

            fid_here = None
            for k, v in out.items():
                if "fid" in k.lower():
                    fid_here = v
                    break

            if fid_here is None:
                continue

            step = rec.get("global_step") or rec.get("step") or rec.get("_i")
            final_step = step
            final_fid = fid_here

    if final_fid is None:
        return None, None

    if final_step is None:
        final_step = default_step
    return final_step, final_fid


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("loss_jsonl", help="path to loss.jsonl")
    parser.add_argument(
        "results_jsonl",
        nargs="?",
        default=None,
        help="optional results.jsonl with final FID",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output PNG path, e.g. docs/assets/E6/loss_fid_overlay.png",
    )
    args = parser.parse_args(argv)

    (loss_steps, losses), (fid_steps, fids) = load_loss_and_fid(args.loss_jsonl)

    # Use max loss step as fallback x-position for final FID if needed
    default_step = max(loss_steps) if loss_steps else None
    final_step, final_fid = load_final_fid(args.results_jsonl, default_step=default_step)

    fig, ax1 = plt.subplots(figsize=(7, 4))

    # Loss curve
    if loss_steps:
        ax1.plot(loss_steps, losses, label="train loss", linewidth=1.5)
    ax1.set_xlabel("step")
    ax1.set_ylabel("loss")

    # FID on secondary axis
    ax2 = ax1.twinx()
    if fid_steps:
        ax2.plot(fid_steps, fids, "o--", label="FID (intermittent)")
    if final_fid is not None:
        ax2.scatter(
            [final_step],
            [final_fid],
            marker="*",
            s=120,
            label=f"FID final={final_fid:.2f}",
        )
    ax2.set_ylabel("FID")

    # Merge legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)


if __name__ == "__main__":
    main()