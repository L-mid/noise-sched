from pathlib import Path
import csv
import matplotlib.pyplot as plt
from .utils import extract_nfe_fid, safe_get

def add_subparser(sp):
    ap = sp.add_parser("plot", help="Plot NFE↔FID from run directories")
    ap.add_argument("runs", nargs="+", help="Run directories (each contains results.jsonl)")
    ap.add_argument("--out-plot", default="nfe_fid.png")
    ap.add_argument("--out-csv", default="nfe_fid.csv")
    return ap

def run(args):
    rows = []  # (nfe, fid, label)
    for r in args.runs:
        jsonl = Path(r) / "results.jsonl"
        nfe, fid, cfg = extract_nfe_fid(jsonl)
        if nfe is None or fid is None:
            print(f"[warn] skipping (missing nfe/fid): {r}")
            continue
        label = safe_get(cfg, ["logging", "wandb", "run_name"]) or Path(r).name
        rows.append((nfe, fid, label))

    if not rows:
        print("[plot] no valid rows")
        return

    rows.sort(key=lambda x: x[0])
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["nfe", "fid", "label"]); w.writerows(rows)
    print(f"[plot] wrote {args.out_csv}")

    nfes = [r[0] for r in rows]
    fids = [r[1] for r in rows]

    plt.figure()
    plt.plot(nfes, fids, marker="o")
    for nfe, fid, lab in rows:
        plt.annotate(lab, (nfe, fid), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("NFE (final sampler steps)")
    plt.ylabel("FID (lower is better)")
    plt.title("NFE ↔ FID")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=160)
    print(f"[plot] wrote {args.out_plot}")