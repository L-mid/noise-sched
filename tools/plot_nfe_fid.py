import argparse, json, sys, csv
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

# plotting & images
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
 
"""

Useage


python tools/plot_nfe_fid.py runs/2025-11-01_E3a-linear-nfe10 \
                             runs/2025-11-01_E3b-linear-nfe20 \
                             runs/2025-11-01_E3c-linear-nfe50 \
  --out-plot nfe_fid_linear.png --out-csv nfe_fid_linear.csv

  
python tools/plot_nfe_fid.py <runA> <runB> <runC> \
  --make-montage --grid-name grid.png
# => writes: grids_montage.png

python tools/plot_nfe_fid.py <runA> <runB> <runC> \
  --progression-dirs out/frames_nfe10 out/frames_nfe20 out/frames_nfe50 \
  --progression-max-frames 16
# => writes: frames_nfe10_progression.png, etc.



Current:
python tools/plot_nfe_fid.py docs/assets/E3/noise-sched-e3a/runs \
                             docs/assets/E3/noise-sched-e3b/runs \
                             docs/assets/E3/noise-sched-e3c/runs \
  --out-plot docs/assets/E3/E3_plots/nfe_fid_linear.png --out-csv docs/assets/E3/E3_plots/nfe_fid_linear.csv \
  --make-montage --grid-name grid.png
 

"""


def _safe_get(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    """For getting through nested cfg."""
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return None
    return cur



def extract_nfe_fid(jsonl_path: Path) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, Any]]]:
    """
    Parse a results.jsonl and return (final_nfe, last_val_fid, cfg_dict).
    Tries a few common key layouts for val/fid.
    """
    cfg = None
    last_fid = None
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue

                if cfg is None and isinstance(obj.get("cfg"), dict):
                    cfg = obj["cfg"]

                # Attempt a few keys to find val/fid
                candidates = [
                    _safe_get(obj, ["out", "val/fid"]),
                    # extra methods if we want
                ]
                for c in candidates:
                    if isinstance(c, (int, float)):
                        last_fid = float(c)

        nfe = None
        if cfg is not None:
            nfe = _safe_get(cfg, ["eval", "final", "nfe"])
            if nfe is not None:
                nfe = int(nfe)

        return nfe, last_fid, cfg
    except FileNotFoundError:
        return None, None, None
    

def find_grid_image(run_dir: Path, grid_name: str = "grid.png") -> Optional[Path]:
    """Common cases: either in run root or a subdir; search shallow→deep (MODIFY THIS IT'S BAD)"""
    p = run_dir / grid_name
    if p.exists():
        return p
    # fallback: recursive search (could be a bit slower)
    for q in run_dir.rglob(grid_name):
        return q
    return None


def make_grids_montage(items: List[Tuple[str, int, float, Path]], out_path: Path, width: int = 384):
    """
    items: list of (label, nfe, fid, grid_path)
    Creates a single-row montage with text under each image.
    """
    if not items:
        print("[montage] No items; skipping.")
        return

    # Load and resize images to the same width
    images = []
    for label, nfe, fid, p in items:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        new_h = int(h * (width / max(1, w)))
        img = img.resize((width, new_h))
        images.append((label, nfe, fid, img))

    # Text area height
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    line_h = 16
    text_h = line_h * 2 + 8

    total_w = sum(img.size[0] for _, _, _, img in images)
    max_h = max(img.size[1] for _, _, _, img in images)
    canvas = Image.new("RGB", (total_w, max_h + text_h), (255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, nfe, fid, img in images:
        canvas.paste(img, (x, 0))
        caption = f"{label}\nNFE={nfe}  FID={fid:.3f}" if fid is not None else f"{label}\nNFE={nfe}  FID=—"
        draw.text((x + 6, max_h + 4), caption, fill=(0, 0, 0), font=font, spacing=2)
        x += img.size[0]

    canvas.save(out_path)
    print(f"[montage] Wrote {out_path}")



def sample_evenly(seq: List[Path], k: int) -> List[Path]:
    if k >= len(seq):
        return seq
    idxs = [round(i * (len(seq) - 1) / (k - 1)) for i in range(k)]
    return [seq[i] for i in idxs]


def make_progression_strip(frames_dir: Path, out_path: Path, max_frames: int = 16, frame_width: int = 128):
    """
    Given a directory containing per-step frames (e.g., x_t_000.png ...),
    make a horizontal strip. Picks ~max_frames evenly.
    """
    if not frames_dir.exists():
        print(f"[progression] {frames_dir} not found; skipping.")
        return
    frames = sorted([p for p in frames_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
    if not frames:
        print(f"[progression] No images in {frames_dir}; skipping.")
        return
    frames = sample_evenly(frames, max_frames)

    imgs = [Image.open(p).convert("RGB") for p in frames]
    resized = []
    for img in imgs:
        w, h = img.size
        new_h = int(h * (frame_width / max(1, w)))
        resized.append(img.resize((frame_width, new_h)))
    h_max = max(im.size[1] for im in resized)
    strip = Image.new("RGB", (frame_width * len(resized), h_max), (255, 255, 255))
    x = 0
    for im in resized:
        y = (h_max - im.size[1]) // 2
        strip.paste(im, (x, y))
        x += im.size[0]
    strip.save(out_path)
    print(f"[progression] Wrote {out_path}")



def main():
    ap = argparse.ArgumentParser(description="Plot NFE↔FID from multiple runs and make optional montages.")
    ap.add_argument("runs", nargs="+", help="Run directories (each containing results.jsonl)")
    ap.add_argument("--jsonl", action="store_true", help="Treat provided paths as results.jsonl files instead of run dirs")
    ap.add_argument("--out-plot", default="nfe_fid.png", help="Output plot path")
    ap.add_argument("--out-csv", default="nfe_fid.csv", help="Output CSV path")
    ap.add_argument("--make-montage", action="store_true", help="Also build a montage from each run's grid.png")
    ap.add_argument("--grid-name", default="grid.png", help="Filename to look for inside each run (default: grid.png)")
    ap.add_argument("--progression-dirs", nargs="*", default=[], help="Zero or more directories with per-step frames")
    ap.add_argument("--progression-max-frames", type=int, default=16, help="Max frames per strip")
    args = ap.parse_args()

    rows = []  # (nfe, fid, label, run_dir)
    montage_items = []  # (label, nfe, fid, grid_path)

    for path_str in args.runs:
        p = Path(path_str)
        jsonl = p if args.jsonl else (p / "results.jsonl")
        nfe, fid, cfg = extract_nfe_fid(jsonl)
        if nfe is None and fid is None:
            print(f"[warn] Could not parse NFE/FID for: {p}")
            continue

        # Label: prefer wandb run_name; else dir name
        label = None
        if cfg:
            label = _safe_get(cfg, ["logging", "wandb", "run_name"])
        if not label:
            label = p.stem

        rows.append((nfe, fid, label, p))

        if args.make_montage:
            grid_path = find_grid_image(p if not args.jsonl else p.parent, args.grid_name)
            if grid_path:
                montage_items.append((label, nfe if nfe is not None else -1, fid if fid is not None else float("nan"), grid_path))
            else:
                print(f"[montage] grid not found in {p}")

    # Sort by NFE
    rows = [r for r in rows if r[0] is not None and r[1] is not None]
    rows.sort(key=lambda t: t[0])

    # Save CSV
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nfe", "fid", "label", "run_path"])
        for nfe, fid, label, rp in rows:
            w.writerow([nfe, fid, label, str(rp)])
    print(f"[plot] Wrote {args.out_csv}")

    # Plot NFE↔FID
    nfes = [r[0] for r in rows]
    fids = [r[1] for r in rows]
    labels = [r[2] for r in rows]

    plt.figure()
    plt.plot(nfes, fids, marker="o")
    for x, y, lab in zip(nfes, fids, labels):
        plt.annotate(f"{lab}", (x, y), textcoords="offset points", xytext=(6, 6))
    plt.xlabel("NFE (final sampler steps)")
    plt.ylabel("FID (lower is better)")
    plt.title("NFE ↔ FID (per run)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(args.out_plot, dpi=160)
    print(f"[plot] Wrote {args.out_plot}")

    # Montage
    if args.make_montage and montage_items:
        make_grids_montage(montage_items, Path("grids_montage.png"))

    # Progression strips (optional)
    for d in args.progression_dirs:
        make_progression_strip(Path(d), Path(f"{Path(d).name}_progression.png"), max_frames=args.progression_max_frames)


if __name__ == "__main__":
    sys.exit(main())