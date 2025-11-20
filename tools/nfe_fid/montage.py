from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from .utils import extract_nfe_fid, safe_get, find_grid_image

def add_subparser(sp):
    ap = sp.add_parser("montage", help="Build side-by-side montage from each run’s grid.png")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--grid-name", default="grid.png")
    ap.add_argument("--out", default="grids_montage.png")
    ap.add_argument("--width", type=int, default=384, help="Resize width per panel")
    return ap

def run(args):
    items = []  # (label, nfe, fid, image)
    for run in args.runs:
        runp = Path(run)
        nfe, fid, cfg = extract_nfe_fid(runp / "results.jsonl")
        label = safe_get(cfg, ["logging", "wandb", "run_name"]) or runp.name
        gp = find_grid_image(runp, args.grid_name)
        if gp is None:
            print(f"[montage] grid not found in {run}")
            continue
        im = Image.open(gp).convert("RGB")
        w, h = im.size
        new_h = int(h * (args.width / max(1, w)))
        im = im.resize((args.width, new_h))
        items.append((label, nfe, fid, im))

    if not items:
        print("[montage] no images to compose")
        return

    font = ImageFont.load_default()
    line_h = 16
    text_h = line_h * 2 + 8
    total_w = sum(im.size[0] for _, _, _, im in items)
    max_h = max(im.size[1] for _, _, _, im in items)
    canvas = Image.new("RGB", (total_w, max_h + text_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    x = 0
    for label, nfe, fid, im in items:
        canvas.paste(im, (x, 0))
        caption = f"{label}\nNFE={nfe}  FID={fid:.3f}" if fid is not None else f"{label}\nNFE={nfe}  FID=—"
        draw.text((x + 6, max_h + 4), caption, fill=(0, 0, 0), font=font, spacing=2)
        x += im.size[0]

    canvas.save(args.out)
    print(f"[montage] wrote {args.out}")