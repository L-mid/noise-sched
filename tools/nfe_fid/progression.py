from pathlib import Path
from PIL import Image
from .utils import sample_evenly

def add_subparser(sp):
    ap = sp.add_parser("progression", help="Make horizontal denoise strips from per-step frame folders")
    ap.add_argument("folders", nargs="+", help="Each folder contains frames like x_t_000.png ...")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--frame-width", type=int, default=128)
    return ap

def run(args):
    for folder in args.folders:
        d = Path(folder)
        imgs = sorted([p for p in d.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not imgs:
            print(f"[progression] no frames in {d}, skipping")
            continue
        imgs = sample_evenly(imgs, args.max_frames)
        opened = [Image.open(p).convert("RGB") for p in imgs]
        resized = []
        for im in opened:
            w, h = im.size
            new_h = int(h * (args.frame_width / max(1, w)))
            resized.append(im.resize((args.frame_width, new_h)))
        hmax = max(im.size[1] for im in resized)
        strip = Image.new("RGB", (args.frame_width * len(resized), hmax), (255, 255, 255))
        x = 0
        for im in resized:
            y = (hmax - im.size[1]) // 2
            strip.paste(im, (x, y))
            x += im.size[0]
        out = d.with_name(d.name + "_progression.png")
        strip.save(out)
        print(f"[progression] wrote {out}")