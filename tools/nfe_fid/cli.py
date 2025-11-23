"""
Useage:

# 1) NFE↔FID plot (+ CSV)
python -m tools.nfe_fid.cli plot docs/assets/E3/noise-sched-e3a/runs docs/assets/E3/noise-sched-e3b/runs docs/assets/E3/noise-sched-e3c/runs \
  --out-plot docs/assets/E3/nfe_fid_linear.png --out-csv docs/assets/E3/nfe_fid_linear.csv

# 2) Montage of each run's grid.png
python -m tools.nfe_fid.cli montage docs/assets/E3/noise-sched-e3a/runs docs/assets/E3/noise-sched-e3b/runs docs/assets/E3/noise-sched-e3c/runs \
  --grid-name grid.png --out docs/assets/E3/grids_montage.png

  
# 3) Progression strips (if you exported per-step frames)
python -m tools.nfe_fid.cli progression out/frames_nfe10 out/frames_nfe20 out/frames_nfe50 \
  --max-frames 16 --frame-width 128

  
path: noise-sched-e3<abc>/runs/diffusion_10k/diffusion_10k__unet_cifar32__cifar10__adam__lr1e-04__ema1__seed=1077/eval/step_x/grid.png


Current:
  python -m tools.nfe_fid.cli plot  \
    docs/assets/E6/e1_data \
    docs/assets/E6/e2_data \
    docs/assets/E6/noise-sched-e6a/runs \
    docs/assets/E6/noise-sched-e6b/runs \
    --out-plot docs/assets/E6/e6_plots/nfe_fid_linear.png --out-csv docs/assets/E6/e6_plots/nfe_fid.csv


"""


import argparse
from . import plot, montage, progression

def main():
    ap = argparse.ArgumentParser(prog="nfe-fid", description="NFE↔FID plotting & visuals")
    sp = ap.add_subparsers(dest="cmd", required=True)

    plot.add_subparser(sp)
    montage.add_subparser(sp)
    progression.add_subparser(sp)

    args = ap.parse_args()
    fn = {"plot": plot.run, "montage": montage.run, "progression": progression.run}[args.cmd]
    return fn(args)

if __name__ == "__main__":
    raise SystemExit(main())