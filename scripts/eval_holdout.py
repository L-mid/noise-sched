import argparse
import subprocess
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--nfe", type=int, default=50)
    args = p.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}", file=sys.stderr)
        sys.exit(2)

    cmd = [
        sys.executable, "-m", "ablation_harness.cli", "eval",
        "--ckpt", str(ckpt),
        "--nfe", str(args.nfe),
        "--metrics", "fid",
    ]
    print(">> Running:", " ".join(cmd), flush=True)
    sys.exit(subprocess.call(cmd))

if __name__ == "__main__":
    main()