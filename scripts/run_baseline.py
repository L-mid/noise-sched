import argparse, sys
from pathlib import Path
from ablation_harness import cli as ah_cli   # <-- must succeed
#print(ah_cli)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()

    cfg = Path(args.config)
    if not cfg.exists():
        print(f"Config not found: {cfg}", file=sys.stderr)
        return 2

    # Call the harness CLI in-process
    return ah_cli.main(["run", "--config", str(cfg)])

if __name__ == "__main__":
    sys.exit(main())