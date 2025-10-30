import sys, argparse, subprocess, importlib
from pathlib import Path

import sys, argparse, importlib, subprocess
from pathlib import Path

def run_harness(argv):
    # Try importable CLI
    try:
        ah_cli = importlib.import_module("ablation_harness.cli")
        return ah_cli.main(argv)
    except Exception:
        pass
    # Try console-script entry points
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group="console_scripts")
        for name in ("ablation-harness", "ablation_harness", "ablate"):
            for ep in eps:
                if ep.name == name:
                    func = ep.load()
                    old = sys.argv[:]
                    try:
                        sys.argv = [name, *argv]
                        rv = func()
                        return 0 if rv is None else int(rv)
                    finally:
                        sys.argv = old
    except Exception:
        pass
    # Last resort
    for mod in ("ablation_harness.cli", "ablation_harness"):
        try:
            return subprocess.call([sys.executable, "-m", mod, *argv])
        except Exception:
            continue
    print("[error] No harness CLI found; update your pin or scripts mapping.", file=sys.stderr)
    return 2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = Path(args.config)
    if not cfg.exists():
        print(f"Config not found: {cfg}", file=sys.stderr); return 2
    return run_harness(["run", "--config", str(cfg)])

if __name__ == "__main__":
    sys.exit(main())
    