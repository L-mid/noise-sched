from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json

def safe_get(d: Dict[str, Any], path: List[str]) -> Optional[Any]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def extract_nfe_fid(jsonl_path: Path) -> Tuple[Optional[int], Optional[float], Optional[Dict[str, Any]]]:
    """
    Parse results.jsonl → (final_nfe, last_val_fid, cfg_dict).
    Looks for FID in a few common places: out.val/fid, val/fid, metrics.val/fid, val.fid
    """
    cfg: Optional[Dict[str, Any]] = None
    last_fid: Optional[float] = None

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

            candidates = [
                safe_get(obj, ["out", "val/fid"]),
                obj.get("val/fid"),
                safe_get(obj, ["metrics", "val/fid"]),
                safe_get(obj, ["val", "fid"]),
            ]
            for c in candidates:
                if isinstance(c, (int, float)):
                    last_fid = float(c)

    nfe: Optional[int] = None
    if cfg is not None:
        nfe_val = safe_get(cfg, ["eval", "final", "nfe"])
        if nfe_val is not None:
            try:
                nfe = int(nfe_val)
            except Exception:
                nfe = None

    return nfe, last_fid, cfg

def find_grid_image(run_dir: Path, grid_name: str = "grid.png") -> Optional[Path]:
    p = run_dir / grid_name
    if p.exists():
        return p
    for q in run_dir.rglob(grid_name):
        return q
    return None

def sample_evenly(paths: List[Path], k: int) -> List[Path]:
    if k >= len(paths):
        return paths
    idxs = [round(i * (len(paths) - 1) / (k - 1)) for i in range(k)]
    return [paths[i] for i in idxs]