"""
Some usage before i forget:


# one-time
python -m pip install -e .

# E2E baseline (S1)
python scripts/run_baseline.py --config configs/baseline.yaml

# Plot a diagnostic
python -m noise_sched.plots.snr_plot --name cosine_beta --T 1000

"""


from typing import Callable, Dict
import numpy as np

_REGISTRY: Dict[str, Callable[..., "np.ndarray"]] = {}

def register_schedule(name: str):
    def deco(fn):
        if name in _REGISTRY:
            raise ValueError(f"schedule '{name}' already registered")
        _REGISTRY[name] = fn
        return fn
    return deco


def build(name: str, **kw):
    try:
        fn = _REGISTRY[name]
    except KeyError as e:
        raise KeyError(f"unknown schedule '{name}'") from e
    return fn(**kw)


