import numpy as np
from .registry import register_schedule

@register_schedule("cosine_beta")
def cosine_beta(T: int, s: float = 0.008) -> np.ndarray:
    """
    Nichol & Dhariwal cosine schedule (beta_t derived from alpha_bar).
    Returns betas with shape [T] in (0,1).
    """
    t = np.linspace(0, T, T + 1)
    f = lambda u: np.cos((u / T + s) / (1 + s) * np.pi / 2) ** 2
    alpha_bar = f(t) / f(0)
    betas = np.clip(1 - (alpha_bar[1:] / alpha_bar[:-1]), 1e-8, 0.999)
    return betas

