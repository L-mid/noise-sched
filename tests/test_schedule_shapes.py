import importlib
import numpy as np
from noise_sched.schedules.registry import build

def test_cosine_beta_shape_and_range():
    importlib.import_module("noise_sched.schedules.cosine")
    T = 1000
    betas = build("cosine_beta", T=T)
    assert betas.shape == (T,)
    assert np.all(betas > 0)
    assert np.all(betas < 1)