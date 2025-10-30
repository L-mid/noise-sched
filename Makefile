PY=python

setup:
	$(PY) -m pip install -e .

run-baseline:
	$(PY) scripts/run_baseline.py --config configs/baseline.yaml

eval-holdout:
	$(PY) scripts/eval_holdout.py --ckpt checkpoints/baseline/last.pt --nfe 50

plot-snr:
	$(PY) -m noise_sched.plots.snr_plot --name cosine_beta --T 1000