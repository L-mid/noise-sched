PY := .venv/Scripts/python.exe

deps-cpu:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e external/ablation-harness[torch-cpu]
	$(PY) -m pip install -e .

deps-gpu:
	$(PY) -m pip install -U pip
	@echo ">>> Install torch/vision/audio for your CUDA from pytorch.org, then:"
	$(PY) -m pip install -e external/ablation-harness
	$(PY) -m pip install -e .

run-baseline:
	$(PY) scripts/run_baseline.py --config configs/baseline.yaml

eval-holdout:
	$(PY) scripts/eval_holdout.py --ckpt checkpoints/baseline/last.pt --nfe 50

plot-snr:
	$(PY) -m noise_sched.plots.snr_plot --name cosine_beta --T 1000