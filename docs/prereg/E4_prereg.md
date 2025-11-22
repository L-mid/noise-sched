# Prereg — E4 (Cosine NFE Sweep · Pilot)

**Project:** Noise Schedule Diagnosis (Oct 29 → Nov 24, 2025)  
**ID:** E4  
**Owner:** l-mid  
**Status:** Preregistered (pilot)

## Purpose
Map the NFE↔FID frontier for **cosine β** to compare against E3 (linear) and guide full runs.

## Hypothesis
At fixed training budget, cosine yields **lower FID at low NFE (10–20)** than linear due to higher effective mid-t SNR.

## Design
- **Data:** CIFAR-10 (32×32), standard split & norm.
- **Model:** Same U-Net/params/EMA as baseline.
- **Schedule:** **cosine β** (this exp only).
- **Sampler:** **DDPM** (fix to avoid schedulexsampler confound).
- **NFE:** {10, 20, 50}.
- **Train budget:** **10k** steps (pilot).
- **Seeds:** {0}.
- **Eval samples:** **5k** per NFE.
- **FID stats:** Same locked file as baseline.

## Outcomes
- **Primary:** FID at NFE ∈ {10,20,50}.  
- **Secondary:** KID, wall-time/sample, **SNR(t)** trajectory, loss curve.

## Decision & Stops (pilot)
- Descriptive pilot (no pass/fail); negative finding is recorded.  
- Early stop on NaN/Inf >1% steps or E2E smoke fails twice/day; fix infra first.

## Confounds (controlled)
Params, EMA, optimizer, sampler, eval sample count, and stats file are identical to E3.

## Analysis
Plot **NFE vs FID** (cosine), **SNR(t)** for cosine, loss vs steps. Keep E3’s linear SNR handy for report overlay.

## Commands (example)
```bash
# train (short pilot)
python -m ablation_harness.cli run \
  --config configs/study/E4/E4a-linear-nfe10.yaml   \
  --out_dir /content/drive/MyDrive/noise-sched-e4a/runs

python -m ablation_harness.cli run \
  --config configs/study/E4/E4b-linear-nfe20.yaml   \
  --out_dir /content/drive/MyDrive/noise-sched-e4b/runs

python -m ablation_harness.cli run \
  --config configs/study/E4/E4c-linear-nfe50.yaml   \
  --out_dir /content/drive/MyDrive/noise-sched-e4c/runs
```

