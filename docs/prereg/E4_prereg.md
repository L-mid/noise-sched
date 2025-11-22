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

## All Commands 
```bash
# training 10k steps
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

```bash
# Plotting:
python tools/plot_nfe_fid.py docs/assets/E4/noise-sched-e4a/runs \
                             docs/assets/E4/noise-sched-e4b/runs \
                             docs/assets/E4/noise-sched-e4c/runs \
  --out-plot docs/assets/E4/E4_plots/nfe_fid_linear.png --out-csv docs/assets/E4/E4_plots/nfe_fid_linear.csv \
  --make-montage --grid-name grid.png 
```



## Definition of Done — E4 (Cosine NFE Sweep · Pilot)

- [X] **Train (10k steps)** with cosine β using the prereg’d config; save run logs + git hash + config diff.
- [X] **Evaluate at NFE = {10, 20, 50}**, 5k samples each, with the **locked FID stats**; write FID/KID/wall-time to `results.jsonl`.
- [ ] **Artifacts saved:** 
      1) NFE↔FID plot (cosine), 2) SNR(t) plot (cosine), 3) loss vs steps.
- [ ] **Tests updated (≥3 for this exp):** schedule shape in-range/monotone, SNR correctness, sampler indexing + an E2E smoke.
- [ ] **Short write-up (≥1 page):** purpose, setup, curves, quick interpretation; link plots and run hash.
- [ ] **Public touch:** post plots + one-paragraph summary (links to run + stats file path).

**Outcome labeling (pilot):**  
- *Supports hypothesis* if cosine shows lower FID at **NFE 10 or 20** than E3 (linear pilot).  
- *Does not support* otherwise. Either result is **complete** if the checklist above is satisfied.