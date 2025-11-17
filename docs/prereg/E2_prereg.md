# Prereg: E2 (Cosine/short)

Goal: Compare cosine vs linear noise schedules at fixed compute to see if cosine improves FID at NFE=50.
Primary metric: FID@10k samples at NFE=50 (lower is better).
Success criterion (decision rule): Accept if cosine improves FID by ≥6% vs E1 (linear/short) with identical setup. If not improved, record as negative with diagnosis (still valid).
 
## Hypothesis
Cosine β schedule yields a more favorable SNR(t) profile in mid-timesteps, leading to ≥6% lower FID than linear at the same compute (10k steps train, NFE=50 at eval) with identical model/EMA/sampler.

### Design: 
Single-arm run (cosine), matched against E1 (linear) baseline. Identical:
- Model: U-Net (CIFAR-32 variant); same params.
- EMA: enabled; decay = 0.9999 (same as E1).
- Optimizer: Adam, lr=1e-4; same batch/augs.
- Sampler: DDPM sampler.
- Eval: NFE=50, 10k generated samples, same FID pipeline & stats.


CLI (repro):
    python -m ablation_harness.cli run --config configs/study/E2_cosine_short.yaml
    git hash in results.jsonl

    
