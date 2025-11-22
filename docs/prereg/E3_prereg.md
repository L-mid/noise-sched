# E3 — Baseline-Linear NFE sweep (pilot) — prereg

Hypothesis. With a linear β schedule, FID improves as NFE increases; the resulting NFE↔FID curve forms a diminishing-returns frontier. This pilot establishes the baseline frontier (no schedule change).

Config diff. Identical to E1/E2 except:
- diffusion.beta_schedule = "linear"
- eval.final.nfe ∈ {10, 20, 50} (three runs)
- eval.final.n_samples = 5000 (pilot speed)

Same EMA (0.9999), same model, same sampler (DDPM) for final eval.

**Primary metric**. FID@final on CIFAR-10 (32×32) with n_samples=5000 per run.
**Secondary.** Wall-time, NFE vs FID plot, qualitative grids (fixed seeds).

**Stop/decision rules.** If any run crashes or shows NaNs/Inf >1% steps ⇒ abort that arm, record diagnosis, re-run once; else accept pilot result even if negative.
Confounds controlled. Same network, optimizer, EMA, eval pipeline, stats file, and seed policy.
Planned plots. (1) NFE↔FID curve for linear (10/20/50) (2) Training loss curves (overlay) (3) Sample grid panels.