### E7 — Baseline-Linear/full (50k steps, NFE=50)

**ID:** E7-baseline-linear-full  
**Parent study:** Noise-Schedule Diagnosis (CIFAR-10, 32×32, UNet_CIFAR32)

---

#### Question

What is the “true” baseline performance of a linear β schedule when we train long enough (50k steps with strong EMA) and evaluate at NFE=50 with 10k samples?

Specifically: how much better is a properly trained linear baseline (E7) compared to the short 10k-step baselines (E1/E2), and how much headroom does this leave for cosine (E8)?

---

#### Hypothesis

- **H1 (longer helps):**  
  Training the linear schedule for 50k steps with EMA=0.999 will **improve FID** vs the 10k-step linear baseline (E1), and produce a stable reference curve for later comparisons.
- **H0 (no real gain):**  
  After ~10k steps, further training gives little or no FID improvement; E7 and E1 have similar FID@10k samples.

No directional hypothesis yet about cosine vs linear here — E7 is the “long, clean” linear anchor.

---

#### Design / Config (planned)

- **Data:** CIFAR-10, 32×32, full train set.
- **Model:** `unet_cifar32` (same architecture as E1).
- **Diffusion:**
  - `beta_schedule: linear`
  - `num_timesteps: 1000` (same as other exps)
- **Train:**
  - `total_steps: 50_000`
  - `batch_size: 4` (match E1/E2)
  - `optimizer: adam`, `lr: 1e-4`
  - `grad_clip: 1.0`
  - deterministic / fixed seed: `1077`
- **EMA:**
  - enabled, `decay: 0.999`
- **Eval:**
  - main metric: `val/fid`
  - final eval at **NFE = 50**
  - use **10k samples** for FID
  - use EMA weights for eval (as in E1/E2).

---

#### Metrics & Analysis Plan

- **Primary outcome:**
  - Best `val/fid` from the run at NFE=50 with 10k samples.
- **Comparisons:**
  - Compare E7 FID to:
    - E1 (linear, 10k steps, short baseline).
    - Later: E8 (cosine, 50k steps) as the “fair” schedule comparison.
- **Secondary diagnostics:**
  - Training loss curve over 50k steps (look for instability / divergence).
  - FID-vs-step trajectory (does it keep improving after 10k? where does it plateau?).

---

#### Logging / Artifacts

- Log full run to `results.jsonl` (loss + metric).
- Save:
  - Final EMA checkpoint.
  - FID vs step curve.
  - Loss vs step curve.
  - At least one sample grid from the final EMA model at NFE=50.

---

#### Risks / Gotchas

- Run may diverge late (long training) → if so, record where/when and keep plots.  
- FID variance is non-trivial at 10k samples → this is just the **baseline anchor**, not the final word on significance.

---

### Definition of Done (E7)

- [X] Linear β run completes to **50k steps** with no crashes.
- [X] EMA with `decay=0.999` is used for evaluation (confirmed in logs / config).
- [X] **FID@10k samples, NFE=50** is logged and summarized for E7. 
- [X] Loss + FID vs step plots are saved and readable. 
- [x] One short paragraph written somewhere (notes / doc) comparing **E7 vs E1** (does longer training help, and by how much?).

Once all boxes are checked, E7 is “done” and ready to serve as the long-train linear baseline for E8/E9/E10.