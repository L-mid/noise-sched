### E5 — Beta-scale match (Cosine with Σβ matched to Linear)

**ID:** E5-beta-scale-match  
**Parent study:** Noise-Schedule Diagnosis (CIFAR-10, 32×32, UNet_CIFAR32, 10k steps)

---

#### Question

Does matching the total noise mass Σβ between cosine and linear schedules reduce the FID gap observed between:

- E1: Linear β (baseline) and  
- E2: Cosine β (unscaled)?

In other words: how much of the E1 vs E2 behavior is explained by *total variance injected* vs *shape of ᾱ_t / SNR(t)*?

---

#### Hypothesis

- H1 (shape-matters): After scaling cosine β to match Σβ of linear, we **still** see a systematic FID difference between:
  - Linear β (E1) and
  - Cosine_scaled β (E5).
- H0 (mass-dominates): Once Σβ is matched, cosine_scaled and linear produce similar FID (difference < ~1 FID point at 10k steps with seed 1077).

No directional hypothesis on which schedule is better under Σβ match; this is a *control* experiment.


---

#### Design

- **Dataset:** CIFAR-10 train set for training, CIFAR-10 test statistics for FID.
- **Resolution:** 32×32 RGB.
- **Model:** `unet_cifar32` (same as E1/E2).
- **Optimizer:** Adam, lr = 1e-4 (same as E1/E2).
- **Training steps:** 10,000 iterations.
- **Batch size:** 4.
- **EMA:** enabled (decay = 0.9999), same config as E1/E2.
- **Sampler:** same DDPM / DDIM variant and NFE=50 as in E1/E2.
- **Seed:** 1077 (as in other baseline exps).


**Schedule definition (E5):**

- Let `betas_linear = get_betas_linear(K)` with the same K and parameters as E1.
- Let `betas_cos = get_betas_cosine(K)` as in E2.
- Define scale factor:
  \[
    s = \frac{\sum_t \beta_t^{\text{linear}}}{\sum_t \beta_t^{\text{cosine}}}
  \]
- Define:
  \[
    \beta_t^{\text{cosine_match_linear}} = \text{clamp}(s \cdot \beta_t^{\text{cosine}}, \epsilon, 0.999)
  \]
- Use this schedule wherever `beta_schedule: "cosine_match_linear"` is selected.

We’ll log `sum_beta_linear`, `sum_beta_cosine`, and `sum_beta_cosine_match_linear` for sanity.

---


#### Outcomes & Metrics

Primary endpoint:

- **FID@10k samples, NFE=50**, EMA model, same sampling config as E1/E2.

Secondary diagnostics:

- Σβ over t for:
  - linear (E1),
  - cosine (E2),
  - cosine_match_linear (E5).
- SNR(t) trajectories for all three schedules:  
  \(\text{SNR}_t = \alphā_t / (1 - \alphā_t)\).

---

#### Analysis Plan

- Compare FID(E1), FID(E2), FID(E5) at 10k steps and NFE=50:
  - Look at whether FID(E5) is closer to E1 or E2.
- Plot SNR(t) vs normalized t for all three schedules and mark total Σβ to visually confirm:
  - Σβ(E5) ≈ Σβ(E1)  
  - shape(E5) ≈ shape(E2).
- Qualitatively inspect sample grids from E1, E2, E5 (same seed).

No hyperparameter tuning for E5 specifically; only change is the β schedule kind.

---


#### Stop / De-scope

- If Σβ(E5) differs from Σβ(E1) by >5% due to clamping, log it and still run E5 but note this as a limitation.
- If E5 training diverges (loss NaN) and E1/E2 do not, treat E5 as failed variant and keep the report as “instability under Σβ match.” 


