# E6 — Sampler parity (DDPM vs DDIM at NFE=50)

**ID:** E6-sampler-parity-50nfe  
**Parent study:** Noise-Schedule Diagnosis (CIFAR-10, 32×32, UNet_CIFAR32, 10k steps)

---

## 1. Question

For fixed β schedules (linear / cosine) and NFE = 50:

- Do **DDPM** and **DDIM** samplers yield similar FID (≈ parity), and  
- Does the **linear vs cosine** ranking stay the same across samplers?

This is a **sanity / control** experiment, not a sampler sweep.

---

## 2. Hypotheses

- **H0 (parity):**  
  For a given β schedule (linear or cosine),  
  \|FID(DDPM, 50) − FID(DDIM, 50)\| ≤ ~1, and linear vs cosine ranking is unchanged.

- **H1 (sampler-effects):**  
  At least one schedule has \|ΔFID\| ≥ ~2 **and/or** the linear vs cosine ranking flips between DDPM and DDIM.

---

## 3. Design

- **Dataset:** CIFAR-10 (train), standard CIFAR-10 FID stats.  
- **Resolution:** 32×32 RGB.  
- **Model:** `unet_cifar32`.  
- **Training:** 10k steps, batch size 4, Adam lr=1e-4, EMA decay=0.9999 (as in E1/E2/E5).  
- **Schedules:**
  - linear β (E1 style)  
  - cosine β (E2 style)
- **Samplers:**
  - DDPM  
  - DDIM

### 3.1 Conditions (2×2 grid)

We compare 4 conditions at NFE=50, FID@10k samples:

| ID | Schedule | Sampler | NFE | Notes                  |
|----|----------|---------|-----|------------------------|
| C1 | linear   | DDPM    | 50  | reuse E1 final        |
| C2 | cosine   | DDPM    | 50  | reuse E2 final        |
| C3 | linear   | DDIM    | 50  | new E6 run            |
| C4 | cosine   | DDIM    | 50  | new E6 run            |

- **Seed:** 1077 base (consistent with E1–E5).  
- **Eval:** FID@10k, EMA model, same stats file as E1/E2/E5.

---

## 4. Metrics & Analysis Plan

### 4.1 Primary metrics

- FID(C1), FID(C2), FID(C3), FID(C4) at NFE=50, 10k samples, EMA.

### 4.2 Derived checks

For each schedule:

- ΔFID_linear  = FID(C3) − FID(C1)  
- ΔFID_cosine  = FID(C4) − FID(C2)

Ranking consistency:

- Under **DDPM**: compare FID(C1) vs FID(C2).  
- Under **DDIM**: compare FID(C3) vs FID(C4).  

We label outcome:

- **“Parity”** if both |ΔFIDs| ≤ ~1 and linear vs cosine ranking is the same.  
- **“Sampler-effects”** if any |ΔFID| ≥ ~2 or the ranking flips.  
- **“Inconclusive”** if FIDs are noisy / overlapping.

Qualitative:

- One 4-way grid (linear/cosine × DDPM/DDIM) at shared seed for visible artifacts.

---

## 5. Commands Used

### (new) Runs:

```bash
python -m ablation_harness.cli run \
  --config configs/study/E6/E6_sampler_parity_linear_ddim_10k.yaml \
  --out_dir /content/drive/MyDrive/noise-sched-e6/runs_linear_ddim
```

### plotting fid together
```bash
  python -m tools.nfe_fid.cli plot  \
    docs/assets/E6/e1_data \
    docs/assets/E6/e2_data \
    + new two
    --out-plot docs/assets/E6/e6_plots/nfe_fid_linear.png --out-csv docs/assets/E6/e6_plots/nfe_fid_linear.csv
```

### plotting samples from linear - cosine - ddpm - ddim
```bash
python tools/plot_grids_together.py \
    --ckpt-linear docs/assets/E6/e1_data/last.pt \
    --ckpt-cosine docs/assets/E6/e2_data/last.pt \
    --out-dir docs/assets/E6/e6_plots \
    --out-name e6_linear_cosine_ddpm_ddim.png \
    --K 1000 \
    --nfe 50 \
    --batch-size 36 \
    --img-size 32 \
    --seed 1077 \
    --device cuda \
    --title-linear-ddpm  "E1-linear — DDPM" \
    --title-linear-ddim  "E1-linear — DDIM" \
    --title-cosine-ddpm  "E2-cosine — DDPM" \
    --title-cosine-ddim  "E2-cosine — DDIM"
```

### plotting fid heatmap
```bash
    python -m tools.plot_e6_heatmap \
    --fid-linear-ddpm 193.1787 \        
    --fid-linear-ddim 194.9572 \
    --fid-cosine-ddpm 194.2269 \
    --fid-cosine-ddim 194.1266 \
    --out-plot docs/assets/E6/e6_plots/e6_fid_heatmap.png
```


## 6. Definition of Done 

- [X] **Runs finished**
  - [X] `E6_sampler_parity_linear_ddim_10k.yaml` (E6-linear-ddim)
  - [X] `E6_sampler_parity_cosine_ddim_10k.yaml` (E6-cosine-ddim)

- [X] **FID summary**
  - [X] Read FID@10k, NFE=50 for C1–C4 (linear/cosine × DDPM/DDIM).
  - [X] Save a tiny 2×2 table as CSV/JSON for the repo + paste into the write-up.

- [X] **Figure**
  - [X]:
    - one 4-way sample grid (linear vs cosine × DDPM vs DDIM),
    - a heatmap figure of the 4 FIDs.

- [ ] **Short write-up (~½–1 page)**
  - [ ] Restate the E6 question in 2–3 sentences.
  - [ ] Include the FID table and report ΔFID for linear and cosine.
  - [ ] Make an explicit call: **Parity / Sampler-effects / Inconclusive**.
  - [ ] Add 1–2 sentences on speed differences and any visible sampler artifacts.

- [ ] **Public touch**
  - [ ] Comment on the prereg GitHub issue with:
    - [ ] Linked writeup
    - [ ] 2 paragraph summary of the E6 result.

