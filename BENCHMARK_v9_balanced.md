# Official Release Benchmark — `mya_dhamma_v9_balanced.traineddata`

- **Base Checkpoint**: `mya_dhamma_v8_clean`
- **Training Round**: Short-round **400 steps** (`lr = 0.0005`, `skip ratio = 0.5%`, Best Training BCER = `3.421%`)
- **Dataset**: 828 total entries (`gt_human` Batch 1–5 + `syn_v8` + balanced Option B `syn_b6_bal` 125 lines with balanced `၊`/`။`, `ပ္ပ` conjuncts, and `၇/၈/၉` vs `ဂ/စ/မ` digit contrasts — **0 pseudo-labels**)

## 3-Page Benchmark Comparison (`v8_clean` vs `v9` vs `v9_balanced`)

| Benchmark Page | `v8_clean` | `v9` | **`v9_balanced` (Final)** | Key Highlights |
| :--- | :--- | :--- | :--- | :--- |
| **Vol 1 Preface p.11** (21 lines, 1,083 chars, `--psm 7`) | 92.43% (82 errs) | 93.26% (73 errs) | **93.35%** (72 errs / 66.5 per 1k) | `၊` restored **11/12 (91.7%)**; `ဩ` 1/1, `၏` 2/2, `၍` 3/3 |
| **Unseen Vol 2 p.23** (full page, 1,779 chars, `--psm 6`) | 96.51% (62 errs) | 96.51% (62 errs) | **98.20%** (32 errs / 18.0 per 1k) | **`၊` 53/54 (98.1%)**, `။` 16/15, **`ဣ` 9/9 (100%)**, **`၇/၈/၉` all recognized** |
| **Unseen Vol 1 p.17** (`သူလိုရဟန်း`, 1,204 chars, `--psm 6`) | 92.77% (87 errs) | 96.43% (43 errs) | **96.68%** (40 errs / 33.2 per 1k) | **`ပ္ပ` 12/12 (100%)**, `၏` 4/4, `၌` 1/1, `၍` 2/2 |
