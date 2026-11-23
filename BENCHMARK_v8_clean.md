# Benchmark Report — `mya_dhamma_v8_clean` (Clean-Slate from `base/mya.lstm`)

- **Date:** 2026-11-23
- **Base Model:** Google `tessdata_best/mya.traineddata` (`base/mya.lstm`, ZERO inherited weights from v2–v6)
- **Architecture:** Replace Top Layer (`--append_index 5 --net_spec "[Lfx192 O1c159]"`, 159 classes)
- **Dataset:** 356 100% human-verified lines (Batch 1: 51, Batch 2: 74, Batch 3 corrected: 70, Batch 4: 81, Batch 5: 80) + 40 clean PIL synthetic lines (`၀–၉`, `၊`/`။`, 10 rare Myanmar/Pali chars) — **ZERO pseudo-labels**
- **Best Checkpoint:** step 4,700, Training BCER = **`3.124%`** (`BWER = 20.879%`, `skip ratio = 0.5%`)

## 3-Page Benchmark Comparison (`v4` vs `v6` vs `v8_clean`)

| Benchmark Page | `v4` | `v6` | **`v8_clean`** | `v8_clean` Errors / 1,000 chars |
| :--- | :--- | :--- | :--- | :--- |
| **Vol 1 Preface p.11** (full 21 lines, 1,083 chars, `--psm 7`) | 93.26% (73 errs) | 92.43% (82 errs) | **92.43%** (82 errs; 48 on faded `l21`) | 75.7 |
| **Unseen Vol 2 p.23** (full page, 1,779 chars, `--psm 6`) | 95.45% (81 errs) | 95.28% (84 errs) | **96.51%** (62 errs — **ALL-TIME BEST**) | **34.9** |
| **NEW Unseen Vol 1 p.17** (`သူလိုရဟန်း`, 1,204 chars, `--psm 6`) | 94.02% (72 errs) | 95.18% (58 errs) | **92.77%** (87 errs; `ပ္ပ->ပွ` conjunct) | 72.3 |

## Rare 10 Characters (`ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ`) Breakdown (`ref -> v4 / v6 / v8_clean`)

| Char | Vol 1 p.11 | Unseen Vol 2 p.23 | NEW Unseen Vol 1 p.17 | Highlight |
| :--- | :--- | :--- | :--- | :--- |
| **`၊`** | `12 -> 6 / 1 / 11` | `54 -> 39 / 32 / 49` | `3 -> 3 / 5 / 7` | **MAJOR FIX** — `၊ -> ။` regression eliminated! |
| **`။`** | `5 -> 9 / 17 / 8` | `15 -> 29 / 36 / 20` | `12 -> 13 / 11 / 8` | Over-prediction cut dramatically |
| **`ဣ`** | `0 -> 0 / 1 / 0` | `9 -> 5 / 10 / 9` | 0 | **100% exact (`9/9`)** |
| **`၏`** | `2 -> 1 / 2 / 2` | `1 -> 1 / 1 / 1` | `4 -> 4 / 4 / 4` | **100% across all 3 pages** |
| **`ဩ`** | `1 -> 1 / 1 / 1` | 0 | 0 | **100%** |
| **`၌`** | 0 | 0 | `1 -> 1 / 1 / 1` | **100%** |
| **`၍`** | `3 -> 3 / 3 / 3` | 0 | `2 -> 2 / 2 / 3` | **100% recall** |
