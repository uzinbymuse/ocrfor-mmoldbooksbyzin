# Benchmark Report — `mya_dhamma_v10.traineddata`

- **Base Checkpoint**: `checkpoints_v9_bal/mya_dhamma_v9_bal_checkpoint`
- **Dataset**: Batch 1–5 Human GT (`332` lines) + `syn_v8` (`59`) + `syn_b6_bal` (`60`) + Batch 6 Real GT (`58 x 3`) + `syn_b6` (`39`) + Batch 7 Surgical Micro-Pack (`34 serif x 3`) = `766` training entries
- **Training**: Surgical fine-tune (`lr=0.0003` -> `0.0004`, best BCER `5.703%`)

## 4-Page Benchmark Comparison

| Benchmark Page | `v8` | `v9_balanced` | **`v10`** | `v10` Errors / 1,000 Chars |
| :--- | :---: | :---: | :---: | :---: |
| **Vol 1 Preface p.11** (21 lines, `--psm 7`) | 92.80% (78) | 93.35% (72) | **92.89%** (77) | 71.1 |
| **Unseen Vol 2 p.23** (1,779 chars, `--psm 6`) | 96.68% (59) | 98.20% (32) | **97.58%** (43) | 24.2 |
| **Unseen Vol 1 p.17 (`သူလိုရဟန်း`)** (1,204 chars) | 96.59% (41) | 96.93% (37) | **97.01%** (36) | 29.9 |
| **Basic Kaccayana Grammar p.5** (1,333 chars) | 94.45% (74) | 88.37% (155) | **94.52%** (73) | 54.8 |
| **4-Page Mean Accuracy** | 95.13% | 94.21% | **95.50%** | — |

## Key Target Checks (`v10`)
- **Unseen Vol 2 p.23**: `ဣ` 9/9, `၏` 1/1, `၊` 50/54, `။` 19/15, digits `၇` 0/1, `၈` 1/1, `၉` 1/1
- **Unseen Vol 1 p.17**: `ပ္ပ` **12/12**, `၏` 4/4, `၌` 1/1, `၍` 2/2
- **Basic Kaccayana p.5**: `ဉာဏ်` **4/4**, `၊` 18/19, `၍` 2/2; remaining physical-font misses: `သင်္ဂြိုဟ်` (0/4), `သေသေးတင်` (0/1), `ဓာတ်အဋ္ဌကထာ` (0/1), `ဋီကာ` (0/1), `ရှားမှရှားပေ` (0/1)
