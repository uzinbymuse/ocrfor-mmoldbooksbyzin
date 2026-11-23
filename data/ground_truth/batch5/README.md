# Ground-Truth Verification Pack — Batch 5 (98 lines, Pre-read by v6)

Target weaknesses:
1. Myanmar digits `၀-၉` (especially `၇`, `၈`, `၉`): `v1_p154` (45 lines)
2. Comma `၊` vs full stop `။` distinction: `v1_p238` (24 lines), `v2_p415` (29 lines)

## Contents
- `gt_proofread_batch5.html` — Self-contained interactive proofreader (Left: Clean Ink Strip, Right: Editable GT, Enter = next line, Export `batch5_verified.json`)
- `batch5_meta.json` — Line IDs, dimensions, and `v6` pre-read draft texts (98 lines)
- `images/` — All 98 cropped line PNG images

## Next Step
1. Open `gt_proofread_batch5.html` in browser, verify/correct all lines, and export `batch5_verified.json`.
2. Return `batch5_verified.json` (zero pseudo-labels) for `mya_dhamma_v7` training + 3-page benchmark.
