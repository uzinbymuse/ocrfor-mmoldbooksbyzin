# TASK — Myanmar Dhamma OCR & Model Training Roadmap (`uzinbymuse/ocrfor-mmoldbooksbyzin`)

## Completed Milestones
- [x] **Phase 1: Volume 1 & Volume 2 Initial OCR -> Word (.docx)** — Converted `Thapyayka_Dhamma_01` (592 pp., 68 TOC items) and `Thapyayka_Dhamma_02` (611 pp., 93 TOC items) with exact original book page numbers and Myanmar/Pali proofreading.
- [x] **Phase 2: Root Cause Analysis & `mya_dhamma` v1/v2 Fine-Tuning** — Discovered 10 missing characters (`ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ`) in official `mya.traineddata` (148 classes); expanded to 159 classes via Replace Top Layer (`v1`: 360 lines, BCER 3.17%; `v2`: 844 lines, BCER 2.74%) and re-OCR'd all 1,203 pages.
- [x] **Phase 3: GitHub Repository & PC 3-System Auto-Update Suite** — Created `uzinbymuse/ocrfor-mmoldbooksbyzin`, uploaded Desktop GUI (`apps/desktop_gui.py`), Local Web UI (`apps/web_app.py`), Batch Folder Tool (`apps/batch_cli.py`), GitHub Auto-Updater (`core/updater.py` + `version.json`), and complete `README.md`.
- [x] **Phase 4: Human-in-the-Loop Ground-Truth Verification (Batch 1)** — Built interactive HTML Proofreader (`gt_proofread_batch1.html`), ingested 51 human-verified lines (`batch1_verified.json`) into `.lstmf` format, and saved reusable skill `myanmar-dhamma-ocr-trainer`.

## Active & Upcoming Milestones (v3 / v4)
- [/] **Phase 5: Batch 2 Human Ground-Truth Collection** — Receive downloaded `batch2_verified.json` (77 lines from Vol. 1 p. 12, Vol. 2 p. 22, Vol. 2 p. 34) and convert to `.lstmf`.
- [x] **Phase 5b: Ground-Truth Dataset Ingestion (81 lines)** — Verified and saved 81 lines (`data/ground_truth/batch_verified_81lines.json`) across Vol. 2 p. 236, Vol. 1 p. 280, Vol. 1 p. 531.
- [x] **Phase 5c: Batch 5 Pack & Ingestion (80 lines)** — Filtered 17 ghost slices, ingested 80 clean lines (`data/ground_truth/batch5_verified.json`).
- [x] **Phase 6: Clean-Slate Training (`mya_dhamma_v8_clean`)** — 100% Human-verified Batches 1–5 (356 lines) + 40 synthetic lines from `base/mya.lstm`. Achieved all-time high **96.51%** on Unseen Vol. 2 p.23, solved `၊ vs ။` regression.
- [x] **Phase 7: Release & Auto-Update Push** — Deployed `mya_dhamma_v8_clean.traineddata`, updated `version.json` (v8.0.0-clean) & `BENCHMARK_v8_clean.md`.
- [x] **Phase 8: Targeted Fine-Tuning (`mya_dhamma_v9_balanced`)** — Option B balanced fine-tune (400 steps, lr=0.0005, BCER 3.421%). Resolved `ပ္ပ → ပွ` (12/12 = 100%), comma `၊` (53/54 = 98.1%), and digits `၇/၈/၉`. Achieved record **98.20%** on Unseen Vol. 2 p.23 and **96.68%** on Unseen Vol. 1 p.17.
- [x] **Phase 9: Web Studio Pre-Processing & Resolution Fix** — Upgraded `apps/web_app.py` to use 180 DPI PyMuPDF rendering, PSM 6 default with PSM 3/4 selector, header-crop toggle, and auto-proofread cleaning to eliminate scan bleed-through and multi-column fragmentation.
- [x] **Phase 10: Multi-Book Grammar & Nissaya Pack (Batch 6, 98 lines)** — Curated 58 real ground-truth lines across Kaccāyana, Medinī, and Rūpasiddhi-gaṇṭhi + 40 targeted synthetic lines. Expanded coverage for critical grammatical words (သဒ္ဒါ: 24, သင်္ဂြိုဟ်: 8, ဉာဏ်: 13, အဋ္ဌကထာ: 5, ဋီကာ: 4, ပိဋကတ်: 4, ဂဏ္ဍိ: 4, ရူပသိဒ္ဓိ: 9, ဓာတ်: 10, စကား: 10, ကင်းစီး: 15) preparing for universal model `mya_dhamma_v10`.
