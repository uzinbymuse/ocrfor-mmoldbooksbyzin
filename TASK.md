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
