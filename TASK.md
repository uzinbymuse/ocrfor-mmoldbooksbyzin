# TASK — Myanmar Dhamma OCR & Model Training Roadmap (`uzinbymuse/ocrfor-mmoldbooksbyzin`)

## Completed Milestones
- [x] **Phase 1: Volume 1 & Volume 2 Initial OCR -> Word (.docx)** — Converted `Thapyayka_Dhamma_01` (592 pp., 68 TOC items) and `Thapyayka_Dhamma_02` (611 pp., 93 TOC items) with exact original book page numbers and Myanmar/Pali proofreading.
- [x] **Phase 2: Root Cause Analysis & `mya_dhamma` v1/v2 Fine-Tuning** — Discovered 10 missing characters (`ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ`) in official `mya.traineddata` (148 classes); expanded to 159 classes via Replace Top Layer (`v1`: 360 lines, BCER 3.17%; `v2`: 844 lines, BCER 2.74%) and re-OCR'd all 1,203 pages.
- [x] **Phase 3: GitHub Repository & PC 3-System Auto-Update Suite** — Created `uzinbymuse/ocrfor-mmoldbooksbyzin`, uploaded Desktop GUI (`apps/desktop_gui.py`), Local Web UI (`apps/web_app.py`), Batch Folder Tool (`apps/batch_cli.py`), GitHub Auto-Updater (`core/updater.py` + `version.json`), and complete `README.md`.
- [x] **Phase 4: Human-in-the-Loop Ground-Truth Verification (Batch 1)** — Built interactive HTML Proofreader (`gt_proofread_batch1.html`), ingested 51 human-verified lines (`batch1_verified.json`) into `.lstmf` format, and saved reusable skill `myanmar-dhamma-ocr-trainer`.

## Active & Upcoming Milestones (v3 / v4)
- [/] **Phase 5: Batch 2 Human Ground-Truth Collection** — Receive downloaded `batch2_verified.json` (77 lines from Vol. 1 p. 12, Vol. 2 p. 22, Vol. 2 p. 34) and convert to `.lstmf`.
- [ ] **Phase 6: Train `mya_dhamma_v3` & `mya_dhamma_v4` on 100% Human-Verified Ground Truth** — Combine Batch 1 + Batch 2 (+ optional Batch 3/4 for `၌`, `ဪ`, `ဎ`, `ဈ` coverage), fine-tune cleanly without pseudo-label noise, and benchmark against `v2` on Vol. 1 Preface p. 11 and Unseen Vol. 2 p. 23.
- [ ] **Phase 7: Release & Auto-Update Push** — Package new `.traineddata`, update `version.json` & `TRAINING_LOG.md` on `uzinbymuse/ocrfor-mmoldbooksbyzin`, and verify PC Auto-Updater pulls the latest version.
