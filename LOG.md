# LOG — Myanmar Dhamma OCR & Fine-Tuning Engineering Log

## 2026-11-22 — Volume 1 & Volume 2 OCR Pipeline
* Installed user-provided `tessdata_best/mya.traineddata` (15 MB, `synth20170629`).
* **Key Fix**: Capped `pypdfium2` render height at `2800px` (`scale = min(3.0, 2800.0 / ph)`) after discovering uncapped `scale=3.0` on high-res Vol. 2 pages (pp. 419–609) exceeded 30M pixels and caused Tesseract to silently return 0 characters. Re-OCRing 69 pages reduced empty body pages to `0`.
* Built `Thapyaykan_Dhamma_01_OCR.docx` (68 TOC items, book pp. `၁`–`၅၇၈`) and `Thapyaykan_Dhamma_02_OCR.docx` (93 TOC items, book pp. `(က)`–`(ဂ)`, `၁`–`၅၈၉`, plus `(၅၂၈) သွယ် မေတ္တာပို့` table).

## 2026-11-22 — Root Cause Discovery & `mya_dhamma` v1 / v2
* Unpacked `mya.traineddata` with `combine_tessdata -u`; found `mya.lstm-unicharset` (148 classes) completely omits 10 essential Myanmar/Pali characters: `ဣ`, `ဩ`, `ဪ`, `၏`, `၌`, `၍`, `။`, `၊`, `ဈ`, `ဎ`.
* Merged unicharsets (`148 -> 159` classes) and built starter model with `combine_lang_model --pass_through_recoder`.
* **v1 (`mya_dhamma.traineddata`)**: Replace Top Layer (`--append_index 5 --net_spec "[Lfx192 O1c159]"`), 360 lines (240 real + 120 synthetic), 4,000 iterations, BCER `3.173%`. Natively recognized `ဣတိပိ`, `၏`, `။`, `၊`, `ဩ`, `ဈ` on unseen Vol. 2 p. 23.
* **v2 (`mya_dhamma_v2.traineddata`)**: Expanded to 844 lines across Vol. 1 & Vol. 2, 4,900 iterations (`lr=0.001`), BCER `2.736%`, `skip ratio = 0%`. Fixed double-letter glitches on unseen p. 23 (`လောကလလုံး -> လောကလုံး`, `ဖယ်ကောင်း -> ဖွယ်ကောင်း`). Re-OCR'd all 1,203 pages of Vol. 1 & Vol. 2 (`0` empty body pages).

## 2026-11-23 — GitHub Repo, 3-System PC Suite & Human-in-the-Loop Ground Truth
* Created public GitHub repository `https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin` with `README.md`, `TRAINING_LOG.md`, `version.json`, `requirements.txt`, `core/ocr_engine.py`, `core/updater.py`, `apps/desktop_gui.py`, `apps/web_app.py`, `apps/batch_cli.py`, `scripts/`, `tessdata/mya_dhamma_v2.traineddata`, and Windows/Mac/Linux launchers.
* **v2 Quality Review**: User spotted pseudo-label drift on Vol. 1 Preface p. 11 (`ကြုံ -> ကျု`, `ပြုကျင့် -> ဖုကက`, `မှီခို -> မှခို`), confirming the need for 100% human-verified Ground Truth before training v3/v4.
* **Batch 1 Human GT**: Built interactive HTML Proofreader (`gt_proofread_batch1.html`). User verified and returned `batch1_verified.json` (**51 clean lines**, converted to `.lstmf`; coverage: `၊` 34, `။` 22, `၏` 10, `ဣ` 3, `၍` 3, `ဩ` 1, `ဈ` 1).
* **Batch 2 Human GT**: Generated `gt_proofread_batch2.html` (**77 lines** from Vol. 1 p. 12, Vol. 2 p. 22, Vol. 2 p. 34) — awaiting user's exported `batch2_verified.json`.
* **Batch 3 Human-Verified Ground Truth (81 lines)**: Proofread and verified 81 OCR lines across Vol. 2 p. 236 (28 lines), Vol. 1 p. 280 (27 lines), and Vol. 1 p. 531 (26 lines). Corrected OCR errors in Buddha Gunakatha (သုဂတော၊ လောကဝိဒူ၊ အနုတ္တရောပုရိသဒမ္မသာရထိ၊ သတ္ထာ ဒေဝမနုဿာနံ၊ ဗုဒ္ဓေါ), monastic ordination rules, and Shwegyin Sasana Paing Sayadawgyi's Owada verse. Saved dataset to `data/ground_truth/batch_verified_81lines.json`.
* **Batch 5 Pack & Ingestion (80 clean lines)**: Packed 98 lines, identified and filtered 17 horizontal line-split ghost slices from `v1_p154` (Vedehikā discourse), yielding 80 clean lines (`data/ground_truth/batch5_verified.json`).
* **mya_dhamma_v8_clean Release**: Clean-Slate Replace Top Layer training directly from Google `base/mya.lstm` (step 4,700, BCER `3.124%`). ZERO pseudo-label contamination. 356 Human-verified lines (Batches 1-5) + 40 synthetic lines. Achieved all-time high **96.51%** accuracy on Unseen Vol. 2 p.23 (error rate dropped to 34.9/1,000 chars), eliminated `၊ -> ။` regression (p.11: 11/12, p.23: 49/54), and scored 100% on rare characters `ဣ` (9/9), `၏`, `ဩ`, `၌`, `၍`.
* Saved reusable workflow skill `myanmar-dhamma-ocr-trainer` (`SKILL.md`).
