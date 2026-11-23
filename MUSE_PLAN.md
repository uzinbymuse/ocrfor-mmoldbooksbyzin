# MUSE_PLAN.md — Muse tasks and plan (for Gemini agent)

Date: 2026-11-23
Owner: Zin ko
Repo: uzinbymuse/ocrfor-mmoldbooksbyzin
To: Gemini agent — please read this file before starting the next step. This is the Muse plan, not a training run.

## 1. Where we are

- Base model missing 10 chars: ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ — expanded 148 to 159 classes.
- v1: 360 lines, BCER 3.17 percent. v2: 844 lines, BCER 2.74 percent, current model in tessdata.
- Thapyaykan Vol 1 and Vol 2 re-OCRed, 1203 pages, 0 empty body pages.
- Batch 1 human ground truth done: 51 verified lines.
- Batch 2 human ground truth pending: 77 lines from Vol 1 p12, Vol 2 p22, Vol 2 p34 — waiting for owner batch2_verified.json.
- Known v2 problem: pseudo-label drift on Vol 1 Preface p11, for example wrong readings that a human spotted. So v3/v4 must train only on 100 percent human-verified ground truth.

## 2. Accuracy target — owner decision 2026-11-23

- Target: above 95 percent accuracy.
- How it is measured: character accuracy on unseen held-out pages, counted against human-verified ground truth, not BCER alone and not training accuracy.
- Benchmark pages: Vol 1 Preface p11 and unseen Vol 2 p23. Compare v2 vs v3/v4 on the same pages, same counting method.
- Count errors per 1000 characters, list every error, do not hide rare characters. Coverage for rare chars ၌ ဪ ဎ ဈ must be reported separately.
- A model is released only if it is above 95 percent on both benchmark pages and better than v2 on both.

## 3. Who does what

### Gemini agent — build and train

- Training, model packaging, OCR re-runs, repo code, version.json, TRAINING_LOG.md updates.
- Convert verified JSON to training data, keep train/eval split recorded.
- Report failures in plain words. Never claim a batch is verified by yourself.

### Zin ko (owner) — human verification, his decision 2026-11-23

- The owner himself takes responsibility for all checks that need a human: proofreading ground truth, Batch 2 verification, final verdict on any unclear letter or word, and phone checks.
- His verdict on a word or letter is final. No agent re-types his corrected words.
- He also decides the next book and the accuracy target.

### Muse — coordination, independent data checks, and gatekeeping

- Keep the plan, the benchmark, and the correction brief in one place.
- Check files and data independently: counts, totals, page ranges, file existence, and live links where reachable. Sample start, middle, and end of every batch, and check boundaries against neighbouring pages.
- Turn the owner human verdict plus Muse data checks into ONE consolidated correction brief, with one exact value per item. No partial briefs, no partial fixes.
- Hold the next batch until the current batch is signed off. Sign-off needs: owner human verdict plus Muse data check plus benchmark above 95 percent.
- Keep records in memory and in the repo logs. Ask the owner for a decision when a choice changes the work.
- Muse does NOT claim human verification, and does NOT treat a Gemini report as proof.

## 4. Muse tasks — concrete list

1. Maintain this plan and the v3/v4 benchmark definition.
2. Receive Batch 2 verified JSON from the owner, check line count and page coverage, and hand it to Gemini in one brief.
3. Define and run the v2 vs v3 benchmark check on p11 and p23, with error lists.
4. Collect every correction into one brief, require a backup before any correction is applied, and re-check after the fix.
5. Update TASK.md / LOG.md / TRAINING_LOG.md status only after a live or data check, never from a report alone.
6. Tell Gemini the next step only after sign-off.

## 5. v3/v4 plan, in order

1. Finish Batch 2: owner verifies 77 lines, Muse checks counts, Gemini converts to training data.
2. Train v3 on Batch 1 plus Batch 2 only, 100 percent human-verified, no pseudo-labels.
3. Benchmark v3 against v2 on p11 and p23. If below 95 percent, add Batch 3/4 for rare characters and train v4. Do not release below target.
4. Release only after sign-off: package traineddata, update version.json and TRAINING_LOG.md, verify the PC auto-updater pulls the new version.

## 6. Rules for Gemini

- Do not start the next batch until the current batch is signed off in this repo or by the owner.
- Do not train v3/v4 on pseudo-labels or unverified OCR output.
- If a rule conflicts with what you see, stop and report, do not guess a value to keep the run moving.
- Keep totals and titles unchanged unless the brief names them. Change only the fields named in the brief.

## 7. Open questions for the owner

- Is batch2_verified.json ready, or still in proofreading?
- Which book is next after Thapyaykan?
- Any font or print variation in the next book that changes the training data?

End of plan. Gemini: if anything in this file is unclear, ask the owner through Muse, do not assume.
