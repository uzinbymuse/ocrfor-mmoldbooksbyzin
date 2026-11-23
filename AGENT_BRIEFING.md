# AGENT_BRIEFING.md — Handover & Status Briefing for Collaborating Agents

**To:** Muse (Coordinator & Gatekeeper), Training Agents, Reviewing Agents  
**From:** Gemini Agent (Core & Data Engineering)  
**Date:** 2026-11-23  
**Repository:** [`uzinbymuse/ocrfor-mmoldbooksbyzin`](https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin)  
**Branch:** `main`  
**Base Commit:** `bbe6d29`  

---

## 1. Executive Summary
* **GitHub Sync & Permissions**: The local workspace is fully connected with remote `https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin.git`. Collaborator write access has been accepted for `uzinlay85`, enabling automated `git pull` and `git push` pipelines.
* **Ground-Truth Ingestion**: A new verified dataset of **81 lines** has been proofread, verified, and ingested into `data/ground_truth/batch_verified_81lines.json`.
* **Logs & Milestones**: Updated `LOG.md` and `TASK.md` (Phase 5b) accordingly.

---

## 2. Ingested Dataset Details
* **File Location:** `data/ground_truth/batch_verified_81lines.json`
* **Total Line Count:** 81 lines
* **Data Schema:**
  ```json
  [
    {
      "id": "v2_p236_l01",
      "text": "..."
    }
  ]
  ```
* **Coverage Scope:**
  1. `v2_p236_l01` – `l28` (28 lines): Buddha Gunakatha Nissaya (သုဂတော၊ လောကဝိဒူ၊ အနုတ္တရောပုရိသဒမ္မသာရထိ၊ သတ္ထာ ဒေဝမနုဿာနံ၊ ဗုဒ္ဓေါ).
  2. `v1_p280_l01` – `l27` (27 lines): Vinaya/monastic discourse on Samanera robe veneration vs Anāgāmī level.
  3. `v1_p531_l01` – `l26` (26 lines): Shwegyin Sasana Paing Sayadawgyi's Owada verse (ပါရမီဒဗ္ဗသမ္ဘာရ သာရောဧဝ အဘိန္နကော...).

---

## 3. Current Project State
| Component | Status | Details |
|---|---|---|
| **Git Working Tree** | Clean / Synced | Up to date with `origin/main` |
| **Active Model** | `mya_dhamma_v2` | Located at `tessdata/mya_dhamma_v2.traineddata` (BCER 2.74%) |
| **Human GT Pool** | 132+ lines ready | Batch 1 (51 lines) + Batch 3 (81 lines) |
| **Pending Ingestion** | Batch 2 (77 lines) | Vol. 1 p. 12, Vol. 2 p. 22, Vol. 2 p. 34 |

---

## 4. Operational Directives for Subsequent Agents
1. **Strict Human-Ground-Truth Policy**:
   As stipulated in `MUSE_PLAN.md`, training for `v3` / `v4` must strictly avoid pseudo-label drift. Only 100% human-verified text can be fed into the training pipeline.
2. **Batch 2 Merging**:
   When the owner provides `batch2_verified.json` (77 lines), merge with existing datasets (`batch1_verified.json` and `batch_verified_81lines.json`) before generating `.lstmf` files.
3. **Accuracy Gatekeeper Rule**:
   Before releasing any updated `.traineddata` file, evaluate against benchmark pages (Vol. 1 Preface p. 11 and Unseen Vol. 2 p. 23). The release threshold is strictly **> 95% character accuracy** and superior to `v2` on both pages.
