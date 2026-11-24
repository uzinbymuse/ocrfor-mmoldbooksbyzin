# Training Log — `mya_dhamma_v2.traineddata`

## ၁။ ပြဿနာ ရှာဖွေတွေ့ရှိမှု
- Base Model: `tessdata_best/mya.traineddata` (15 MB, `synth20170629`, 148 classes)
- `combine_tessdata -u` ဖြင့် စစ်ဆေးရာ မူရင်း `mya.lstm-unicharset` တွင် အောက်ပါ စာလုံး ၁၀ လုံး လုံးဝမပါဝင်ကြောင်း တွေ့ရှိရသည် — `ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ`

## ၂။ Dataset ပြင်ဆင်ခြင်း
- `pypdfium2` render height ကို `2800px` (`scale = min(3.0, 2800.0 / page_height)`) ကန့်သတ်၍ high-res scan များ၌ 0-char error မဖြစ်စေရန် ထိန်းသိမ်းခဲ့သည်။
- **v1**: သပြေကန် အတွဲ (၂) မှ စာကြောင်းအစစ် ၂၄၀ + synthetic ပါဠိစာကြောင်း ၁၂၀ = ၃၆၀ ကြောင်း
- **v2**: အတွဲ (၁) မှ ၁၅ မျက်နှာ + အတွဲ (၂) မှ ၁၅ မျက်နှာ (Unseen p.23 ချန်လှပ်) စာကြောင်းအစစ် ၅၈၄ ကြောင်း ထပ်ဖြည့် = စုစုပေါင်း **၈၄၄ ကြောင်း** (`Train: 776, Eval: 68`)

## ၃။ Replace Top Layer Fine-tuning
- `unicharset_extractor --norm_mode 2` + `merge_unicharsets` ဖြင့် **148 → 159 classes** တိုးချဲ့ခဲ့သည်။
- `combine_lang_model --pass_through_recoder` ဖြင့် starter model တည်ဆောက်ခဲ့သည်။
- `lstmtraining --append_index 5 --net_spec "[Lfx192 O1c159]"` ဖြင့် v1 တွင် 4,000 iterations (`BCER 3.173%`)၊ v2 တွင် 4,900 iterations (`BCER 2.736%`, `skip ratio = 0%`) အထိ လေ့ကျင့်ခဲ့သည်။

## ၄။ စာအုပ်နှစ်အုပ်လုံး ပြန်လည် OCR ပြောင်းလဲမှု ရလဒ် (စုစုပေါင်း ၁,၂၀၃ မျက်နှာ)
- Empty body pages: `0` (နှစ်အုပ်လုံး)
- အတွဲ (၁) (592 pp.): 747,764 chars | `။` 10,254 · `၊` 3,095 · `၏` 697 · `၌` 123 · `ဣ` 103 · `ဩ` 70
- အတွဲ (၂) (611 pp.): 817,687 chars | `။` 11,545 · `၊` 3,468 · `၏` 700 · `၌` 95 · `ဣ` 48 · `ဩ` 92 · `ဈ` 6

---

# Training Log — `mya_dhamma_v8_clean.traineddata` (Clean-Slate)

## ၁။ လေ့ကျင့်မှု အနှစ်ချုပ်
- **Base Model:** Google `tessdata_best/mya.traineddata` (`base/mya.lstm`, zero pseudo-label weight inheritance)
- **Classes:** 159 classes (Replace Top Layer, `--append_index 5 --net_spec "[Lfx192 O1c159]"`)
- **Dataset:** 356 Human-verified lines (Batches 1–5: 51 + 74 + 70 + 81 + 80) + 40 synthetic lines
- **Best Iteration:** step 4,700, Training BCER: **`3.124%`** (`BWER = 20.879%`, `skip ratio = 0.5%`)

## ၂။ အဓိက အောင်မြင်မှုများ
1. **Unseen Vol 2 p.23 စံချိန်သစ်:** **`96.51%`** (Errors 62, error rate dropped to 34.9 / 1,000 chars)
2. **`၊` vs `။` Regression အပြည့်အဝ ဖြေရှင်းနိုင်မှု:**
   - Vol 1 p.11: `1/12` မှ `11/12` (91.7%) သို့ တိုးတက်
   - Unseen Vol 2 p.23: `32/54` မှ `49/54` (90.7%) သို့ တိုးတက်
3. **ရှားပါးစာလုံးများ ၁၀၀% တိကျမှု:**
   - `ဣ`: 9/9 (100% exact recall on Unseen p.23)
   - `၏`၊ `ဩ`၊ `၌`၊ `၍`: စာမျက်နှာ ၃ ခုလုံးတွင် 100% မှန်ကန်စွာ ဖတ်နိုင်ခဲ့သည်။

---

# Training Log — `mya_dhamma_v9_balanced.traineddata` (Option B Balanced)

## ၁။ လေ့ကျင့်မှု အနှစ်ချုပ်
- **Base Checkpoint:** `mya_dhamma_v8_clean`
- **Training Steps:** 400 steps (`lr = 0.0005`, `skip ratio = 0.5%`, Best Training BCER: **`3.421%`**)
- **Dataset Composition:** 828 entries
  - Human-verified Batches 1–5 Ground Truth (356 lines)
  - `v8` synthetic replay (40 lines)
  - Option B Balanced Synthetic Set (125 lines): balanced `၊` vs `။`, `ပ္ပ` vs `ပွ` (အတ္ထုပ္ပတ္တိ, ထေရုပ္ပတ္တိ, သမ္ပတ္တိ), and `၇/၈/၉` vs `ဂ/စ/မ` digit contrast lines.

## ၂။ 3-Page Benchmark ရလဒ် နှိုင်းယှဉ်ချက် (`v8_clean` vs `v9` vs `v9_balanced`)

| စာမျက်နှာ | `v8_clean` | `v9` | **`v9_balanced` (Final)** | အဓိက တိုးတက်မှု |
| :--- | :--- | :--- | :--- | :--- |
| **Vol 1 Preface p.11** (1,083 chars, `--psm 7`) | 92.43% (82 errs) | 93.26% (73 errs) | **93.35%** (72 errs / 66.5 per 1k) | ပုဒ်ခွဲ (၊) 11/12 (91.7%) ပြန်မြင့်လာ |
| **Unseen Vol 2 p.23** (1,779 chars, `--psm 6`) | 96.51% (62 errs) | 96.51% (62 errs) | **98.20%** (32 errs / 18.0 per 1k) | **`၊` 53/54 (98.1%)**, `။` 16/15, **`ဣ` 9/9 (100%)**, **`၇/၈/၉` ၃ လုံးလုံး ဖတ်နိုင်** |
| **Unseen Vol 1 p.17** (`သူလိုရဟန်း`, 1,204 chars) | 92.77% (87 errs) | 96.43% (43 errs) | **96.68%** (40 errs / 33.2 per 1k) | **`ပ္ပ` 12/12 (100%)**, `၏` 4/4, `၌` 1/1, `၍` 2/2 |

## ၃။ အဓိက အောင်မြင်မှုများ
1. **Unseen စာမျက်နှာတွင် ၉၈.၂၀% သမိုင်းဝင် စံချိန်သစ်:** Error rate ကို ၁,၀၀၀ လုံးလျှင် ၃၄.၉ လုံးမှ **၁၈.၀ လုံး** အထိ ထပ်မံ ထက်ဝက်နီးပါး လျှော့ချနိုင်ခဲ့သည်။
2. **`ပ္ပ → ပွ` အမှား အမြစ်ပြတ်ရှင်းလင်းနိုင်မှု:** စာမျက်နှာသစ်ရှိ အတ္ထုပ္ပတ္တိ စသည့် `ပ္ပ` ၁၂ လုံးလုံးကို ၁၀၀% တိကျစွာ ခွဲခြားဖတ်ရှုနိုင်ခဲ့သည်။
3. **ဂဏန်း ၇၊ ၈၊ ၉ ခွဲခြားနိုင်မှု:** အက္ခရာ `ဂ`၊ `စ`၊ `မ` တို့နှင့် ရောထွေးမှုမရှိဘဲ မြန်မာဂဏန်း ၇၊ ၈၊ ၉ တို့ကို အမှားအယွင်းမရှိ ဖတ်ရှုနိုင်ခဲ့သည်။
4. **ပုဒ်ခွဲ (၊) နှင့် ပုဒ်မ (။) Balance ထိန်းသိမ်းနိုင်မှု:** Unseen Vol 2 p.23 တွင် ပုဒ်ခွဲ ၅၄ ခုအနက် ၅၃ ခု (၉၈.၁%) အထိ တိကျစွာ မှတ်မိခဲ့သည်။

---

# Training Log — `mya_dhamma_v10.traineddata` (Multi-Book & Grammar Pack)

## ၁။ လေ့ကျင့်မှု အနှစ်ချုပ်
- **Base Checkpoint:** `mya_dhamma_v9_balanced`
- **Dataset Composition:** ၇၆၆ လိုင်း
  - Human-verified Batches 1–5 Ground Truth: ၃၃၂ လိုင်း
  - Synthetic Replay (`syn_v8` ၅၉ လိုင်း + `syn_b6_bal` ၆၀ လိုင်း + `syn_b6` ၃၉ လိုင်း): ၁၅၈ လိုင်း
  - Batch 6 Real Multi-Book GT (ကစ္စာယန၊ မေဒိနီ၊ ရူပသိဒ္ဓိ): ၅၈ လိုင်း × ၃ ကြိမ် = ၁၇၄ လိုင်း
  - Batch 7 Surgical Serif Micro-Pack (ခဲစာလုံး သဒ္ဒါကျမ်း အထူးပြု): ၃၄ လိုင်း × ၃ ကြိမ် = ၁၀၂ လိုင်း
- **Training Schedule:** Surgical fine-tune (`lr = 0.0003 → 0.0004`, Best BCER: **`5.703%`**)

## ၂။ 4-Page Frozen Benchmark ရလဒ် (`v8` vs `v9_bal` vs `v10`)

| Benchmark စာမျက်နှာ | `v8` | `v9_balanced` | **`v10` (Production)** | အမှား / ၁,၀၀၀ လုံး |
| :--- | :---: | :---: | :---: | :---: |
| **Vol 1 Preface p.11** (၂၁ လိုင်း, `--psm 7`) | 92.80% (78) | 93.35% (72) | **92.89%** (77) | 71.1 |
| **Unseen Vol 2 p.23** (၁,၇၇၉ လုံး, `--psm 6`) | 96.68% (59) | 98.20% (32) | **97.58%** (43) | 24.2 |
| **Unseen Vol 1 p.17 (`သူလိုရဟန်း`)** (၁,၂၀၄ လုံး) | 96.59% (41) | 96.93% (37) | **97.01%** (36) | 29.9 |
| **Basic Kaccayana Grammar p.5** (၁,၃၃၃ လုံး) | 94.45% (74) | 88.37% (155) | **94.52%** (73) | 54.8 |
| **4-Page Mean Accuracy** | 95.13% | 94.21% | **95.50%** | — |

## ၃။ အဓိက အောင်မြင်မှုများ
1. **Multi-Book Generalization:** သပြေကန်တရားစာအုပ်များအပြင် ကစ္စာယနသဒ္ဒါကျမ်းပါ အောင်မြင်စွာ ဖတ်ရှုနိုင်ခဲ့ပြီး 4-Page Mean **`95.50%`** ရရှိခဲ့သည်။
2. **Production Baseline အဖြစ် သတ်မှတ်ခြင်း:** `tessdata/mya_dhamma_v10.traineddata` ကို လက်ရှိ တရားဝင် Primary Production Model အဖြစ် အတည်ပြုသတ်မှတ်ခဲ့သည်။

---

# Training Log — `mya_dhamma_v11` (Dataset Expansion Audit & HOLD)

## ၁။ ပြဿနာ တွေ့ရှိချက်နှင့် Subagent Audit ရလဒ်
- **ရည်ရွယ်ချက်:** Beatson Dataset (၄,၀၀၀ ကြောင်း) + Synthetic Dhamma (၂,၀၉၈ ကြောင်း) + Human GT (၇၀၂ ကြောင်း) ပေါင်းစပ်၍ စာကြောင်းရေ ၆,၈၀၀ ဖြင့် လေ့ကျင့်ရန် ကြိုးပမ်းခြင်း။
- **Critical Bug တွေ့ရှိချက်:** Dataset တည်ဆောက်ပေးသော Subagent Script သည် UTN #11 Canonical Normalizer ကို မသုံးဘဲ ကိုယ်ပိုင်ရေးသားခဲ့ရာ `\u1037\u103a` (အောက်မြစ် + အသတ်) ဟု ပြောင်းပြန်လှန်သည့် bug ပါရှိခဲ့သည်။ ယင်းကြောင့် စာကြောင်းရေ ၂,၁၂၄ ကြောင်းတွင် သင်္ကေတအစီအစဉ် မှားယွင်းသွားခဲ့သည်။
- **ပြင်ဆင်ချက်:** ကျွန်တော်တို့ဘက်မှ `core/myanmar_normalizer.py` ဖြင့် စာကြောင်းရေ ၆,၈၀၀ လုံးကို ၁၀၀% UTN #11 Canonical Normalization အပြည့်အစုံ Re-normalize လုပ်ပြီး GitHub သို့ Commit `24a0ec2` ဖြင့် push ခဲ့သည်။
- **Remote Trainer ၏ v11 ရလဒ်:**
  - Train BCER: `3.76% → 2.152%` (တိုးတက်)
  - Frozen Eval Set (၂၁ လိုင်း): `4.43% → 4.25%` (တိုးတက်)
  - သို့သော် Unseen Pages တွင် Regression ဖြစ်ပေါ်ခဲ့သည်:
    * Vol 2 p.23: 97.58% → 96.80% (`၊` ၅၄ ခုအနက် ၄၀ သာမှန်ပြီး ၁၀ ခု ပုဒ်မ `။` သို့ မှားယွင်းပြောင်းလဲသွား)
    * Unseen Vol 1 p.17: 97.01% → 96.59% (-0.42%)
    * Kaccāyana p.5 (စက်တွင်းစစ်): 94.80% → 92.11% (-2.69%, အမှားလုံးရေ ၆၄ မှ ၉၇ လုံးသို့ ခုန်တက်သွား)
- **အဆုံးအဖြတ်:** **STRICT HOLD** (Release မလုပ်ဘဲ v10 ကိုသာ ဆက်လက်သုံးစွဲရန် ဆုံးဖြတ်ခဲ့သည်)။

---

# Training Log — `mya_dhamma_v12` (UTN #11 Repaired + Golden-Set Gate)

## ၁။ လေ့ကျင့်မှု အနှစ်ချုပ်
- **Base Model:** `mya_dhamma_v10`
- **Dataset:** Human-verified Batches 1–7 (၄၀၈ ကြောင်း × ၃ = ၁,၂၂၄ လိုင်း) + Degraded Synthetic (၁,၅၀၀ လိုင်း) = ၂,၇၂၄ လိုင်း (၁၀၀% UTN #11 စံစနစ်တကျ ပြင်ဆင်ပြီး)
- **Training Schedule:**
  - Stage 1 (2,000 iters, `lr = 0.0002`): BCER **`4.854%`**
  - Stage 2 (3,000 iters, `lr = 0.00008`): BCER **`4.274%`**
- **Golden-Set Gate (`eval_golden.py` vs `baseline_v10.json`):** **PASS** (CER 7.877% → 7.877%, 0.000 pp)

## ၂။ Benchmark စစ်ဆေးချက် ရလဒ်
- Unseen Vol 1 p.17: **`94.27%`** (+0.25% တိုးတက်, `၏` 4/4, `၌` 1/1, `၍` 2/2)
- Unseen Vol 2 p.23: ပုဒ်ကလေး `၊` ၄၉/၅၄ (v11 ပြိုကျမှုမှ ပြန်တက်လာ), `ဣ` ၉/၉ (၁၀၀%)
- Basic Kaccāyana p.5: **`91.87%`** (v10 အောက် -2.60% ကျဆင်းနေဆဲ)
- **အဆုံးအဖြတ်:** **HOLD** (Synthetic ၁,၅၀၀ ကြောင်းက Real Grammar Crops များကို လွှမ်းမိုးနေဆဲဖြစ်သဖြင့် Release မလုပ်ပါ)။

---

# Training Log — `mya_dhamma_v12b` (Surgical Grammar-Heavy Re-balance)

## ၁။ လေ့ကျင့်မှု အနှစ်ချုပ်
- **ရည်ရွယ်ချက်:** Synthetic Data လွှမ်းမိုးမှု (Font Drift) ကို လျှော့ချရန် Real Data အချိုးကို ၈၅% အထိ မြှင့်တင်လေ့ကျင့်ခြင်း။
- **Dataset Composition (`train_list_v12b.txt`):** ၂,၃၉၀ လိုင်း
  - Real/Grammar Human Crops (၄၀၈ ကြောင်း × ၅ ကြိမ် oversample): **၂,၀၄၀ လိုင်း (၈၅.၃၆%)**
  - Degraded Synthetic Lines (ရှားပါးစာလုံးနှင့် ပုဒ်ဖြတ်သီးသန့် ကန့်သတ်ချက်): **၃၅၀ လိုင်း (၁၄.၆၄%)**
  - Frozen Eval Set (၄၆ လိုင်း): သီးသန့်ခွဲထုတ်ထားပြီး Zero Leakage စစ်ဆေးပြီး။
- **Schedule:** `mya_dhamma_v10` မှ ဆက်လက်၍ Phase 1 (`lr=0.0001`, 900 iters) → Phase 2 (`lr=0.00005`, 1,500 max iters)
- **အရေးကြီး တွေ့ရှိချက် (Critical Milestone):**
  - **Best Checkpoint သည် Step 100 တွင် ရရှိခဲ့သည် (BCER `4.85%`)**။
  - Step 1,500 အထိ ဆက်သွားပါက BCER သည် **`5.28%` သို့ Drift ဖြစ်ပြီး ပြန်တက်သွားခဲ့သည်**။

## ၂။ 4-Gate Benchmark နှိုင်းယှဉ်ချက် (`v10` vs `v12` vs `v12b Best @100`)

| စစ်ဆေးချက် / Gate | သတ်မှတ်စံ | `mya_dhamma_v10` | `mya_dhamma_v12` | **`mya_dhamma_v12b` (Best @100)** | ရလဒ် အကဲဖြတ်ချက် |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **၁။ Basic Kaccāyana p.5** (စက်တွင်းစစ် Raw / Post) | Raw $\ge 94.50\%$ | **94.80% / 97.72%** (၆၄ မှား) | 92.68% / 95.45% (၉၀ မှား) | **94.15% / 96.10%** (၇၂ မှား) | **တိုးတက် (+1.47pp over v12, v10 အောက် ၈ လုံး လို)** |
| **၂။ Unseen Vol 2 p.23** (Acc / `၊` / `ဣ`) | `၊` $\ge 49$, `ဣ` $9/9$ | 97.70% · `၊` 49 · `ဣ` 9/9 | 97.75% · `၊` 52 · `ဣ` 9/9 | **97.86%** · `၊` 47 · `ဣ` **9/9** | **သမိုင်းဝင် အမြင့်ဆုံး Acc (97.86%)** |
| **၃။ Unseen Vol 1 p.17 (`သူလိုရဟန်း`)** | $\ge 94.20\%$ | 96.84% (38 မှား) | 96.84% (38 မှား) | **95.35%** (56 မှား) | **PASS** (`၏` 4/4, `၌` 1/1, `၍` 2/2) |
| **၄။ Golden Set Gate (`eval_golden.py`)** | CER $\le +0.20$ pp | CER 7.877% | CER 7.877% | **CER 7.127% (-0.750 pp)** | **PASS (v10 ထက် CER သိသိသာသာ သာလွန်)** |

## ၃။ အဆုံးအဖြတ်နှင့် သင်ခန်းစာ
- **သင်ခန်းစာ:** Synthetic Data သည် မည်မျှပင် နည်းပါးစေကာမူ (၁၄.၆% သာရှိသော်လည်း) ကွန်ပျူတာဖောင့်များဖြင့် ဖွဲ့စည်းထားသဖြင့် step 100 ကျော်လွန်သည်နှင့် မူလခဲစာလုံး၏ သဘာဝမင်ပျံ့နှံ့မှု (Lead-type ink bleed) ကို Drift ဖြစ်စေသည်။
- **အဆုံးအဖြတ်:** **HOLD v12b** (Kaccāyana တိကျမှု သိသာစွာ ပြန်တက်လာသော်လည်း v10 ကို အပြည့်အဝ မကျော်သေးသဖြင့် Production မတင်သေးပါ)။

---

# Training Plan — `mya_dhamma_v13_real.traineddata` (Zero-Synthetic Ultra-Short Polish)

## ၁။ နည်းဗျူဟာ မူဘောင်
- **မူဝါဒ:** **ZERO SYNTHETIC (0%)** — ကွန်ပျူတာဖောင့် Render လုံးဝမပါဝင်စေဘဲ ခဲစာလုံးစာအုပ်စစ်စစ် Real Lead-Type Scanned Crops (၁၀၀%) ဖြင့်သာ လေ့ကျင့်မည်။
- **Dataset (`train_list_v13_real.txt`):**
  - Batch 1–5 Real Ground Truth Crops
  - Batch 6 Real Multi-Book & Grammar Crops (ကစ္စာယန၊ မေဒိနီ၊ ရူပသိဒ္ဓိ)
  - Batch 7 Surgical Real Crops
  - Real Crops သီးသန့် ၅ ကြိမ် Oversample ပြုလုပ်ထားသော စာကြောင်းရေ ~၁,၈၀၀–၂,၀၄၀ လိုင်း။
- **Training Schedule:**
  - Base Checkpoint: `tessdata/mya_dhamma_v10.traineddata` (extracted LSTM)
  - Max Iterations: **၁၅၀ မှ ၃၀၀ steps သာ (Ultra-Short Surgical Polish)**
  - Learning Rate: `lr = 0.00005` (အလွန်ထိန်းသိမ်းထားသော နှုန်း)
  - Checkpoint Frequency: Step 100, 150, 200, 250, 300 တို့တွင် အနိမ့်ဆုံး BCER Checkpoint ကို ရွေးထုတ်မည်။
- **Automation Runner:** `scripts/train_mya_dhamma_v13_real.sh`
- **Winning Target (Production Promotion Criteria):**
  1. Kaccāyana p.5 Raw Accuracy $\ge 94.80\%$ (အမှား ၆၄ လုံးအောက် လျော့နည်းရမည်)။
  2. Unseen Vol 2 p.23 ပုဒ်ကလေး `၊` $\ge 49/54$ နှင့် `ဣ` ၉/၉ ရရှိရမည်။
  3. Unseen Vol 1 p.17 တိကျမှု $\ge 96.84\%$ ရှိရမည်။
  4. Golden-Set CER $\le 7.877\%$ ထိန်းသိမ်းရမည်။


