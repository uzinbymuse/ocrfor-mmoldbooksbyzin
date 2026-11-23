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

