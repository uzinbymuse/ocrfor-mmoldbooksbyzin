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

