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
