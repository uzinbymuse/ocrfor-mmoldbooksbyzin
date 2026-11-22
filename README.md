# ocrfor-mmoldbooksbyzin (`mya_dhamma_v2.traineddata`)

မြန်မာ ဓမ္မစာအုပ်ဟောင်းများ (WinInnwa / ခဲစာလုံး ပုံနှိပ်ဖောင့်) နှင့် ပါဠိဂါထာများအတွက် အထူး Fine-tune ပြုလုပ်ထားသော **Tesseract 5 LSTM** မြန်မာ OCR Model ဖြစ်ပါသည်။

## ဘာကြောင့် ဒီ Model ကို ဖန်တီးရသလဲ?
မူရင်း Google `tessdata_best/mya.traineddata` (`synth20170629`, 148 classes) တွင် အောက်ပါ **မြန်မာ/ပါဠိ စာလုံး ၁၀ လုံး လုံးဝမပါဝင်** ပါ—
> `ဣ` · `ဩ` · `ဪ` · `၏` · `၌` · `၍` · `။` · `၊` · `ဈ` · `ဎ`

ထို့ကြောင့် `merge_unicharsets` + Tesseract 5 **Replace Top Layer** (`--append_index 5 --net_spec "[Lfx192 O1c159]"`) ဖြင့် **148 → 159 classes** သို့ တိုးချဲ့ပြီး တကယ့်ပုံနှိပ်စာအုပ်စာကြောင်းအစစ်များဖြင့် ပြန်လည်သင်ကြားထားပါသည်။

## လုပ်ဆောင်ခဲ့သည့် မှတ်တမ်းအကျဉ်း
1. **Base Model ခွဲထုတ်ခြင်း** — `combine_tessdata -u` / `-e` ဖြင့် `mya.lstm` နှင့် `mya.lstm-unicharset` ထုတ်ယူစစ်ဆေးခဲ့သည်။
2. **Dataset တည်ဆောက်ခြင်း** — သပြေကန် တရားပေါင်းချုပ် အတွဲ (၁) နှင့် (၂) စာအုပ်အစစ်များမှ စာကြောင်းပုံ ၇၂၄ ကြောင်း + synthetic ပါဠိစာကြောင်း ၁၂၀ ကြောင်း (စုစုပေါင်း **၈၄၄ ကြောင်း**၊ Train 776 / Eval 68) ပြင်ဆင်ခဲ့သည်။
3. **Unicharset ပေါင်းထည့်ခြင်း** — `unicharset_extractor --norm_mode 2` + `merge_unicharsets` (159 classes) + `combine_lang_model --pass_through_recoder` ဖြင့် starter model တည်ဆောက်ခဲ့သည်။
4. **Replace Top Layer Fine-tuning** — `lstmtraining` ဖြင့် Iteration **၄,၉၀၀** အထိ လေ့ကျင့်ခဲ့ရာ BCER **2.74%** (`skip ratio = 0%`) အထိ ကျဆင်းခဲ့သည်။
5. **Unseen စာမျက်နှာ ၂၃ စမ်းသပ်ချက်** —
   - မူရင်း `mya`: `သည်! က္ကတိပိ-သီလ` / `မြတ်စွာဘုရားဇါ`
   - အသစ် `mya_dhamma_v2`: **`သည်။ ဣတိပိ-သီလ`** / **`မြတ်စွာဘုရား၏`**

## အသုံးပြုနည်း
```bash
tesseract page.png output --tessdata-dir ./tessdata -l mya_dhamma_v2 --psm 6
```
အင်္ဂလိပ်စာ ရောနေပါက `-l mya_dhamma_v2+eng` ဟု တွဲသုံးနိုင်ပါသည်။
