# ocrfor-mmoldbooksbyzin (`mya_dhamma_v9_balanced.traineddata`)

မြန်မာ ဓမ္မစာအုပ်ဟောင်းများ (WinInnwa / ခဲစာလုံး ပုံနှိပ်ဖောင့်) နှင့် ပါဠိဂါထာများအတွက် အထူး Fine-tune ပြုလုပ်ထားသော **Tesseract 5 LSTM OCR Model (၉၈.၂၀% တိကျမှုစံချိန်သစ်)** နှင့် PC ပေါ်တွင် PDF မှ Word (`.docx`) သို့ အလွယ်တကူပြောင်းလဲနိုင်သည့် **စနစ် ၃ မျိုး + GitHub Auto-Update Suite** ဖြစ်ပါသည်။

## ၁။ ဘာကြောင့် ဒီ Model ကို ဖန်တီးရသလဲ?
မူရင်း Google `tessdata_best/mya.traineddata` (`synth20170629`, 148 classes) တွင် အောက်ပါ **မြန်မာ/ပါဠိ စာလုံး ၁၀ လုံး လုံးဝမပါဝင်** ပါ—
> `ဣ` · `ဩ` · `ဪ` · `၏` · `၌` · `၍` · `။` · `၊` · `ဈ` · `ဎ`

ထို့ကြောင့် `merge_unicharsets` + Tesseract 5 **Replace Top Layer** (`--append_index 5 --net_spec "[Lfx192 O1c159]"`) ဖြင့် **148 → 159 classes** သို့ တိုးချဲ့ကာ သပြေကန် တရားပေါင်းချုပ် အတွဲ (၁) နှင့် (၂) မှ စာကြောင်းအစစ်များအပါအဝင် စာကြောင်း **၈၄၄ ကြောင်း** ဖြင့် Iteration **၄,၉၀၀** အထိ လေ့ကျင့်ခဲ့ရာ BCER **2.74%** (`skip ratio = 0%`) ရရှိခဲ့ပါသည်။ (အသေးစိတ်မှတ်တမ်းကို [`TRAINING_LOG.md`](TRAINING_LOG.md) တွင် ကြည့်နိုင်ပါသည်။)

## ၂။ စတင်မသုံးမီ လိုအပ်ချက်များ (Prerequisites)
1. **Python 3.9+** — https://www.python.org/downloads/ (Windows တွင် install လုပ်စဉ် **"Add Python to PATH"** ကို အမှန်ခြစ်ပေးပါ)
2. **Tesseract OCR 5.x** —
   - **Windows**: UB-Mannheim Tesseract installer မှ သွင်းပါ (`C:\Program Files\Tesseract-OCR\tesseract.exe`)
   - **macOS**: `brew install tesseract`
   - **Ubuntu / Debian**: `sudo apt install tesseract-ocr`

## ၃။ ထည့်သွင်းနည်း (Quick Setup)
Repository ကို Download ZIP (သို့မဟုတ် `git clone`) လုပ်ပြီး folder ထဲဝင်ကာ တစ်ကြိမ်သာ install လုပ်ပါ—
- **Windows**: `install.bat` ကို Double-click နှိပ်ပါ
- **macOS / Linux**:
  ```bash
  chmod +x install.sh run_*.sh
  ./install.sh
  ```
*(သို့မဟုတ် `pip install -r requirements.txt` ဖြင့် `pypdfium2`, `pypdf`, `Pillow`, `python-docx` ၄ ခုသာ သွင်းရန်လိုအပ်ပါသည်။)*

## ၄။ PC တွင် PDF → Word (`.docx`) ပြောင်းလဲအသုံးပြုနည်း (စနစ် ၃ မျိုး)

### (က) Desktop App (အပေါ့ပါးဆုံး Tkinter GUI)
- Windows: `run_desktop.bat` | Mac/Linux: `./run_desktop.sh`
- **Browse...** ဖြင့် PDF ရွေး → လိုအပ်ပါက **Page Offset** (မူရင်းစာမျက်နှာနံပါတ်ကွာဟချက်) နှင့် **CPU Cores** ညှိ → **Start OCR → Word (.docx)** နှိပ်ပါ။

### (ခ) Local Web UI (Browser မှ သုံးရန်)
- Windows: `run_web.bat` | Mac/Linux: `./run_web.sh`
- ဘရောက်ဇာတွင် `http://localhost:8080` အလိုအလျောက်ပွင့်လာမည်ဖြစ်ပြီး PDF တင်၍ Live progress ကြည့်ကာ `.docx` ကို တိုက်ရိုက်ဒေါင်းလုဒ်ဆွဲနိုင်ပါသည်။

### (ဂ) 1-Click Batch Folder Tool (စာအုပ်အများကြီး တစ်ခါတည်းပြောင်းရန်)
- ပြောင်းလိုသည့် PDF စာအုပ်များအားလုံးကို `input/` folder ထဲ ထည့်ပါ။
- Windows: `run_batch.bat` | Mac/Linux: `./run_batch.sh`
- ပြီးစီးသည့် Word (`.docx`) ဖိုင်များအားလုံး `output/` folder ထဲသို့ အလိုအလျောက် ထွက်လာပါမည်။

## ၅။ GitHub Auto-Update စနစ် (Model ဗားရှင်းအသစ် ထွက်တိုင်း အလိုအလျောက် Update ယူခြင်း)
- Desktop App နှင့် Web UI ထဲရှိ **Check GitHub Update** ခလုတ်ကို နှိပ်ခြင်း (သို့မဟုတ် `python core/updater.py` run ခြင်း) ဖြင့် ဤ repository ၏ `version.json` ကို စစ်ဆေးကာ နောက်ဆုံးထွက် `.traineddata` ဗားရှင်းအသစ် (`v3`, `v4` စသည်) ကို `tessdata/` ထဲသို့ အလိုအလျောက် ဒေါင်းလုဒ်ဆွဲ၍ ပြောင်းသုံးပေးပါမည်။
- Batch Tool (`run_batch`) သည် run တိုင်း နောက်ဆုံးဗားရှင်း model ရှိမရှိ အလိုအလျောက် စစ်ဆေးပေးပါသည်။ (အော့ဖ်လိုင်းသုံးလိုပါက `python apps/batch_cli.py --no-update` ဖြင့် သုံးနိုင်ပါသည်။)

## ၆။ Command Line ဖြင့် တိုက်ရိုက်သုံးလိုပါက
```bash
tesseract page.png output --tessdata-dir ./tessdata -l mya_dhamma_v9_balanced --psm 6
```
အင်္ဂလိပ်စာပါ ရောနှောနေပါက `-l mya_dhamma_v2+eng` ဟု တွဲသုံးနိုင်ပါသည်။
