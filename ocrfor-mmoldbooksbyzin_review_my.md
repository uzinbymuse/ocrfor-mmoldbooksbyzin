# `ocrfor-mmoldbooksbyzin` သုံးသပ်ချက် — မြန်မာစာ OCR ကို ယုံကြည်စိတ်ချရအောင် တိုးတက်စေခြင်း

စစ်ဆေးသည့် repository snapshot သည် ၂၀၂၆-၁၁-၂၃ ရက် commit `b265c73` ဖြစ်သည်။ အဓိကတွေ့ရှိချက်မှာ ဒီ project သည် သပြေကန်ဓမ္မစာအုပ်ဟောင်းများ၊ WinInnwa/ခဲစာလုံးပုံစံများနှင့် ပါဠိစာသားအတွက် **ကောင်းမွန်သော domain-specific စတင်ချက်** ဖြစ်နေပြီဖြစ်သော်လည်း၊ ယခုဖော်ပြထားသည့် ၉၈.၂၀% ကို မြန်မာ OCR အားလုံးအတွက် တိကျမှုဟု မယူဆသင့်သေးပါ။ အရည်အသွေးကို အမှန်တကယ် မြှင့်ရန် အရေးအကြီးဆုံးက model ကိုချက်ချင်းပိုရှည် train လုပ်ခြင်းမဟုတ်ဘဲ၊ benchmark ကိုကျယ်ပြန့်စေခြင်း၊ cleanup အမှားကိုပြင်ခြင်းနှင့် OCR မူရင်းစာသားကို မပျက်စီးစေဘဲ ထိန်းသိမ်းခြင်းဖြစ်သည်။ [1] [2]

## လက်ရှိအားသာချက်များ

Project က မူရင်း `mya` model တွင် ဓမ္မစာအုပ်များအတွက် အရေးပါတဲ့ `ဣ၊ ဩ၊ ဪ၊ ၏၊ ၌၊ ၍၊ ။၊ ၊၊ ဈ၊ ဎ` စတဲ့ glyph များ မပါဝင်ကြောင်း တွေ့ရှိပြီး output class များကို ၁၄၈ မှ ၁၅၉ သို့ တိုးချဲ့ထားသည်။ ထို့နောက် pseudo-label များကို အခြေခံမထားဘဲ လူကစစ်ဆေးထားသည့်စာကြောင်းများနှင့် targeted synthetic data ကို အသုံးပြုပြီး `၊`/`။`, `ပ္ပ`/`ပွ`, မြန်မာဂဏန်းနှင့် ဆင်တူပုံရှိသော အက္ခရာများကို ရည်ရွယ်ချက်ရှိရှိ train လုပ်ထားခြင်းက project ၏ အားကောင်းဆုံးအချက်ဖြစ်သည်။ [1] [3]

စမ်းသပ်စာမျက်နှာ ၃ မျက်နှာအတွက် ရေးသားထားသောရလဒ်များမှာ ၁,၀၈၃ လုံးတွင် error ၇၂ ခု (၉၃.၃၅%)၊ ၁,၇၇၉ လုံးတွင် error ၃၂ ခု (၉၈.၂၀%)၊ ၁,၂၀၄ လုံးတွင် error ၄၀ ခု (၉၆.၆၈%) ဖြစ်သည်။ ဒီရလဒ်တွေက အမှားအမျိုးအစားအလိုက် အာရုံစိုက်ပြင်ဆင်ခြင်း အကျိုးရှိနိုင်ကြောင်း ပြသသော်လည်း၊ အဲဒီစာမျက်နှာများနှင့် ဆင်တူသည့်စာအုပ်၊ ပုံနှိပ်နည်း၊ scan အခြေအနေများအပြင် အခြားအခြေအနေများကိုပါ အာမခံမပေးပါ။ [2]

အသုံးပြုသူအတွက် Desktop, local web inspector နှင့် batch PDF→DOCX လမ်းကြောင်းများပါဝင်တာလည်း အားသာချက်ဖြစ်သည်။ သို့သော် UI ရှိ PSM ရွေးချယ်မှုသည် အဓိကအားဖြင့် web page inspector မှာသာ မြင်ရပြီး စာအုပ်တစ်အုပ်လုံးပြောင်းသည့် engine က အများအားဖြင့် `SINGLE_BLOCK`/PSM 6 ကို သုံးထားသည်။ ထို့ကြောင့် စာမျက်နှာ layout ကွဲပြားမှုကို OCR လမ်းကြောင်းက အပြည့်အဝ မကိုင်တွယ်သေးပါ။ [4]

## “၉၈.၂၀%” ကို ဘယ်လိုဖတ်သင့်သလဲ

၁,၇၇၉ လုံးအပေါ် error ၃၂ ခုဆိုသည့် ကိန်းဂဏန်းကို `1 − 32/1779` ဖြင့်တွက်လျှင် ၉၈.၂၀% ရသည်။ ထို့ကြောင့် ၎င်းသည် ထိုစာမျက်နှာအတွက် character-level error count ၏ ပြောင်းပြန်အနီးစပ်ဆုံး တိုင်းတာချက်ဖြစ်သည်။ သို့သော် error ကို Levenshtein edit distance ဖြင့်တွက်သလား၊ space/ပုဒ်ဖြတ်သင်္ကေတ/Unicode sequence မတူမှုများကို ဘယ်လိုကိုင်တွယ်သလဲ၊ အတိအကျမဖော်ပြထားပါ။ “accuracy” ကို ပြန်ထုတ်တွက်နိုင်ရန် benchmark script နှင့် ground-truth စာသားကိုပါ ထုတ်ပြန်သင့်သည်။ [2]

စာမျက်နှာသုံးမျက်နှာ၏ error/character စုစုပေါင်းကို တူညီသောနည်းဖြင့်တွက်ထားသည်ဟုယူဆလျှင် ၁၄၄/၄,၀၆၆ error ဖြစ်ပြီး စုစုပေါင်း character accuracy သည် ခန့်မှန်း ၉၆.၄၆% ဖြစ်သည်။ ဒါက project ၏ အောင်မြင်မှုကို လျှော့ပြောခြင်းမဟုတ်ပါ။ စာမျက်နှာတစ်မျက်နှာရဲ့အကောင်းဆုံးရလဒ်ကို corpus တစ်ခုလုံး၏ စွမ်းဆောင်ရည်အဖြစ် မရောထွေးစေရန်သာဖြစ်သည်။

ထို့အပြင် အချက်အလက်စာရွက်တွင် v9 training set ကို ၈၂၈ entries ဟုဆိုထားသော်လည်း ဖော်ပြထားသောအစိတ်အပိုင်းများဖြစ်သည့် လူစစ်ပြီးစာကြောင်း ၃၅၆၊ synthetic replay ၄၀ နှင့် balanced synthetic ၁၂၅ ကိုပေါင်းလျှင် ၅၂၁ သာဖြစ်သည်။ ကျန် entry ၃၀၇ ခု၏ ရင်းမြစ်ကို မဖော်ပြထားပါ။ v9 ကို train လုပ်ရာတွင် အသုံးပြုသော manifest၊ စာကြောင်းအရေအတွက်နှင့် train/eval ခွဲထားပုံကို မှတ်တမ်းတင်ပေးရန်လိုသည်။ ထုတ်ဝေထားသည့် generic training script တွင် `.lstmf` ဖိုင်များကို random shuffle လုပ်ပြီး ၉၀/၁၀ ဖြင့်ခွဲထားသည်။ ဒါကို v9 အတွက် အတိအကျအသုံးပြုခဲ့သည်ဟု မဆိုနိုင်သော်လည်း၊ စာမျက်နှာ/စာအုပ်အလိုက် holdout မလုပ်ပါက တူညီစာမျက်နှာမှ အလွန်ဆင်တူ line များ train နှင့် evaluation နှစ်ဖက်လုံးထဲဝင်ပြီး ရလဒ်က အမှန်ထက်မြင့်နိုင်သည်။ [3] [5]

## ချက်ချင်းပြင်သင့်သော software အန္တရာယ်

`core/ocr_engine.py` ၏ `clean_line()` တွင် `re.sub(r"\s+([။၊])", r"\x01", line)` ဟူသော replacement ပါသည်။ Python regex replacement တွင် `\x01` သည် ဖမ်းယူထားသော ပုဒ်ဖြတ်အမှတ်ကို ပြန်ထည့်ခြင်းမဟုတ်ဘဲ control character U+0001 ကို ထည့်ခြင်းဖြစ်သည်။ ဥပမာ `စာသား ။` သည် `စာသား` နောက်တွင် မမြင်ရသော control character ပါသည့် စာသားဖြစ်သွားနိုင်သည်။ ဒါကို `r"\g<1>"` ဖြင့်ပြင်နိုင်သော်လည်း၊ test ထည့်ပြီး DOCX နှင့် plain-text output ထဲတွင် C0 control character မရှိကြောင်း စစ်ဆေးရမည်။ [4]

အခြား `WORD_FIXES` များကလည်း စကားလုံးတချို့ကို pattern မရွေးဘဲ အစားထိုးခြင်း၊ codepoint အချို့ကို ဖယ်ရှားခြင်း၊ `|` သို့မဟုတ် bracket ပုံစံများကို ပုဒ်ဖြတ်အဖြစ် ပြောင်းခြင်းတို့ လုပ်သည်။ ပုံမှန်စာသားမှာ အမှားတချို့ကို ပြင်ပေးနိုင်သော်လည်း ပါဠိအမည်၊ ရှေးစာလုံးပေါင်း၊ ကိုးကားချက် သို့မဟုတ် OCR ကတကယ်ဖတ်ထားသည့် ပုံစံကို မတော်တဆ ဖျက်နိုင်သည်။ Default လမ်းကြောင်းမှာ OCR raw text ကို အမြဲသိမ်းပြီး၊ ပြင်ဆင်ချက်ကို သီးခြား “အကြံပြုချက်” အဖြစ်ပြသသင့်သည်။ လူကအတည်ပြုမှသာ canonical text ကိုပြောင်းပါ။ [4]

PDF engine သည် FitZ ရှိလျှင် ၁၈၀ DPI ဖြင့် render လုပ်ပြီး `crop_hf=True` ဖြင့် စာမျက်နှာအပေါ်/အောက်ပိုင်းကို ဖြတ်တောက်သည်။ Margins ထဲတွင် စာသားရှိသောစာမျက်နှာ၊ ခေါင်းစီး/အောက်ခြေမှတ်စုပါစာမျက်နှာ၊ နှစ်ကော်လံ၊ ဂါထာပုံစံ စာမျက်နှာများတွင် ဖြတ်တောက်ခြင်းနှင့် PSM တစ်မျိုးတည်းသုံးခြင်းက စာကြောင်းပျောက်ခြင်း သို့မဟုတ် အစီအစဉ်မှားခြင်း ဖြစ်စေနိုင်သည်။ Scan အရည်အသွေးနှင့် layout ကိုအခြေခံ၍ DPI, deskew, crop, column detection နှင့် PSM ကိုရွေးနိုင်စေရန်လုပ်ပါ။ Confidence နိမ့်သော line များကို ပုံ crop နဲ့အတူ user ထံပြန်ပြသပါ။ [4]

## အရည်အသွေးမြှင့်တင်ရေးကို ဦးစားပေးလုပ်ရန်

### ၁။ Benchmark ကို စံတစ်ခုဖြစ်အောင်လုပ်ပါ

စာကြောင်း random split မဟုတ်ဘဲ **စာမျက်နှာလုံးဝမတူ၊ စာအုပ်/ထုတ်ဝေမှုပုံစံလုံးဝမတူ** သည့်အုပ်စုကို final test set အဖြစ် အရင်ပိတ်ထားပါ။ Training, tuning နှင့် final test အဆင့်များကို page/image group အလိုက်ခွဲခြင်းဖြင့် data leakage ဖြစ်နိုင်ခြေကို လျှော့ချနိုင်သည်။ အနည်းဆုံး ပုံနှိပ်ပုံစံအဟောင်း အမျိုးမျိုး၊ မှုန်ဝါး/အစွန်းအထင်းပါ scan၊ ကော်လံတစ်ခုနှင့် အများပါသည့်စာမျက်နှာ၊ ရှင်းလင်းချက်စာသားနှင့် ဂါထာ၊ ပါဠိနှင့် မြန်မာစာရောနှောသည့်ပုံစံများကို စမ်းသပ်ပါ။ Test set ကို model ရွေးပြီးနောက် ထပ်ခါတလဲလဲကြည့်ကာ tune မလုပ်ပါနှင့်။

Metric အနေဖြင့် page/line အားလုံးအပေါ် edit distance ဖြင့် **CER = (substitution + deletion + insertion)/ground-truth character count** ကို အဓိကဖော်ပြပါ။ CER အပြင် စာမျက်နှာတစ်မျက်နှာချင်းစီရလဒ်၊ median နှင့် အဆိုးဆုံးစာမျက်နှာ၊ အရေးကြီးရှားပါး glyph များ၏ precision/recall၊ blank/missing-page နှုန်း၊ line order မှန်ကန်မှုတို့ကို ခွဲဖော်ပြပါ။ Myanmar စာမှာ space အခြေပြု “word” ခွဲခြားပုံက corpus တစ်ခုနဲ့တစ်ခုကွဲနိုင်လို့ WER ကို ထပ်ဆောင်း metric အဖြစ်သာ သုံးတာ ပိုသင့်တော်သည်။ `mya` baseline, v9 model, post-correction on/off တို့ကို တူညီသော scan, DPI, PSM နှင့် preprocessing ဖြင့်ယှဉ်ပါ။ Tesseract training docs တွင်လည်း evaluation list ကို training data မှ သီးခြားပေးနိုင်သည်ဟု ဖော်ပြထားသည်။ [7]

### ၂။ Data ကို အရေအတွက်ထက် အမျိုးအစားကွဲပြားမှုအတွက် တိုးပါ

လက်ရှိ human-verified lines သည် စာမျက်နှာအပြည့် benchmark များစွာအတွက် နည်းသေးသည်။ တိုးချဲ့ရာတွင် လူကအတည်ပြုထားသော real scan များကို အဓိကထားပြီး အသုံးပြုထားသည့်စာအုပ်၊ စာမျက်နှာ၊ စာကြောင်း၊ ပုံနှိပ်ပုံစံ၊ ground-truth Unicode နှင့် စစ်ဆေးသူအမည်တို့ကို metadata ထဲတွင် သိမ်းထားပါ။ Synthetic text သည် `၊`/`။` ကဲ့သို့ ရှားပါးသင်္ကေတများနှင့် `ပ္ပ` ကဲ့သို့ ပုံစံဆင်တူ၍ ရောထွေးနိုင်သည့် အတွဲများကို ဖြည့်စွက်ရာတွင် အသုံးဝင်သည်။ သို့သော် အမှန်တကယ် ရှေးပုံနှိပ်စာလုံးပုံစံ၊ မှင်ပျံ့နှံ့မှု၊ စက္ကူအစွန်းအထင်းနှင့် စာလုံးဟောင်းများ၏ ပုံစံကို အစားမထိုးနိုင်ပါ။

`myOCR` တွင် ဖောင့် ၁၄ မျိုးဖြင့် ဖန်တီးထားသော synthetic line ပုံ ၂၅,၇၉၀ ခု ပါဝင်ပြီး OCR နှင့် post-OCR correction ကို လေ့လာထားသည်။ Hugging Face ပေါ်က Burmese OCR dataset တွင် လူစစ်ထားသော line ပုံ ၉,၀၆၅ ခုနှင့် မစစ်ရသေးသော crop ၁၆၂,၈၆၃ ခု ရှိသည်ဟု dataset card ကဆိုသော်လည်း၊ အရင်းအမြစ်မှာ အစိုးရ/NGO စာရွက်စာတမ်းများဖြစ်၍ ဓမ္မစာအုပ်ဟောင်းနှင့် ဖောင့်၊ စာမျက်နှာပုံစံ မတူနိုင်ပါ။ ထို့ကြောင့် မျက်စိမှိတ်ပေါင်းထည့်မည့်အစား label များကို နမူနာစစ်ပြီး သက်ဆိုင်ရာ set အဖြစ် သီးခြားစမ်းသပ်ပါ။ Dataset card က မြန်မာ virama ကိုင်တွယ်ပုံကြောင့် Tesseract 5.4.1 တွင် `tesstrain` အစား `lstmtraining` ကို တိုက်ရိုက်သုံးရန် အကြံပြုထားသည်။ ဒီ project ၏ script ကလည်း `lstmtraining` ကို တိုက်ရိုက်သုံးထားသော်လည်း virama/conjunct regression test ကို မဖြစ်မနေထည့်ပါ။ [6] [8] [9]

### ၃။ Post-OCR ပြင်ဆင်မှုကို အထောက်အကူအဖြစ်သာထားပါ

`myOCR` သုတေသနက synthetic corpus ပေါ်တွင် Transformer correction ထည့်ပြီး WER သိသိသာသာလျှော့နိုင်ကြောင်း report လုပ်ထားသည်။ ဒါက post-correction ကို စမ်းသပ်သင့်ကြောင်းပြသော်လည်း ထိုရလဒ်သည် ဒီ project ၏ ရှေးဓမ္မစာအုပ်များပေါ်တွင် အလိုအလျောက်ရမည်ဟု မဆိုလိုပါ။ Domain lexicon, n-gram သို့မဟုတ် neural corrector တစ်ခုခုကို ထည့်လျှင် raw OCR, suggested correction, final human-approved text ကို ခွဲသိမ်းပြီး test set ပေါ်မှာ အပြောင်းအလဲကြောင့် မူရင်းမှန်စာ ပျက်သွားသည့်နှုန်းကိုပါ တိုင်းပါ။ [9]

Myanmar Unicode သည် Zawgyi font မဟုတ်ဘဲ codepoint အစီအစဉ်ပါသော encoding ဖြစ်ပြီး စာလုံးများကို အစီအစဉ်မှားသိမ်းလျှင် rendering နှင့် text-processing ထိခိုက်နိုင်သည်။ OCR output အတွက် Unicode validation, control-character check, malformed sequence warning တို့ထည့်ပါ။ သို့သော် Unicode normalization ကို “အကုန် NFC လုပ်လိုက်” ဆိုသည့် blanket rule ဖြင့်မလုပ်ဘဲ ground-truth စာကြောင်းများနှင့် codepoint အဆင့်မှာစမ်းသပ်ပါ။ Source PDF ထဲမှာ Zawgyi encoded text ဖြစ်နိုင်သည့်ကိစ္စကိုလည်း အလိုအလျောက်ပြောင်းမချမီ detect လုပ်ပြီး user ကို သတိပေးသင့်သည်။ Google `myanmar-tools` တွင် Zawgyi detection/conversion ရှိသော်လည်း detection threshold ကို ကိုယ့်စာသားဒေတာနှင့်ကိုက်အောင် စစ်ဆေးရန် သူတို့ကလည်း အကြံပြုထားသည်။ [10] [11]

### ၄။ Release ကို စိတ်ချရပြီး ပြန်ထုတ်လို့ရအောင်လုပ်ပါ

`version.json` မှာ model filename, URL, version တို့ကိုဖတ်ပြီး model ကို download လုပ်သော်လည်း digest/checksum အတည်ပြုခြင်းကို updater ထဲမှာ မတွေ့ရပါ။ Model file အတွက် SHA-256, file size, Tesseract version, training-data manifest, benchmark script နှင့် changelog ကို release တိုင်းတွင်ထည့်ပါ။ Download ကို `.part` ဖြင့်ယူပြီး checksum အောင်မှ active model ကိုအစားထိုးပါ။ [12]

အတူတကွ unit tests ထည့်ရန်လည်းလိုသည်။ အနည်းဆုံး punctuation နောက် whitespace, C0 control character, Myanmar virama/kinzi, ရှားပါးစာလုံး ၁၀ လုံး, Zawgyi-like input, OCR blank page, DOCX export ကိုစမ်းပါ။ `requirements.txt` ထဲရှိ dependency များနှင့် app က မဖြစ်မနေလိုအပ်သည့် Tesseract/PyMuPDF/tesserocr ကဲ့သို့ external dependency များကို version/installation ခွဲခြားပြီး smoke test လုပ်စေပါ။

## အကြံပြုဦးစားပေးအစီအစဉ်

**ပထမဆုံး** U+0001 cleanup bug ကိုပြင်၊ raw OCR ကို မပြောင်းဘဲသိမ်း၊ automatic word-fix များကို suggestion mode သို့ပြောင်းပြီး unit tests ထည့်ပါ။ **နောက်တစ်ဆင့်** train/test data manifest ထုတ်ပြီး စာမျက်နှာ/စာအုပ်အလိုက် leakage မရှိသော frozen benchmark နှင့် CER script ထုတ်ပြန်ပါ။ **ထို့နောက်** မတူညီသောအဟောင်းစာအုပ်များမှ human-verified full-page data ကို စုဆောင်းကာ v9 နှင့် baseline တို့ကို တူညီသော setting ဖြင့်ပြန်တိုင်းပါ။ ထိုအဆင့်များပြီးမှ preprocessing, layout routing နှင့် correction model ကို ထပ်တိုးပါ။

ဒီအစီအစဉ်က model ကို လုံးဝပြန်ရေးရန် မလိုဘဲ၊ ရှိပြီးသားတန်ဖိုးကို ထိန်းသိမ်းကာ OCR ကို စစ်ဆေးနိုင်၊ အမှားကိုပြန်ရှာနိုင်၊ တခြားစာအုပ်များပေါ်တွင်လည်း ယုံကြည်ရအောင် တိုးတက်စေမည်။

## ကိုးကားရင်းမြစ်များ

[1]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin "ocrfor-mmoldbooksbyzin repository and README"
[2]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin/blob/main/BENCHMARK_v9_balanced.md "v9 balanced benchmark"
[3]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin/blob/main/TRAINING_LOG.md "OCR model training log"
[4]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin/blob/main/core/ocr_engine.py "OCR engine and cleanup code"
[5]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin/blob/main/scripts/train_mya_dhamma.sh "Tesseract training script"
[6]: https://huggingface.co/datasets/alexbeatson/burmese_ocr_data "Burmese OCR data dataset card"
[7]: https://tesseract-ocr.github.io/tessdoc/tess5/TrainingTesseract-5.html "Tesseract 5 training documentation"
[8]: https://huggingface.co/datasets/alexbeatson/burmese_ocr_data "Tesseract training notes in the Burmese OCR dataset card"
[9]: https://github.com/ye-kyaw-thu/myOCR "myOCR corpus and post-OCR correction research"
[10]: https://www.unicode.org/faq/myanmar.html "Unicode Consortium Myanmar FAQ"
[11]: https://github.com/google/myanmar-tools "Google Myanmar tools for Zawgyi detection and conversion"
[12]: https://github.com/uzinbymuse/ocrfor-mmoldbooksbyzin/blob/main/core/updater.py "Model updater implementation"
