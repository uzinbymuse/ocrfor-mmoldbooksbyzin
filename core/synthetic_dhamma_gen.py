#!/usr/bin/env python3
"""Degraded Lead-Type Synthetic Dhamma & Pali Line Generator for Tesseract 5 LSTM.

Generates realistic old Myanmar printed book lines with:
- Ink bleed-through (ghost reverse text on paper)
- Gaussian blur (sigma = 0.4 ~ 0.8)
- Random speckle noise (paper dust)
- Slight shear / slant (-1.5 deg to +1.5 deg)
- Variable stroke thickness (morphological ink pressure variation)
- Strict >=12% vertical padding top & bottom to protect stacked conjuncts & upper vowels
"""
import os
import math
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

try:
    from core.myanmar_normalizer import normalize_myanmar_unicode, validate_myanmar_line
except ImportError:
    from myanmar_normalizer import normalize_myanmar_unicode, validate_myanmar_line


# Curated classical Dhamma, Pali, Kaccayana, Medini & Rupasiddhi seed lines
# Deliberately rich in ဣ ဩ ဪ ၏ ၌ ၍ ။ ၊ ဈ ဎ, stacked conjuncts (္ဓ ္ပ ္ဌ င်္ ဏ္ဍ ဿ), and Myanmar digits ၀–၉
CLASSICAL_SEED_LINES = [
    # Kaccayana Sadda (ကစ္စည်းသဒ္ဒါ)
    "၁။ အတ္ထော အက္ခရသညာတော၊ အက္ခရာပါဒါယော ဧကစတ္တာလီသံ။",
    "၂။ တတ္ထောဒန္တာ သရာ အဋ္ဌ၊ လောပဉ္စ တတြာကာရော။",
    "၃။ သရလောပေါ မာဒေသပစ္စယာဒိမှိ သရလောပေ တု ပကတိ။",
    "၄။ ကွစိ အသဝဏ္ဏံ လုတ္တေ၊ ဒီဃံ၊ ပုဗ္ဗော စ။",
    "၅။ ယမေဒန္တဿာဒေသော၊ ဝမောဒုဒန္တာနံ၊ သဗ္ဗော စံတိ။",
    "၆။ နိဂ္ဂဟီတဉ္စ၊ ဝဂ္ဂန္တံ ဝါ ဝဂ္ဂေ၊ ဧဟေညံ၊ သယေ စ။",
    "၇။ ဇိနဝစနယုတ္တံ ဟိ၊ လိင်္ဂဉ္စ နိပ္ပဇ္ဇတေ၊ သဒ္ဒတ္ထနယေန။",
    "၈။ ပဌမာ၊ ဒုတိယာ၊ တတိယာ၊ စတုတ္ထီ၊ ပဉ္စမီ၊ ဆဋ္ဌီ၊ သတ္တမီ ဝိဘတ္တိယော။",
    "၉။ ဩကာသေ သတ္တမီ၊ ကမ္မတ္ထေ ဒုတိယာ၊ ကရဏေ တတိယာ ဟောတိ။",
    "၁၀။ သမ္ပဒါနေ စတုတ္ထီ၊ အပါဒါနေ ပဉ္စမီ၊ သာမိသ္မိံ ဆဋ္ဌီ။",
    # Rupasiddhi & Medini (ရူပသိဒ္ဓိ နှင့် မေဒိနီ)
    "နာမာနံ သမာသော ယုတ္တတ္ထော၊ အမာဒယော ပရပဒေဘိ ဗဟုဗ္ဗီဟိ။",
    "ကမ္မဓာရယသမာသ်၊ ဒိဂုသမာသ်၊ တပ္ပုရိသသမာသ်၊ ဒွန္ဒသမာသ်၊ အဗျယီဘာဝသမာသ်။",
    "အညပဒတ္ထေသု ဗဟုဗ္ဗီဟိ၊ တုလျာဓိကရဏေ ဗဟုဗ္ဗီဟိသမာသ် ဖြစ်၏။",
    "ဓာတွတ္ထေ ဝတ္တမာနာ၊ ဘဝိဿန္တိ၊ အဇ္ဇတနီ၊ ပရောက္ခာ၊ ကာလာတိပတ္တိ ဝိဘတ်များ။",
    "ကတ္တရိ ပရဿပဒံ၊ အတ္တနောပဒါနိ ဘာဝေ စ ကမ္မနိ ဟူ၍ ရူပသိဒ္ဓိကျမ်း၌ ဆို၏။",
    "မေဒိနီကျမ်း၌ ပဋ္ဌာန်းပစ္စည်း ၂၄ ပါး၏ အနက်အဓိပ္ပာယ်ကို အကျယ်ဖွင့်ဆိုထား၏။",
    "ဟေတုပစ္စယော၊ အာရမ္မဏပစ္စယော၊ အဓိပတိပစ္စယော၊ အနန္တရပစ္စယော။",
    "သမနန္တရပစ္စယော၊ သဟဇာတပစ္စယော၊ အညမညပစ္စယော၊ နိဿယပစ္စယော။",
    "ဥပနိဿယပစ္စယော၊ ပုရေဇာတပစ္စယော၊ ပစ္ဆာဇာတပစ္စယော၊ အာသေဝနပစ္စယော။",
    "ကမ္မပစ္စယော၊ ဝိပါကပစ္စယော၊ အာဟာရပစ္စယော၊ ဣန္ဒြိယပစ္စယော။",
    "ဈာနပစ္စယော၊ မဂ္ဂပစ္စယော၊ သမ္ပယုတ္တပစ္စယော၊ ဝိပ္ပယုတ္တပစ္စယော။",
    "အတ္ထိပစ္စယော၊ နတ္ထိပစ္စယော၊ ဝိဂတပစ္စယော၊ အဝိဂတပစ္စယောတိ။",
    # Thapyaykan & Pali Nissaya (သပြေကန် နှင့် ပါဠိနိဿယ)
    "ဣတိပိ သော ဘဂဝါ အရဟံ၊ သမ္မာသမ္ဗုဒ္ဓေါ၊ ဝိဇ္ဇာစရဏသမ္ပန္နော၊ သုဂတော။",
    "လောကဝိဒူ၊ အနုတ္တရော ပုရိသဒမ္မသာရထိ၊ သတ္ထာ ဒေဝမနုဿာနံ၊ ဗုဒ္ဓေါ၊ ဘဂဝါ။",
    "သဗ္ဗေ သတ္တာ၊ သဗ္ဗေ ပါဏာ၊ သဗ္ဗေ ဘူတာ၊ သဗ္ဗေ ပုဂ္ဂလာ၊ သဗ္ဗေ ဣတ္ထိယော။",
    "အဝေရာ ဟောန္တု၊ အဗျာပဇ္ဇာ ဟောန္တု၊ အနီဃာ ဟောန္တု၊ သုခီ အတ္တာနံ ပရိဟရန္တု။",
    "ဣဒ္ဓိပါဒ်လေးပါး၊ ဣန္ဒြေငါးပါး၊ ဗိုလ်ငါးပါး၊ ဗောဇ္ဈင်ခုနစ်ပါး၊ မဂ္ဂင်ရှစ်ပါး။",
    "ဩဝါဒပါတိမောက်တရားတော်သည် ဘုရားရှင်တိုင်း၏ အဆုံးအမတော် ဖြစ်ပါပေ၏။",
    "ဩပနေယျိကဂုဏ်တော်၊ ပစ္စတ္တံ ဝေဒိတဗ္ဗဂုဏ်တော်၊ ဧဟိပဿိကဂုဏ်တော်။",
    "ဪ ဖြစ်ရလေခြင်း၊ ဪ သံသရာဘေးသည် ကြောက်မက်ဖွယ်ရာ ကောင်းလှပေ၏။",
    "ဩဃလေးဖြာ သံသရာမှ လွတ်မြောက်စေရန် သတိပဋ္ဌာန်တရားကို ပွားများအပ်၏။",
    "ဈာန်၊ မဂ်၊ ဖိုလ်၊ နိဗ္ဗာန်ချမ်းသာကို မျက်မှောက်ပြု၍ အေးငြိမ်းကြပါစေ။",
    "ဝုဍ္ဎိပ္ပတ္တော၊ အဘိဝုဍ္ဎိ၊ ဒဠှံ၊ ဂာဠှံ၊ မဇ္ဈိမပဋိပဒါ အကျင့်မြတ်၌ တည်ကြကုန်၏။",
    "သပြေကန်ဆရာတော်ဘုရားကြီး အရှင်ဝါသေဋ္ဌာဘိဝံသ၏ တရားပေါင်းချုပ် အမှတ် (၁) နှင့် (၂)။",
    "ရဟန်းတော်များ၊ ဓမ္မကထိကဆရာတော်များ၊ ဒကာ၊ ဒကာမများအား ဟောကြားတော်မူခဲ့၏။",
    "သာသနာသက္ကရာဇ် ၂၅၄၆၊ ကောဇာသက္ကရာဇ် ၁၃၆၄ ခုနှစ်၊ ခရစ်သက္ကရာဇ် ၂၀၀၃ ခုနှစ်။",
    "အခန်း (၇)၊ အပိုဒ် (၈)၊ စာမျက်နှာ (၉) မှ စာမျက်နှာ (၇၈၉) အထိ လေ့လာဖတ်ရှုနိုင်၏။",
    "၇။ သတ္ထာဒေဝမနုဿာနံ၊ ၈။ ဗုဒ္ဓေါ၊ ၉။ ဘဂဝါ ဟူသော ဘုရားဂုဏ်တော် ကိုးပါး။",
    "၀၊ ၁၊ ၂၊ ၃၊ ၄၊ ၅၊ ၆၊ ၇၊ ၈၊ ၉ ဟူသော မြန်မာဂဏန်း ဆယ်လုံးကို မှန်ကန်စွာ ဖတ်ရမည်။",
    "သတ္တမီဝိဘတ် (၇)၊ အဋ္ဌမအခန်း (၈)၊ နဝမအကြိမ် (၉)၊ ဒသမပုဒ် (၁၀) တို့၌ တွေ့ရ၏။",
    "ပိဋကတ်သုံးပုံ၊ ဝိနည်းငါးကျမ်း၊ သုတ္တန်နိကာယ်ငါးရပ်၊ အဘိဓမ္မာခုနစ်ကျမ်း။",
    "အဋ္ဌကထာ၊ ဋီကာ၊ အနုဋီကာ၊ ယောဇနာ၊ ဂဏ္ဌိကျမ်းများကို ကြည့်ရှု၍ ရေးသားအပ်၏။",
    "သီလ၊ သမာဓိ၊ ပညာ သိက္ခာသုံးပါးကို ဖြည့်ကျင့်၍ နိဗ္ဗာန်သို့ မျက်မှောက်ပြုအပ်၏။",
    "ဒါန၊ သီလ၊ ဘာဝနာ ကုသိုလ်ကောင်းမှုတို့ကို မမေ့မလျော့ ပြုလုပ်အားထုတ်ကြကုန်လော့။",
    "အနိစ္စ၊ ဒုက္ခ၊ အနတ္တ လက္ခဏာရေးသုံးပါးကို ဝိပဿနာဉာဏ်ဖြင့် ရှုမှတ်ပွားများကြ၏။",
    "ပုဂ္ဂိုလ်တစ်ဦးတစ်ယောက်၏ အကြောင်းကို ရေးသားရာ၌ အမှန်နှင့်နီးစပ်အောင် စိစစ်ရ၏။",
    "ဣမံ ပုညဘာဂံ သဗ္ဗသတ္တာနံ ဘာဇေမ၊ သာဓု၊ သာဓု၊ သာဓု ဟု ခေါ်ဆိုကြကုန်၏။",
    "ဣဓ လောကေ၊ ပရလောကေ ဟူ၍ နှစ်ဘဝလုံး ချမ်းသာစေရန် ဆုံးမဟောကြားတော်မူ၏။",
    "သိင်္ဂါလောဝါဒသုတ်တော်နှင့်အညီ လူ့ဘောင်ယဉ်ကျေးအောင် သွန်သင်ဆုံးမတော်မူကြ၏။",
    "ပရိတ်တရားတော်မြတ်ကို ပါဠိမပျက်၊ အနက်ကိုသိ၊ မေတ္တာကရုဏာရှိစွာ ရွတ်ပွားသရဇ္ဈာယ်ကြကုန်စို့။",
]

# Phrase building blocks for combinatorial expansion (ensuring rich variety of ၊ vs ။, ဣ, ၇/၈/၉, conjuncts)
SUBJECTS = [
    "ကျေးဇူးတော်ရှင် သပြေကန်ဆရာတော်ဘုရားကြီးသည်",
    "ရွှေကျင်နိကာယ ဥပဥက္ကဋ္ဌ ဆရာတော်ကြီးသည်",
    "ကျောင်းထိုင် ပဓာနနာယက ဆရာတော်များ၊ ဓမ္မကထိကဆရာတော်များသည်",
    "ပါဠိပညာရှင် ရဟန်းတော်များနှင့် သာသနာပြု ပုဂ္ဂိုလ်များသည်",
    "ကစ္စည်းသဒ္ဒါကျမ်း၊ ရူပသိဒ္ဓိကျမ်းနှင့် မေဒိနီကျမ်းတို့၌",
    "ဗုဒ္ဓဘာသာဝင် ဒကာ၊ ဒကာမ သူတော်စင်အပေါင်းတို့သည်",
    "ဝိပဿနာ ယောဂီပုဂ္ဂိုလ်အပေါင်းတို့သည်",
    "သာသနာတော်ကို ကြည်ညိုလေးစားကြကုန်သော အမျိုးကောင်းသား၊ အမျိုးကောင်းသမီးတို့သည်",
]
MIDDLES = [
    "ဒါန၊ သီလ၊ ဘာဝနာ ကုသိုလ်ကောင်းမှုတို့ကို မပြတ်ပွားများ၍",
    "သီလ၊ သမာဓိ၊ ပညာ သိက္ခာသုံးပါးကို ကြိုးစားဖြည့်ကျင့်၍",
    "ပိဋကပါဠိတော်၊ အဋ္ဌကထာ၊ ဋီကာကျမ်းများကို ကြည့်ရှု၍",
    "ဣတိပိ သော ဘဂဝါ ဟူသော ဘုရားဂုဏ်တော် ၉ ပါးကို ရွတ်ပွား၍",
    "သတိပဋ္ဌာန် ၄ ပါး၊ ဣဒ္ဓိပါဒ် ၄ ပါး၊ ဗောဇ္ဈင် ၇ ပါး၊ မဂ္ဂင် ၈ ပါးတို့၌ တည်၍",
    "ဩဝါဒပါတိမောက်တရားတော်နှင့် ဩပနေယျိကဂုဏ်တော်တို့ကို နှလုံးသွင်း၍",
    "အနိစ္စ၊ ဒုက္ခ၊ အနတ္တ လက္ခဏာရေး ၃ ပါးကို ဉာဏ်ဖြင့် ရှုမှတ်၍",
    "ဈာန်၊ မဂ်၊ ဖိုလ်၊ နိဗ္ဗာန်ချမ်းသာကို ရည်မှန်းတောင့်တ၍",
    "အဘိဝုဍ္ဎိ မင်္ဂလာတရား ၃၈ ဖြာနှင့် သိင်္ဂါလောဝါဒသုတ်တော်အတိုင်း ကျင့်ကြံ၍",
    "စာမျက်နှာ ၇၈၊ ၈၉၊ ၉၇ နှင့် အခန်း ၇၊ ၈၊ ၉ တို့၌ လေ့လာမှတ်သား၍",
]
ENDINGS = [
    "သံသရာဝဋ်ဒုက္ခမှ လွတ်မြောက်စေရန် ကြိုးစားအားထုတ်ကြကုန်၏။",
    "လောကလူ့ဘောင် ယဉ်ကျေးအောင် ဟောကြားဆုံးမတော်မူခဲ့ပါပေ၏။",
    "အေးချမ်းသာယာသော နိဗ္ဗာန်ချမ်းသာသို့ မျက်မှောက်ပြုနိုင်ကြပါစေသတည်း။",
    "ပစ္စုပ္ပန်၊ တမလွန် နှစ်တန်သော အကျိုးစီးပွားကို ရရှိခံစားကြရကုန်၏။",
    "မိမိအကျိုး၊ အများအကျိုး၊ သာသနာ့အကျိုးကို သည်ပိုးဆောင်ရွက်ကြကုန်၏။",
    "ဩဃလေးဖြာ သံသရာမှ ကူးမြောက်နိုင်ကြောင်း ဟောပြတော်မူခဲ့သည်။",
    "အမှားကင်းစင်၍ ရှင်းလင်းလွယ်ကူစွာ နားလည်သဘောပေါက်နိုင်ကြ၏။",
    "ဘုရား၊ တရား၊ သံဃာ ရတနာသုံးပါး၌ သဒ္ဓါတရား ခိုင်မြဲကြပါစေ။",
]


def discover_myanmar_fonts():
    """Find installed Myanmar TrueType/OpenType fonts across Linux, Windows, and macOS."""
    preferred_patterns = [
        "*NotoSerifMyanmar*.ttf",
        "*Pyidaungsu*.ttf",
        "*Padauk*.ttf",
        "*NotoSansMyanmar*.ttf",
    ]
    search_dirs = [
        "/usr/share/fonts",
        os.path.expanduser("~/.local/share/fonts"),
        os.path.expanduser("~/Library/Fonts"),
        "/Library/Fonts",
        "C:/Windows/Fonts",
    ]
    found = []
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for pat in preferred_patterns:
            matches = glob.glob(os.path.join(d, "**", pat), recursive=True)
            for m in sorted(matches):
                if m not in found:
                    found.append(m)
    return found


def build_synthetic_text_pool(target_count: int = 5000, extra_corpus_texts=None, seed: int = 1101):
    """Build a diverse, 100% UTN #11 normalized and validated pool of Dhamma/Pali lines."""
    rng = random.Random(seed)
    pool = []
    seen = set()

    def try_add(txt: str):
        norm = normalize_myanmar_unicode(txt)
        ok = 10 <= len(norm) <= 120 and len(validate_myanmar_line(norm)) == 0
        if ok and norm not in seen:
            seen.add(norm)
            pool.append(norm)

    # 1. Seed classical Pali / Kaccayana / Medini / Rupasiddhi / Thapyaykan lines
    for s in CLASSICAL_SEED_LINES:
        try_add(s)

    # 2. Clean validated lines from extra corpus texts (if provided)
    if extra_corpus_texts:
        shuffled = list(extra_corpus_texts)
        rng.shuffle(shuffled)
        for line in shuffled:
            try_add(line)
            if len(pool) >= int(target_count * 0.72):
                break

    # 3. Combinatorial Dhamma/Pali + numbered list variations to reach target_count
    num_idx = 1
    combos = [(s, m, e) for s in SUBJECTS for m in MIDDLES for e in ENDINGS]
    rng.shuffle(combos)
    for s, m, e in combos:
        if len(pool) >= target_count:
            break
        # Format 1: standard prose with ၊ and ။
        try_add(f"{s} {m} {e}")
        if len(pool) >= target_count:
            break
        # Format 2: numbered item with Myanmar digits (heavy weight on ၇, ၈, ၉)
        d = rng.choice(["၁", "၂", "၃", "၄", "၅", "၆", "၇", "၈", "၉", "၁၇", "၂၈", "၃၉", "၇၈", "၈၉", "၉၇"])
        try_add(f"{d}။ {m} {e}")
        num_idx += 1

    # 4. Guarantee reaching target_count via varied Dhamma/Pali clause & citation patterns
    mya_map = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")
    prefixes = [
        "စာမျက်နှာ", "အခန်း", "အပိုဒ်", "သုတ်စဉ်", "ဂါထာအမှတ်", "နိဿယအပိုဒ်",
    ]
    attempts = 0
    seq_n = 1
    while len(pool) < target_count and attempts < target_count * 6:
        attempts += 1
        s1 = rng.choice(CLASSICAL_SEED_LINES)
        m_num = str(seq_n).translate(mya_map)
        pref = rng.choice(prefixes)
        punc = rng.choice(["၊", "။"])
        cand = f"{pref} {m_num}။ {s1.rstrip('။၊')}{punc}"
        norm = normalize_myanmar_unicode(cand)
        ok = 10 <= len(norm) <= 140 and len(validate_myanmar_line(norm)) == 0
        seq_n += 1
        if ok and norm not in seen:
            seen.add(norm)
            pool.append(norm)

    return pool[:target_count]


def apply_lead_type_degradations(gray: np.ndarray, rng: random.Random, ghost_gray: np.ndarray = None) -> np.ndarray:
    """Apply realistic old Myanmar lead-type print degradations to a grayscale line image."""
    h, w = gray.shape
    img = gray.astype(np.float32)

    # 1. Variable stroke thickness (erode = thicker ink, dilate = thinner/broken ink)
    mode = rng.choice(["thick", "thin", "normal", "normal"])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    if mode == "thick":
        img = cv2.erode(img, kernel, iterations=1)
    elif mode == "thin":
        img = cv2.dilate(img, kernel, iterations=1)

    # 2. Slight shear / slant (-1.5 deg to +1.5 deg)
    angle_deg = rng.uniform(-1.5, 1.5)
    shear = math.tan(math.radians(angle_deg))
    M = np.float32([[1.0, shear, -shear * h / 2.0], [0.0, 1.0, 0.0]])
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255.0)

    # 3. Ink bleed-through (faint reversed ghost text from back of page)
    if ghost_gray is not None and rng.random() < 0.75:
        gh, gw = ghost_gray.shape
        ghost = cv2.flip(ghost_gray, 1)  # mirror horizontal as reverse side
        ghost = cv2.resize(ghost, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
        dx = rng.randint(-12, 12)
        dy = rng.randint(-6, 6)
        M_shift = np.float32([[1, 0, dx], [0, 1, dy]])
        ghost = cv2.warpAffine(ghost, M_shift, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=255.0)
        ghost = cv2.GaussianBlur(ghost, (5, 5), 1.2)
        bleed_alpha = rng.uniform(0.04, 0.10)
        img = np.minimum(img, 255.0 - (255.0 - ghost) * bleed_alpha)

    # 4. Slight paper tone + Gaussian blur (sigma = 0.4 ~ 0.8)
    paper_bg = rng.uniform(242.0, 254.0)
    img = np.clip(img * (paper_bg / 255.0), 0, 255)
    sigma = rng.uniform(0.4, 0.8)
    img = cv2.GaussianBlur(img, (3, 3), sigmaX=sigma, sigmaY=sigma)

    # 5. Random speckle noise (paper dust + light ink grain)
    noise_density = rng.uniform(0.0005, 0.0022)
    num_speckles = int(h * w * noise_density)
    if num_speckles > 0:
        ys = np.random.randint(0, h, size=num_speckles)
        xs = np.random.randint(0, w, size=num_speckles)
        vals = np.random.uniform(40, 190, size=num_speckles)
        img[ys, xs] = np.minimum(img[ys, xs], vals)

    return np.clip(img, 0, 255).astype(np.uint8)


def render_degraded_line(
    text: str,
    font_path: str,
    font_size: int = 34,
    vertical_pad_ratio: float = 0.12,
    rng: random.Random = None,
    ghost_text: str = None,
) -> Image.Image:
    """Render a Myanmar/Pali line with HarfBuzz shaping, 12% vertical padding, and lead-type degradations."""
    rng = rng or random.Random()
    font = ImageFont.truetype(font_path, font_size)

    # Measure text generously on temporary canvas
    canvas_w = max(800, len(text) * font_size + 120)
    canvas_h = font_size * 4
    tmp = Image.new("L", (canvas_w, canvas_h), 255)
    draw = ImageDraw.Draw(tmp)
    draw.text((40, font_size), text, font=font, fill=rng.randint(10, 35))

    arr = np.array(tmp)
    ys, xs = np.where(arr < 245)
    if len(ys) == 0 or len(xs) == 0:
        return tmp.crop((0, 0, 400, 80))

    y1, y2 = int(ys.min()), int(ys.max()) + 1
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    ink_h = y2 - y1

    # Strictly enforce >= 12% vertical padding top & bottom (protects ္ဓ, ္ပ, ္ဌ, ိ, ီ, ံ)
    pad_y = max(10, int(math.ceil(ink_h * max(vertical_pad_ratio, 0.12))))
    pad_x = max(16, int(font_size * 0.6))

    cy1 = max(0, y1 - pad_y)
    cy2 = min(canvas_h, y2 + pad_y)
    cx1 = max(0, x1 - pad_x)
    cx2 = min(canvas_w, x2 + pad_x)
    cropped = arr[cy1:cy2, cx1:cx2]

    ghost_arr = None
    if ghost_text:
        g_tmp = Image.new("L", (cropped.shape[1], cropped.shape[0]), 255)
        ImageDraw.Draw(g_tmp).text((pad_x, pad_y // 2), ghost_text[: len(text)], font=font, fill=20)
        ghost_arr = np.array(g_tmp)

    degraded = apply_lead_type_degradations(cropped, rng=rng, ghost_gray=ghost_arr)
    return Image.fromarray(degraded, mode="L")


if __name__ == "__main__":
    fonts = discover_myanmar_fonts()
    print("Discovered Myanmar fonts:", fonts)
    if fonts:
        os.makedirs("/tmp/v11_pipeline/samples", exist_ok=True)
        rng = random.Random(42)
        for i, line in enumerate(CLASSICAL_SEED_LINES[:4], 1):
            im = render_degraded_line(
                line, fonts[i % len(fonts)], font_size=34, vertical_pad_ratio=0.12,
                rng=rng, ghost_text=CLASSICAL_SEED_LINES[(i + 5) % len(CLASSICAL_SEED_LINES)]
            )
            out = f"/tmp/v11_pipeline/samples/sample_{i:02d}.png"
            im.save(out)
            print(f"Rendered {out}: {im.size} | {line}")
