import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import subprocess
from core.myanmar_normalizer import normalize_myanmar_unicode
from core.post_processor import proofread_text
TESSDATA_DIR = os.path.join(REPO_ROOT, "tessdata")
IMAGE_PATH = os.path.join(REPO_ROOT, "scratch", "kaccayana_p5_crops", "page5_clean.png")
if not os.path.exists(IMAGE_PATH):
    IMAGE_PATH = os.path.join(REPO_ROOT, "data", "eval_benchmarks", "kaccayana_p5.png")
GT_PATH = os.path.join(REPO_ROOT, "data", "ground_truth", "batch6_kaccayana_p5.json")

def get_tesseract_cmd():
    import shutil
    cand = shutil.which("tesseract")
    if cand:
        return cand
    win_def = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(win_def):
        return win_def
    return "tesseract"

def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def run_ocr(model_name, psm="6"):
    tess = get_tesseract_cmd()
    cmd = [
        tess, IMAGE_PATH, "stdout",
        "--tessdata-dir", TESSDATA_DIR,
        "-l", model_name,
        "--psm", str(psm)
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
    return res.stdout

def evaluate():
    with open(GT_PATH, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    gt_lines = [normalize_myanmar_unicode(item["text"]) for item in gt_data]
    gt_full = "\n".join(gt_lines)
    gt_chars = len(gt_full.replace("\n", "").replace(" ", ""))
    gt_commas = gt_full.count("၊")
    gt_dandas = gt_full.count("။")
    
    print(f"=== GROUND TRUTH STATS ===")
    print(f"Lines: {len(gt_lines)}, Chars: {gt_chars}, Commas (၊): {gt_commas}, Dandas (။): {gt_dandas}")
    print()

    for model in ["mya_dhamma_v10", "mya_dhamma_v11"]:
        for psm in ["6"]:
            raw_out = run_ocr(model, psm=psm)
            norm_out = normalize_myanmar_unicode(raw_out)
            post_out = proofread_text(norm_out)
            
            # Compare raw
            clean_gt = "".join(gt_full.split())
            clean_norm = "".join(norm_out.split())
            clean_post = "".join(post_out.split())
            
            raw_dist = levenshtein_distance(clean_gt, clean_norm)
            raw_acc = max(0.0, (1.0 - raw_dist / max(len(clean_gt), 1)) * 100)
            
            post_dist = levenshtein_distance(clean_gt, clean_post)
            post_acc = max(0.0, (1.0 - post_dist / max(len(clean_gt), 1)) * 100)
            
            hyp_commas = norm_out.count("၊")
            hyp_dandas = norm_out.count("။")
            
            print(f"[{model} | PSM {psm}]")
            print(f"  Raw Char Accuracy: {raw_acc:.2f}% (Errors: {raw_dist} / {len(clean_gt)})")
            print(f"  Post-Processed Accuracy: {post_acc:.2f}% (Errors: {post_dist} / {len(clean_gt)})")
            print(f"  Punctuation: Commas(၊)={hyp_commas} (GT={gt_commas}), Dandas(။)={hyp_dandas} (GT={gt_dandas})")
            print("-" * 50)

if __name__ == "__main__":
    evaluate()
