import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
from core.myanmar_normalizer import normalize_myanmar_unicode
GOLDEN_DIR = os.path.join(REPO_ROOT, "data", "golden")
REFS_DIR = os.path.join(GOLDEN_DIR, "refs")
os.makedirs(REFS_DIR, exist_ok=True)

# 1. kaccayana_p5
gt_json_path = os.path.join(REPO_ROOT, "data", "ground_truth", "batch6_kaccayana_p5.json")
with open(gt_json_path, "r", encoding="utf-8") as f:
    gt_data = json.load(f)

kaccayana_lines = [normalize_myanmar_unicode(item["text"]) for item in gt_data]
kaccayana_ref_path = os.path.join(REFS_DIR, "kaccayana_p5.txt")
with open(kaccayana_ref_path, "w", encoding="utf-8") as f:
    f.write("\n".join(kaccayana_lines) + "\n")

manifest = {
    "version": "1.0",
    "description": "Frozen Golden Evaluation Set for Dhamma Lead-Type Books",
    "baseline_model": "mya_dhamma_v10",
    "pages": [
        {
            "page_id": "kaccayana_p5",
            "book_id": "basic_kaccayana",
            "stratum": "grammar_leadtype",
            "image_path": "data/eval_benchmarks/kaccayana_p5.png",
            "ref_txt": "refs/kaccayana_p5.txt"
        }
    ]
}

manifest_path = os.path.join(GOLDEN_DIR, "manifest.json")
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)

print(f"[SUCCESS] Created {kaccayana_ref_path} ({len(kaccayana_lines)} lines)")
print(f"[SUCCESS] Created {manifest_path}")
