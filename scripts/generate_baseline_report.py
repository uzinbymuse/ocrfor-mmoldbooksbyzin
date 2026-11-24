import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.myanmar_normalizer import normalize_myanmar_unicode
from scripts.eval_golden import evaluate_manifest

def get_tesseract_cmd():
    cand = shutil.which("tesseract")
    if cand:
        return cand
    win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return win if os.path.exists(win) else "tesseract"

def main():
    manifest_path = REPO_ROOT / "data" / "golden" / "manifest.json"
    hyp_dir = REPO_ROOT / "reports" / "hyp_v10"
    hyp_dir.mkdir(parents=True, exist_ok=True)
    
    tess = get_tesseract_cmd()
    tessdata_dir = str(REPO_ROOT / "tessdata")
    
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    for page in raw.get("pages", []):
        pid = page["page_id"]
        img_rel = page["image_path"]
        img_path = REPO_ROOT / img_rel
        out_txt = hyp_dir / f"{pid}.txt"
        
        print(f"[INFO] Running OCR with mya_dhamma_v10 on {img_path.name}...")
        cmd = [
            tess, str(img_path), "stdout",
            "--tessdata-dir", tessdata_dir,
            "-l", "mya_dhamma_v10",
            "--psm", "6"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        norm = normalize_myanmar_unicode(res.stdout)
        out_txt.write_text(norm, encoding="utf-8")
        print(f"[INFO] Wrote {out_txt.name} ({len(norm)} chars)")

    out_report = REPO_ROOT / "reports" / "baseline_v10.json"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    report = evaluate_manifest(
        manifest_path,
        hyp_dir=hyp_dir,
        metadata={"model": "mya_dhamma_v10", "psm": 6, "date": "2026-11-24"}
    )
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[SUCCESS] Baseline v10 report saved to {out_report}")
    print(f"Overall CER: {report['overall']['cer_pct']}% | Accuracy: {report['overall']['char_accuracy_pct']}%")

if __name__ == "__main__":
    main()
