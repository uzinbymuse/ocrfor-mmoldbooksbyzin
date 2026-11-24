#!/usr/bin/env python3
"""
Line-Level OCR & Dataset Harvesting Tool (စာကြောင်းအလိုက် ဖြတ်တောက်ဖတ်ရှုခြင်းနှင့် Dataset ထုတ်ယူခြင်း)

Methodology:
1. Line Segmentation: Identifies line bounding boxes from page image.
2. 12% Vertical Padding: Preserves Burmese upper vowels (ိ ီ ံ င်္) & subjoined consonants (္ဓ ္ပ ္ဌ).
3. Line Recognition (--psm 7): Runs Tesseract LSTM in single-line mode for highest accuracy.
4. Normalization & Post-processing: Applies UTN #11 canonical ordering.
5. Export: Produces full-page stitched text AND optional line crop datasets (.png + .gt.txt).
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.hitl_confidence import parse_tesseract_tsv, crop_line_with_padding
from core.myanmar_normalizer import normalize_myanmar_unicode
from core.post_processor import proofread_text

def get_tesseract_cmd():
    cand = shutil.which("tesseract")
    if cand:
        return cand
    win = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    return win if os.path.exists(win) else "tesseract"

def process_page_line_by_line(
    image_path: Path,
    model_name: str = "mya_dhamma_v10",
    tessdata_dir: Path = None,
    export_dataset_dir: Path = None,
    book_id: str = "book",
    page_no: int = 1,
    vertical_pad_ratio: float = 0.12,
) -> str:
    tess = get_tesseract_cmd()
    tessdata_dir = str(tessdata_dir or (REPO_ROOT / "tessdata"))
    img = Image.open(image_path).convert("RGB")
    
    # Step 1: Detect line bounding boxes via TSV
    cmd_tsv = [
        tess, str(image_path), "stdout",
        "--tessdata-dir", tessdata_dir,
        "-l", model_name,
        "--psm", "6",
        "-c", "tessedit_create_tsv=1"
    ]
    tsv_res = subprocess.run(cmd_tsv, capture_output=True, text=True, encoding="utf-8")
    lines = parse_tesseract_tsv(tsv_res.stdout)
    
    if not lines:
        print(f"[WARN] No lines detected in {image_path.name}")
        return ""

    print(f"[INFO] Detected {len(lines)} lines in {image_path.name}. Processing line-by-line (--psm 7)...")
    
    extracted_page_lines = []
    temp_dir = REPO_ROOT / "scratch" / "temp_line_crops"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    if export_dataset_dir:
        export_dataset_dir.mkdir(parents=True, exist_ok=True)

    for idx, line in enumerate(lines):
        # Step 2: Crop line with 12% vertical padding
        crop_img = crop_line_with_padding(img, line.bbox, vertical_pad_ratio=vertical_pad_ratio)
        temp_crop_file = temp_dir / f"temp_{idx:03d}.png"
        crop_img.save(temp_crop_file)

        # Step 3: Run single-line OCR (--psm 7)
        cmd_line = [
            tess, str(temp_crop_file), "stdout",
            "--tessdata-dir", tessdata_dir,
            "-l", model_name,
            "--psm", "7"
        ]
        line_res = subprocess.run(cmd_line, capture_output=True, text=True, encoding="utf-8")
        
        # Step 4: Normalize & Post-process
        norm_txt = normalize_myanmar_unicode(line_res.stdout).replace("\n", " ").strip()
        clean_txt = proofread_text(norm_txt)
        
        extracted_page_lines.append(clean_txt)
        
        # Step 5: Optional Dataset Export (Line crop PNG + GT TXT)
        if export_dataset_dir and clean_txt:
            stem = f"{book_id}_p{page_no:04d}_l{idx:03d}"
            out_png = export_dataset_dir / f"{stem}.png"
            out_gt = export_dataset_dir / f"{stem}.gt.txt"
            crop_img.save(out_png)
            out_gt.write_text(clean_txt + "\n", encoding="utf-8")
            
        try:
            temp_crop_file.unlink()
        except Exception:
            pass

    full_page_text = "\n".join(extracted_page_lines)
    return full_page_text

def main():
    parser = argparse.ArgumentParser(description="Line-Level OCR & Dataset Harvester")
    parser.add_argument("--image", type=str, required=True, help="Path to page image (.png, .jpg)")
    parser.add_argument("--model", type=str, default="mya_dhamma_v10", help="Tesseract model name")
    parser.add_argument("--export-dataset", type=str, default=None, help="Directory to save line crops and GT texts")
    parser.add_argument("--book-id", type=str, default="sample_book", help="Book ID prefix for exported files")
    parser.add_argument("--page-no", type=int, default=1, help="Page number")
    parser.add_argument("--out-txt", type=str, default=None, help="Path to save assembled page text")
    
    args = parser.parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)
        
    export_dir = Path(args.export_dataset) if args.export_dataset else None
    
    page_text = process_page_line_by_line(
        image_path=img_path,
        model_name=args.model,
        export_dataset_dir=export_dir,
        book_id=args.book_id,
        page_no=args.page_no,
    )
    
    if args.out_txt:
        out_p = Path(args.out_txt)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(page_text, encoding="utf-8")
        print(f"[SUCCESS] Wrote full page text to {out_p}")
    else:
        print("\n=== EXTRACTED PAGE TEXT (LINE-BY-LINE) ===")
        print(page_text)

if __name__ == "__main__":
    main()
