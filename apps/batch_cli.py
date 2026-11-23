#!/usr/bin/env python3
"""System 3: 1-Click Batch Folder PDF -> Word (.docx) Converter with GitHub Model Auto-Update."""
import os, sys, glob, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.updater import ROOT_DIR, TESSDATA_DIR, get_active_model_name, check_and_update
from core.ocr_engine import convert_pdf_to_docx


def main():
    ap = argparse.ArgumentParser(description="Batch convert all PDFs in input/ to proofread Word (.docx) in output/")
    ap.add_argument("--input", default=os.path.join(ROOT_DIR, "input"), help="Input folder containing PDFs")
    ap.add_argument("--output", default=os.path.join(ROOT_DIR, "output"), help="Output folder for .docx files")
    ap.add_argument("--offset", type=int, default=0, help="Original book page offset (pdf_page - offset)")
    ap.add_argument("--no-update", action="store_true", help="Skip GitHub model update check")
    args = ap.parse_args()

    if not args.no_update:
        check_and_update()

    model = get_active_model_name()
    os.makedirs(args.input, exist_ok=True)
    os.makedirs(args.output, exist_ok=True)
    pdfs = sorted(glob.glob(os.path.join(args.input, "*.pdf")))
    if not pdfs:
        print(f"No PDF files found in {args.input}. Drop PDFs there and re-run!")
        return

    print(f"Found {len(pdfs)} PDF(s). Active model: {model}")
    for i, pdf_p in enumerate(pdfs, 1):
        name = os.path.basename(pdf_p)
        out_p = os.path.join(args.output, os.path.splitext(name)[0] + "_OCR.docx")
        print(f"[{i}/{len(pdfs)}] Converting {name} -> {os.path.basename(out_p)}")
        res = convert_pdf_to_docx(
            pdf_p, out_p, TESSDATA_DIR, lang=model, page_offset=args.offset,
            progress_cb=lambda d, t, p: print(f"  {d}/{t} pages...", flush=True) if (d % 25 == 0 or d == t) else None
        )
        print(f"  Done in {res['elapsed_sec']}s -> {res['out_docx']}")


if __name__ == "__main__":
    main()
