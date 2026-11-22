#!/usr/bin/env python3
import os, sys, json, subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import pypdfium2 as pdfium
from PIL import ImageOps

def ocr_page(args):
    pdf_path, tessdata_dir, lang, pno, out_dir = args
    txt_path = os.path.join(out_dir, f"page_{pno:03d}.txt")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        return pno, open(txt_path, encoding="utf-8").read()
    pdf = pdfium.PdfDocument(pdf_path)
    page = pdf[pno - 1]
    _, ph = page.get_size()
    img = page.render(scale=min(3.0, 2800.0 / ph)).to_pil()
    pdf.close()
    img = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)
    tmp_png = os.path.join(out_dir, f"_tmp_{pno:03d}.png")
    img.save(tmp_png)
    res = subprocess.run(
        ["tesseract", tmp_png, "stdout", "--tessdata-dir", tessdata_dir, "-l", lang, "--psm", "6"],
        capture_output=True, text=True, env={"OMP_THREAD_LIMIT": "1", "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    )
    os.remove(tmp_png)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(res.stdout)
    return pno, res.stdout

def main():
    pdf_path = sys.argv[1]
    tessdata_dir = sys.argv[2] if len(sys.argv) > 2 else "./tessdata"
    lang = sys.argv[3] if len(sys.argv) > 3 else "mya_dhamma_v2"
    out_dir = os.path.splitext(os.path.basename(pdf_path))[0] + "_ocr_pages"
    os.makedirs(out_dir, exist_ok=True)
    pdf = pdfium.PdfDocument(pdf_path)
    total = len(pdf)
    pdf.close()
    results = {}
    with ProcessPoolExecutor(max_workers=2) as ex:
        futs = {ex.submit(ocr_page, (pdf_path, tessdata_dir, lang, p, out_dir)): p for p in range(1, total + 1)}
        for fut in as_completed(futs):
            pno, txt = fut.result()
            results[str(pno)] = txt
    out_json = os.path.splitext(os.path.basename(pdf_path))[0] + "_ocr.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"Saved -> {out_json}")

if __name__ == "__main__":
    main()
