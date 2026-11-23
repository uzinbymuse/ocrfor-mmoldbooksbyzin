#!/usr/bin/env python3
"""Shared multi-core PDF -> OCR -> Proofread Word (.docx) engine for Myanmar Dhamma books."""
import os
import re
import time
import shutil
import io
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    import fitz
except ImportError:
    fitz = None
try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None
try:
    import tesserocr
except ImportError:
    tesserocr = None
try:
    import pypdf
except ImportError:
    pypdf = None
from PIL import Image, ImageOps, ImageDraw
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

EN_TO_MYA = str.maketrans("0123456789", "၀၁၂၃၄၅၆၇၈၉")
PRIMARY_RGB = RGBColor(0x7A, 0x1F, 0x1F)
SUBHEAD_RGB = RGBColor(0x4A, 0x18, 0x18)
TEXT_RGB = RGBColor(0x1A, 0x1A, 0x1A)
MUTED_RGB = RGBColor(0x55, 0x55, 0x55)
BORDER_HEX = "D8C9B8"
FONT_NAME = "Noto Serif Myanmar"

WORD_FIXES = [
    (r"ဣတတိပိ", "ဣတိပိ"), (r"က္ကတိပိ|က္တိပိ", "ဣတိပိ"),
    (r"က္ကတ္ထိယော|က္တ္ထိယော", "ဣတ္ထိယော"), (r"က္ကဒ္ဓိ|က္တဒ္ဓိ", "ဣဒ္ဓိ"),
    (r"က္ကမံ|က္တမံ", "ဣမံ"), (r"သရဇ္စျာယ်", "သရဇ္ဈာယ်"),
    (r"ပုဏ္ဌား|ပုဏ္လား", "ပုဏ္ဏား"), (r"သြဝါဒ", "ဩဝါဒ"),
    (r"သြဃ", "ဩဃ"), (r"သြဘာသ", "ဩဘာသ"), (r"သြပနေယျိက", "ဩပနေယျိက"),
    (r"ဥာဏ်", "ဉာဏ်"), (r"ဥာဏ", "ဉာဏ"), (r"ရွဲ", "၍"),
    (r"[ႆၱၿ]", ""), (r"\[\||\|\]|\|", "။"),
    (r"(?<=[က-အာ-်])[\]\[]+(?=\s|$|[။၊])", "၊"),
]
COMPILED_FIXES = [(re.compile(p), r) for p, r in WORD_FIXES]


def to_mya_num(n: int) -> str:
    return str(n).translate(EN_TO_MYA)


def find_tesseract_cmd():
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    for win_path in [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]:
        if os.path.exists(win_path):
            return win_path
    return "tesseract"


def clean_line(line: str) -> str:
    line = line.replace("​", "").replace("﻿", "").strip()
    if not line or "dhammadownload" in line.lower() or "http://" in line.lower():
        return ""
    for pat, repl in COMPILED_FIXES:
        line = pat.sub(repl, line)
    line = re.sub(r"[။]{2,}", "။", line)
    line = re.sub(r"[၊]{2,}", "၊", line)
    line = re.sub(r"\s+([။၊])", r"", line)
    return re.sub(r"[ 	]{2,}", " ", line).strip()


def _ocr_worker(args):
    pdf_path, pno, tessdata_dir, lang, cache_dir, crop_hf = args
    txt_path = os.path.join(cache_dir, f"page_{pno:04d}.txt")
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        with open(txt_path, encoding="utf-8") as f:
            return pno, f.read()

    img = None
    if fitz is not None:
        doc = fitz.open(pdf_path)
        page = doc[pno - 1]
        pix = page.get_pixmap(dpi=180)
        img = Image.open(io.BytesIO(pix.tobytes()))
        doc.close()
    elif pdfium is not None:
        pdf = pdfium.PdfDocument(pdf_path)
        page = pdf[pno - 1]
        _, ph = page.get_size()
        scale = min(3.0, 2800.0 / ph)
        img = page.render(scale=scale).to_pil()
        pdf.close()
    else:
        raise RuntimeError("Neither PyMuPDF (fitz) nor pypdfium2 is installed.")

    w, h = img.size
    if crop_hf:
        img = img.crop((0, int(h * 0.045), w, int(h * 0.915)))
        cw, ch = img.size
        ImageDraw.Draw(img).rectangle([int(cw * 0.78), 0, cw, int(ch * 0.045)], fill=(255, 255, 255))

    img = ImageOps.autocontrast(ImageOps.grayscale(img), cutoff=1)

    text = ""
    if tesserocr is not None:
        try:
            api = tesserocr.PyTessBaseAPI(path=tessdata_dir, lang=lang, psm=tesserocr.PSM.SINGLE_BLOCK)
            api.SetImage(img)
            text = api.GetUTF8Text()
        except Exception:
            text = ""

    if not text:
        tmp_png = os.path.join(cache_dir, f"_tmp_{pno:04d}.png")
        img.save(tmp_png)
        env = os.environ.copy()
        env["OMP_THREAD_LIMIT"] = "1"
        tess_cmd = find_tesseract_cmd()
        res = subprocess.run(
            [tess_cmd, tmp_png, "stdout", "--tessdata-dir", tessdata_dir, "-l", lang, "--psm", "6"],
            capture_output=True, text=True, env=env
        )
        try:
            os.remove(tmp_png)
        except OSError:
            pass
        text = res.stdout

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)
    return pno, text


def extract_pdf_toc(pdf_path):
    if fitz is not None:
        try:
            doc = fitz.open(pdf_path)
            toc = doc.get_toc()
            doc.close()
            if toc:
                return [(t[1].strip(), t[2]) for t in toc if t[1].strip()]
        except Exception:
            pass
    if pypdf is not None:
        try:
            reader = pypdf.PdfReader(pdf_path)
            def walk(items):
                out = []
                for it in items:
                    if isinstance(it, list):
                        out.extend(walk(it))
                    else:
                        try:
                            p = reader.get_destination_page_number(it) + 1
                            out.append((it.title.strip(), p))
                        except Exception:
                            pass
                return out
            return walk(reader.outline or [])
        except Exception:
            pass
    return []


def _set_font(run, size_pt=11, bold=False, color=TEXT_RGB):
    run.font.name = FONT_NAME
    run.font.size = Pt(size_pt)
    run.font.bold = bold
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    rf = rPr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rPr.insert(0, rf)
    for attr in ("w:ascii", "w:hAnsi", "w:cs"):
        rf.set(qn(attr), FONT_NAME)


def _add_bottom_border(p, color=BORDER_HEX, sz="4"):
    p._p.get_or_add_pPr().append(
        parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="{sz}" w:space="3" w:color="{color}"/></w:pBdr>')
    )


def convert_pdf_to_docx(
    pdf_path,
    out_docx,
    tessdata_dir,
    lang="mya_dhamma_v10",
    start_page=1,
    end_page=None,
    page_offset=0,
    workers=None,
    crop_hf=True,
    progress_cb=None,
):
    """Full pipeline: parallel OCR with resume cache -> proofread Word (.docx) with TOC & original page numbers."""
    t0 = time.time()
    total_pdf = 0
    if fitz is not None:
        doc_in = fitz.open(pdf_path)
        total_pdf = len(doc_in)
        doc_in.close()
    elif pdfium is not None:
        pdf = pdfium.PdfDocument(pdf_path)
        total_pdf = len(pdf)
        pdf.close()
    else:
        raise RuntimeError("No PDF library available.")

    end_page = min(end_page or total_pdf, total_pdf)
    start_page = max(1, start_page)
    pages = list(range(start_page, end_page + 1))
    workers = workers or max(1, min(os.cpu_count() or 2, 8))

    cache_dir = os.path.splitext(out_docx)[0] + "_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_docx)), exist_ok=True)

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_ocr_worker, (pdf_path, p, tessdata_dir, lang, cache_dir, crop_hf)): p for p in pages}
        for done, fut in enumerate(as_completed(futs), 1):
            pno, txt = fut.result()
            results[pno] = txt
            if progress_cb:
                progress_cb(done, len(pages), pno)

    # Build Word (.docx)
    doc = docx.Document()
    for sec in doc.sections:
        sec.page_width = Inches(8.5); sec.page_height = Inches(11.0)
        sec.top_margin = Inches(0.9); sec.bottom_margin = Inches(0.9)
        sec.left_margin = Inches(1.0); sec.right_margin = Inches(1.0)
    doc.styles["Normal"].font.name = FONT_NAME
    doc.styles["Normal"].font.size = Pt(11)

    book_title = os.path.splitext(os.path.basename(pdf_path))[0].replace("_", " ")
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    _set_font(p_title.add_run(book_title), 18, True, PRIMARY_RGB)

    toc = extract_pdf_toc(pdf_path)
    if toc:
        p_th = doc.add_paragraph()
        p_th.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_font(p_th.add_run(f"မာတိကာ ({to_mya_num(len(toc))} ပုဒ်)"), 14, True, PRIMARY_RGB)
        for title, tp in toc:
            p = doc.add_paragraph()
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
            _set_font(p.add_run(title), 10.5)
            p.add_run("	")
            disp = tp - page_offset if (tp - page_offset) > 0 else tp
            _set_font(p.add_run(to_mya_num(disp)), 10.5, True, SUBHEAD_RGB)

    for idx, pno in enumerate(pages):
        orig_p = pno - page_offset if (pno - page_offset) > 0 else pno
        pm = doc.add_paragraph()
        pm.paragraph_format.page_break_before = True
        pm.paragraph_format.space_after = Pt(6)
        pm.paragraph_format.keep_with_next = True
        pm.paragraph_format.tab_stops.add_tab_stop(Inches(6.5), WD_TAB_ALIGNMENT.RIGHT)
        _set_font(pm.add_run(book_title[:40]), 9.5, False, MUTED_RGB)
        pm.add_run("	")
        _set_font(pm.add_run(f"မူရင်းစာမျက်နှာ {to_mya_num(orig_p)}"), 9.5, True, MUTED_RGB)
        _add_bottom_border(pm)

        buf = []
        for raw_l in results.get(pno, "").splitlines():
            c = clean_line(raw_l)
            if not c:
                if buf:
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                    p.paragraph_format.first_line_indent = Inches(0.25)
                    p.paragraph_format.line_spacing = 1.15
                    p.paragraph_format.space_after = Pt(5)
                    _set_font(p.add_run(" ".join(buf)), 11)
                    buf = []
                continue
            buf.append(c)
            if c.endswith("။") and len(c) < 42:
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
                p.paragraph_format.first_line_indent = Inches(0.25)
                p.paragraph_format.line_spacing = 1.15
                p.paragraph_format.space_after = Pt(5)
                _set_font(p.add_run(" ".join(buf)), 11)
                buf = []
        if buf:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            p.paragraph_format.first_line_indent = Inches(0.25)
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(5)
            _set_font(p.add_run(" ".join(buf)), 11)

    doc.save(out_docx)
    return {"out_docx": out_docx, "pages": len(pages), "elapsed_sec": round(time.time() - t0, 1), "model": lang}
