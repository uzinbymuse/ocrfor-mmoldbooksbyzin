"""Unified Integration Pipeline for core/ocr_engine.py (All 4 Modules Combined).

Connects:
    1. ``image_preprocessor.py`` — 300 DPI render, deskew, CLAHE, Sauvola/Niblack
    2. ``myanmar_normalizer.py`` — UTN #11 canonical Unicode ordering + NFC
    3. ``hitl_confidence.py``    — tesserocr/TSV line & word confidence + HITL export
    4. ``post_corrector.py``     — confidence-gated local Ollama + archaic guard

Design highlight (per Myanmar OCR best practices):
    - Keeps a **deskewed grayscale** page image (``page_XXXX_gray.png``) for
      ``apps/web_app.py`` HITL ``gt/*.png`` line cropping so training data is
      never polluted by binarization artifacts.
    - Uses the winning per-stratum recipe (e.g., ``clean`` vs ``faded``) for
      Tesseract inference.
    - Preserves ``core/ocr_engine.py``'s ``page_XXXX.txt`` cache format while
      adding ``page_XXXX_qc.json`` for instant Web Studio highlighting & diffs.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

# Support both `from core.<mod>` (inside repo) and flat sibling imports (standalone)
try:
    from core.hitl_confidence import (
        LineResult,
        extract_lines_with_tesserocr,
        handle_save_gt_payload,
        parse_tesseract_tsv,
        serialize_page_qc,
    )
    from core.image_preprocessor import (
        PreprocessConfig,
        preprocess_page_image,
        render_pdf_page,
    )
    from core.myanmar_normalizer import normalize_myanmar_unicode, validate_myanmar_line
    from core.post_corrector import (
        CorrectorConfig,
        correct_page_lines,
        load_archaic_map,
    )
except ImportError:
    from hitl_confidence import (  # type: ignore
        LineResult,
        extract_lines_with_tesserocr,
        handle_save_gt_payload,
        parse_tesseract_tsv,
        serialize_page_qc,
    )
    from image_preprocessor import (  # type: ignore
        PreprocessConfig,
        preprocess_page_image,
        render_pdf_page,
    )
    from myanmar_normalizer import normalize_myanmar_unicode, validate_myanmar_line  # type: ignore
    from post_corrector import (  # type: ignore
        CorrectorConfig,
        correct_page_lines,
        load_archaic_map,
    )


# Example deterministic watermark/URL filter matching core/ocr_engine.py
_WATERMARK_RE = re.compile(r"(dhammadownload|https?://|www\.)", re.IGNORECASE)


@dataclass
class PipelineOptions:
    """Runtime options passed into ``_ocr_worker`` in ``core/ocr_engine.py``."""

    tessdata_dir: Union[str, Path] = "tessdata"
    lang: str = "mya_dhamma_v9_balanced"
    psm: int = 6
    page_stratum: str = "clean"  # "clean" | "faded" | "preface"
    recipes_json: Optional[Union[str, Path]] = None
    enable_ollama: bool = False
    ollama_model: str = "qwen3:8b"
    archaic_map_path: Optional[Union[str, Path]] = None
    save_gray_for_hitl: bool = True


# ---------------------------------------------------------------------------
# 1. Drop-In Replacement for clean_line() in core/ocr_engine.py
# ---------------------------------------------------------------------------

def clean_line_v2(
    line: str,
    compiled_word_fixes: Sequence[Tuple[re.Pattern[str], str]] = (),
) -> str:
    """Clean a single OCR line: strip watermarks -> UTN #11 -> WORD_FIXES."""
    if not line or _WATERMARK_RE.search(line):
        return ""

    # 1. Canonical UTN #11 ordering + zero-width/punctuation cleanup
    out = normalize_myanmar_unicode(line)

    # 2. Existing book-specific WORD_FIXES regex rules from core/ocr_engine.py
    for pattern, repl in compiled_word_fixes:
        out = pattern.sub(repl, out)

    # 3. Final UTN #11 pass in case WORD_FIXES rewrote diacritics
    return normalize_myanmar_unicode(out)


# ---------------------------------------------------------------------------
# 2. Load Stratum-Specific Preprocessing Recipe (Clean vs Faded)
# ---------------------------------------------------------------------------

def load_stratum_recipe(
    stratum: str,
    recipes_json: Optional[Union[str, Path]] = None,
) -> Tuple[PreprocessConfig, int]:
    """Load the winning PreprocessConfig & PSM for ``stratum`` from A/B grid JSON."""
    if recipes_json is not None and Path(recipes_json).exists():
        data = json.loads(Path(recipes_json).read_text(encoding="utf-8"))
        best = data.get("best_by_stratum", {}).get(stratum)
        if best:
            cfg_dict = dict(best["config"])
            if "clahe_tile_grid" in cfg_dict and isinstance(cfg_dict["clahe_tile_grid"], list):
                cfg_dict["clahe_tile_grid"] = tuple(cfg_dict["clahe_tile_grid"])
            return PreprocessConfig(**cfg_dict), int(best.get("psm", 6))

    # Sensible defaults when no A/B grid JSON has been generated yet
    if stratum in ("faded", "preface"):
        return PreprocessConfig(dpi=300, apply_clahe=True, output_mode="sauvola"), 6
    return PreprocessConfig(dpi=300, apply_clahe=False, output_mode="sauvola"), 6


# ---------------------------------------------------------------------------
# 3. OCR Runner Returning Structured LineResults (tesserocr -> CLI TSV Fallback)
# ---------------------------------------------------------------------------

def run_ocr_with_confidence(
    pil_img: Image.Image,
    *,
    tessdata_dir: Union[str, Path],
    lang: str,
    psm: int,
) -> List[LineResult]:
    """Extract lines + word confidences using tesserocr or CLI TSV fallback."""
    try:
        return extract_lines_with_tesserocr(
            pil_img,
            tessdata_dir=tessdata_dir,
            lang=lang,
            psm=psm,
        )
    except ImportError:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        img_path = Path(tmp) / "page.png"
        pil_img.save(img_path, format="PNG")
        proc = subprocess.run(
            [
                "tesseract",
                str(img_path),
                "stdout",
                "--tessdata-dir",
                str(tessdata_dir),
                "-l",
                lang,
                "--psm",
                str(psm),
                "tsv",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return parse_tesseract_tsv(proc.stdout)


# ---------------------------------------------------------------------------
# 4. Drop-In Page Worker for _ocr_worker() in core/ocr_engine.py
# ---------------------------------------------------------------------------

def process_single_page(
    *,
    page_no: int,
    cache_dir: Union[str, Path],
    options: PipelineOptions,
    raw_pil_img: Optional[Image.Image] = None,
    pdf_path: Optional[Union[str, Path]] = None,
    compiled_word_fixes: Sequence[Tuple[re.Pattern[str], str]] = (),
    ocr_lines_fn: Optional[Callable[[Image.Image, int], List[LineResult]]] = None,
    llm_fn: Optional[Callable[[str, Optional[Image.Image]], str]] = None,
) -> Dict[str, Any]:
    """End-to-end single-page OCR + QC + optional Ollama post-correction.

    Writes:
      - ``cache_dir/page_{page_no:04d}.txt``      (compatible with core/ocr_engine.py)
      - ``cache_dir/page_{page_no:04d}_qc.json``  (confidence highlights + diffs for Web Studio)
      - ``cache_dir/page_{page_no:04d}_gray.png`` (clean deskewed grayscale for HITL GT crops)
    """
    cdir = Path(cache_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    txt_cache = cdir / f"page_{page_no:04d}.txt"
    qc_cache = cdir / f"page_{page_no:04d}_qc.json"
    gray_cache = cdir / f"page_{page_no:04d}_gray.png"

    # Fast cache hit (preserves core/ocr_engine.py resume behavior)
    if txt_cache.exists() and qc_cache.exists():
        return {
            "page_no": page_no,
            "cached": True,
            "text": txt_cache.read_text(encoding="utf-8"),
            "qc_path": str(qc_cache),
            "gray_path": str(gray_cache) if gray_cache.exists() else None,
        }

    # Step 1: Render 300 DPI page (or accept pre-loaded PIL image)
    if raw_pil_img is None:
        if pdf_path is None:
            raise ValueError("Provide either raw_pil_img or pdf_path.")
        raw_pil_img = render_pdf_page(pdf_path, page_index=page_no - 1, dpi=300)

    # Step 2: Produce deskewed grayscale image for HITL ground-truth cropping
    gray_cfg = PreprocessConfig(dpi=300, output_mode="grayscale", apply_clahe=False)
    hitl_gray_img = preprocess_page_image(raw_pil_img, config=gray_cfg)
    if options.save_gray_for_hitl:
        hitl_gray_img.save(gray_cache, format="PNG")

    # Step 3: Produce inference image using the winning stratum recipe
    infer_cfg, recipe_psm = load_stratum_recipe(options.page_stratum, options.recipes_json)
    psm = options.psm or recipe_psm
    infer_img = preprocess_page_image(raw_pil_img, config=infer_cfg)

    # Step 4: Run OCR with line/word confidence extraction
    if ocr_lines_fn is not None:
        lines = ocr_lines_fn(infer_img, psm)
    else:
        lines = run_ocr_with_confidence(
            infer_img,
            tessdata_dir=options.tessdata_dir,
            lang=options.lang,
            psm=psm,
        )

    # Step 5: Apply clean_line_v2() + UTN #11 validation per line
    cleaned_lines: List[LineResult] = []
    validation_flags: List[Dict[str, Any]] = []
    for ln in lines:
        cleaned_text = clean_line_v2(ln.text, compiled_word_fixes=compiled_word_fixes)
        if not cleaned_text:
            continue
        ln.text = cleaned_text
        issues = validate_myanmar_line(cleaned_text)
        if issues:
            validation_flags.append(
                {
                    "line_index": ln.line_index,
                    "issues": [{"kind": i.kind, "text": i.text} for i in issues],
                }
            )
        cleaned_lines.append(ln)

    qc_payload = serialize_page_qc(cleaned_lines, page_no=page_no, image_size=infer_img.size)
    qc_payload["validation_flags"] = validation_flags

    # Step 6: Optional confidence-gated local Ollama post-correction
    if options.enable_ollama:
        archaic_map = load_archaic_map(options.archaic_map_path)
        corr_cfg = CorrectorConfig(model=options.ollama_model)
        corr_payload = correct_page_lines(
            cleaned_lines,
            config=corr_cfg,
            archaic_map=archaic_map,
            page_img=hitl_gray_img,
            llm_fn=llm_fn,
        )
        qc_payload["post_correction"] = corr_payload
        final_text = corr_payload["corrected_page_text"]
    else:
        final_text = "\n".join(ln.text for ln in cleaned_lines)

    # Step 7: Persist cache files for .docx export & apps/web_app.py
    txt_cache.write_text(final_text + "\n", encoding="utf-8")
    qc_cache.write_text(json.dumps(qc_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return {
        "page_no": page_no,
        "cached": False,
        "text": final_text,
        "txt_path": str(txt_cache),
        "qc_path": str(qc_cache),
        "gray_path": str(gray_cache) if options.save_gray_for_hitl else None,
        "summary": qc_payload["summary"],
    }


# ---------------------------------------------------------------------------
# End-to-End Integration Self-Test (All 4 Modules Together)
# ---------------------------------------------------------------------------

def _run_integration_test() -> None:
    sample_tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t80\t100\t140\t36\t94.0\tဗုဒ္ဓ\n"
        "5\t1\t1\t1\t1\t2\t230\t100\t140\t36\t93.0\tဓမ္မ\n"
        "5\t1\t1\t2\t1\t1\t80\t160\t120\t36\t66.0\tတရုပ်\n"
        "5\t1\t1\t2\t1\t2\t210\t160\t220\t36\t64.0\tပုံနှိပ်စာလုံးဟောင်။\n"
    )

    def _mock_ocr(_img: Image.Image, _psm: int) -> List[LineResult]:
        return parse_tesseract_tsv(sample_tsv)

    def _mock_ollama(prompt: str, _crop: Optional[Image.Image]) -> str:
        # Fixes typo ဟောင်->ဟောင်း while attempting to modernize တရုပ်->တရုတ်
        if "တရုပ်" in prompt:
            return "တရုတ် ပုံနှိပ်စာလုံးဟောင်း။"
        return ""

    raw_page = Image.new("RGB", (800, 1000), color=(240, 235, 225))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cache_dir = root / "cache"
        gt_dir = root / "data" / "ground_truth"

        result = process_single_page(
            page_no=17,
            cache_dir=cache_dir,
            raw_pil_img=raw_page,
            options=PipelineOptions(
                page_stratum="faded",
                enable_ollama=True,
                save_gray_for_hitl=True,
            ),
            ocr_lines_fn=_mock_ocr,
            llm_fn=_mock_ollama,
        )

        # Verify caches exist and archaic spelling was preserved while typo was fixed
        assert Path(result["txt_path"]).exists()
        assert Path(result["qc_path"]).exists()
        assert Path(result["gray_path"]).exists()
        assert "တရုပ် ပုံနှိပ်စာလုံးဟောင်း။" in result["text"]

        # Simulate Web Studio HITL save using the clean deskewed grayscale image
        with Image.open(result["gray_path"]) as gray_im:
            gt_res = handle_save_gt_payload(
                {
                    "book_id": "vol1_dhamma",
                    "page_no": 17,
                    "line_index": 1,
                    "bbox": [80, 160, 430, 196],
                    "original_ocr_text": "တရုပ် ပုံနှိပ်စာလုံးဟောင်။",
                    "corrected_text": "တရုပ် ပုံနှိပ်စာလုံးဟောင်း။",
                    "original_confidence": 65.0,
                },
                page_img=gray_im,
                gt_dir=gt_dir,
            )
        assert Path(gt_res["png_path"]).exists()
        assert Path(gt_res["gt_txt_path"]).read_text(encoding="utf-8") == "တရုပ် ပုံနှိပ်စာလုံးဟောင်း။\n"

    print("Full 4-module end-to-end integration test PASSED.")


if __name__ == "__main__":
    _run_integration_test()