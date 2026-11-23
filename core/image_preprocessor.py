"""300 DPI Rendering, Deskewing, CLAHE, Sauvola/Niblack/Otsu & A/B Grid Module.

Drop-in preprocessing & A/B benchmarking module designed for ``core/`` in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Capabilities:
1. Exact 300 DPI PDF rendering (PyMuPDF primary, pypdfium2 fallback).
2. Uneven paper illumination normalization & pre-rotation scanner-border trim.
3. Coarse-to-fine projection-profile deskewing.
4. Dynamic content-box margin cropping before CLAHE/binarization so local
   contrast enhancement focuses purely on the text region.
5. Optional 2x bicubic upscaling & CLAHE contrast enhancement for faded scans.
6. Exact O(N) integral-image Sauvola and Niblack adaptive binarization, plus
   Otsu and clean grayscale modes.
7. Stratified Golden-Set A/B Grid Runner that benchmarks recipes separately for
   each page stratum (``clean``, ``faded``, ``preface``, etc.) and exports the
   winning per-stratum recipes to JSON.

Usage:
    # Run built-in self-tests
    python3 image_preprocessor.py --self-test

    # Run A/B preprocessing grid on your Golden Set
    python3 image_preprocessor.py \\
        --ab-grid \\
        --manifest data/golden/manifest.json \\
        --tessdata tessdata \\
        --lang mya_dhamma_v9_balanced \\
        --fast-grid \\
        --save-grid-report reports/preprocess_ab_grid.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image


BinarizeMode = Literal["sauvola", "niblack", "otsu", "grayscale"]


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for Myanmar historical book page preprocessing."""

    dpi: int = 300
    remove_shadows: bool = True
    strip_scanner_borders: bool = True
    deskew: bool = True
    max_skew_angle: float = 5.0
    min_skew_angle: float = 0.15
    apply_clahe: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    upscale: float = 1.0
    output_mode: BinarizeMode = "sauvola"
    sauvola_window: int = 31
    sauvola_k: float = 0.20
    niblack_window: int = 31
    niblack_k: float = -0.20
    denoise_speckles: bool = True
    min_speckle_area: int = 3
    dynamic_crop: bool = True
    crop_header_footer: bool = False
    margin_pad_px: int = 24


@dataclass(frozen=True)
class GridRecipe:
    """One candidate preprocessing + PSM combination for A/B grid search."""

    name: str
    config: PreprocessConfig
    psm: int = 6


# ---------------------------------------------------------------------------
# 1. PDF Page Rendering (300 DPI)
# ---------------------------------------------------------------------------

def render_pdf_page(
    pdf_path: Union[str, Path],
    page_index: int,
    dpi: int = 300,
) -> Image.Image:
    """Render a 0-indexed PDF page at ``dpi`` (PyMuPDF primary, pypdfium2 fallback)."""
    pdf_str = str(pdf_path)

    try:
        import fitz  # PyMuPDF

        with fitz.open(pdf_str) as doc:
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=dpi, alpha=False)
            return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    except ImportError:
        pass

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_str)
    try:
        page = doc[page_index]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        return bitmap.to_pil().convert("RGB")
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# 2. Background Illumination Normalization, Scanner-Border Trim & CLAHE
# ---------------------------------------------------------------------------

def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Divide out uneven paper illumination while preserving thin diacritics."""
    h, w = gray.shape
    scale = max(1.0, max(h, w) / 600.0)
    small_w = max(16, int(w / scale))
    small_h = max(16, int(h / scale))
    small = cv2.resize(gray, (small_w, small_h), interpolation=cv2.INTER_AREA)

    kernel_size = max(15, (min(small_h, small_w) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    bg_small = cv2.morphologyEx(small, cv2.MORPH_CLOSE, kernel)
    bg_small = cv2.GaussianBlur(bg_small, (0, 0), sigmaX=kernel_size / 3.0)

    bg = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_LINEAR)
    normalized = np.clip(
        (gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)) * 255.0,
        0,
        255,
    )
    return normalized.astype(np.uint8)


def strip_black_scanner_borders(
    gray: np.ndarray,
    *,
    max_border_frac: float = 0.06,
    dark_thresh: int = 128,
    density_thresh: float = 0.60,
) -> np.ndarray:
    """Trim solid dark scanner edges along the outer perimeter before deskewing."""
    h, w = gray.shape
    dark = (gray < dark_thresh).astype(np.uint8)

    row_density = dark.mean(axis=1)
    col_density = dark.mean(axis=0)

    max_y = int(h * max_border_frac)
    max_x = int(w * max_border_frac)

    top = 0
    while top < max_y and row_density[top] >= density_thresh:
        top += 1

    bottom = h
    while bottom > h - max_y and row_density[bottom - 1] >= density_thresh:
        bottom -= 1

    left = 0
    while left < max_x and col_density[left] >= density_thresh:
        left += 1

    right = w
    while right > w - max_x and col_density[right - 1] >= density_thresh:
        right -= 1

    if bottom <= top + 20 or right <= left + 20:
        return gray
    return gray[top:bottom, left:right]


def apply_clahe(
    gray: np.ndarray,
    *,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Enhance local contrast on faded scans via CLAHE before binarization."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray)


# ---------------------------------------------------------------------------
# 3. Projection-Profile Deskewing
# ---------------------------------------------------------------------------

def _rotate_gray(gray: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate a grayscale image around its center with white border fill."""
    h, w = gray.shape
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )


def estimate_skew_angle(gray: np.ndarray, max_angle: float = 5.0) -> float:
    """Estimate page skew angle (degrees) via coarse-to-fine projection variance."""
    h, w = gray.shape
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, binary = cv2.threshold(small, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    sh, sw = binary.shape
    my, mx = max(1, int(sh * 0.05)), max(1, int(sw * 0.05))
    roi = binary[my : sh - my, mx : sw - mx]
    if roi.size == 0 or np.count_nonzero(roi) < 50:
        return 0.0

    def _score(angle: float) -> float:
        rh, rw = roi.shape
        mat = cv2.getRotationMatrix2D((rw / 2.0, rh / 2.0), angle, 1.0)
        rot = cv2.warpAffine(roi, mat, (rw, rh), flags=cv2.INTER_NEAREST, borderValue=0)
        row_sums = np.sum(rot, axis=1, dtype=np.float64)
        diffs = np.diff(row_sums)
        return float(np.dot(diffs, diffs))

    coarse_angles = np.arange(-max_angle, max_angle + 0.01, 0.5)
    best_angle = max(coarse_angles, key=_score)

    fine_angles = np.arange(best_angle - 0.5, best_angle + 0.51, 0.1)
    best_angle = max(fine_angles, key=_score)
    return float(round(best_angle, 2))


def deskew_image(
    gray: np.ndarray,
    *,
    max_angle: float = 5.0,
    min_angle: float = 0.15,
) -> Tuple[np.ndarray, float]:
    """Estimate and correct page skew. Returns ``(corrected_gray, angle_deg)``."""
    angle = estimate_skew_angle(gray, max_angle=max_angle)
    if abs(angle) < min_angle:
        return gray, 0.0
    return _rotate_gray(gray, angle), angle


# ---------------------------------------------------------------------------
# 4. Binarization (O(N) Integral Sauvola & Niblack + Otsu)
# ---------------------------------------------------------------------------

def _local_mean_std(gray: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact local mean and standard deviation in O(N) via integrals."""
    win = max(3, window_size | 1)
    half = win // 2

    padded = cv2.copyMakeBorder(gray, half, half, half, half, cv2.BORDER_REPLICATE)
    integral, sq_integral = cv2.integral2(padded)

    h, w = gray.shape
    area = float(win * win)

    sum_val = (
        integral[win : h + win, win : w + win]
        - integral[:h, win : w + win]
        - integral[win : h + win, :w]
        + integral[:h, :w]
    )
    sq_sum_val = (
        sq_integral[win : h + win, win : w + win]
        - sq_integral[:h, win : w + win]
        - sq_integral[win : h + win, :w]
        + sq_integral[:h, :w]
    )

    mean = sum_val / area
    variance = np.maximum(sq_sum_val / area - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def sauvola_binarize(
    gray: np.ndarray,
    *,
    window_size: int = 31,
    k: float = 0.20,
    r: float = 128.0,
) -> np.ndarray:
    """Sauvola adaptive thresholding: ``T = mean * (1 + k * (std / R - 1))``."""
    mean, std = _local_mean_std(gray, window_size)
    thresh = mean * (1.0 + k * (std / r - 1.0))
    return np.where(gray > thresh, 255, 0).astype(np.uint8)


def niblack_binarize(
    gray: np.ndarray,
    *,
    window_size: int = 31,
    k: float = -0.20,
    flat_paper_std: float = 3.0,
    flat_paper_mean: float = 200.0,
) -> np.ndarray:
    """Niblack adaptive thresholding: ``T = mean + k * std`` with flat-paper guard.

    Strong on severely faded ink while suppressing classic Niblack noise in
    blank bright paper margins where local variance is near zero.
    """
    mean, std = _local_mean_std(gray, window_size)
    thresh = mean + k * std
    is_paper = (gray > thresh) | ((std < flat_paper_std) & (mean > flat_paper_mean))
    return np.where(is_paper, 255, 0).astype(np.uint8)


def otsu_binarize(gray: np.ndarray) -> np.ndarray:
    """Global Otsu thresholding (effective on clean, evenly lit pages)."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def remove_tiny_speckles(binary: np.ndarray, min_area: int = 3) -> np.ndarray:
    """Remove isolated dust specks while preserving Burmese dots (့, း, ံ)."""
    if min_area <= 1:
        return binary

    inv = 255 - binary
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if num_labels <= 1:
        return binary

    cleaned = binary.copy()
    areas = stats[1:, cv2.CC_STAT_AREA]
    tiny_labels = np.where(areas < min_area)[0] + 1
    if tiny_labels.size > 0:
        mask = np.isin(labels, tiny_labels)
        cleaned[mask] = 255
    return cleaned


# ---------------------------------------------------------------------------
# 5. Dynamic Margin & Gap-Aware Header/Footer Cropping
# ---------------------------------------------------------------------------

def dynamic_margin_crop(
    img: np.ndarray,
    *,
    crop_header_footer: bool = False,
    pad_px: int = 24,
) -> np.ndarray:
    """Dynamically crop around the actual text block with clean white padding."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    h, w = mask.shape
    ink = mask < 128
    row_counts = ink.sum(axis=1)
    col_counts = ink.sum(axis=0)

    min_row_ink = max(3, int(w * 0.002))
    min_col_ink = max(3, int(h * 0.002))

    active_rows = np.where(row_counts >= min_row_ink)[0]
    active_cols = np.where(col_counts >= min_col_ink)[0]

    if active_rows.size == 0 or active_cols.size == 0:
        return img

    top, bottom = int(active_rows[0]), int(active_rows[-1]) + 1
    left, right = int(active_cols[0]), int(active_cols[-1]) + 1

    if crop_header_footer:
        body_h = bottom - top
        gap_min = max(18, int(body_h * 0.018))

        header_limit = top + int(body_h * 0.08)
        empty_run = 0
        for y in range(top, min(header_limit, bottom)):
            if row_counts[y] < min_row_ink:
                empty_run += 1
                if empty_run >= gap_min:
                    next_rows = active_rows[active_rows > y]
                    if next_rows.size > 0:
                        top = int(next_rows[0])
                    break
            else:
                empty_run = 0

        footer_limit = bottom - int(body_h * 0.10)
        empty_run = 0
        for y in range(bottom - 1, max(footer_limit, top), -1):
            if row_counts[y] < min_row_ink:
                empty_run += 1
                if empty_run >= gap_min:
                    prev_rows = active_rows[active_rows < y]
                    if prev_rows.size > 0:
                        bottom = int(prev_rows[-1]) + 1
                    break
            else:
                empty_run = 0

    cropped = img[top:bottom, left:right]
    if pad_px > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_px,
            pad_px,
            pad_px,
            pad_px,
            cv2.BORDER_CONSTANT,
            value=255,
        )
    return cropped


# ---------------------------------------------------------------------------
# 6. End-to-End Drop-In Pipeline for core/ocr_engine.py
# ---------------------------------------------------------------------------

def preprocess_page_image(
    pil_img: Image.Image,
    config: Optional[PreprocessConfig] = None,
) -> Image.Image:
    """Run full preprocessing pipeline on a rendered PIL page image."""
    cfg = config or PreprocessConfig()
    gray = np.array(pil_img.convert("L"), dtype=np.uint8)

    if cfg.strip_scanner_borders:
        gray = strip_black_scanner_borders(gray)

    if cfg.remove_shadows:
        gray = normalize_background(gray)

    if cfg.deskew:
        gray, _ = deskew_image(
            gray,
            max_angle=cfg.max_skew_angle,
            min_angle=cfg.min_skew_angle,
        )

    # Crop tightly to the text block BEFORE CLAHE/binarization so local
    # contrast enhancement focuses on text rather than rotated blank margins.
    if cfg.dynamic_crop:
        gray = dynamic_margin_crop(
            gray,
            crop_header_footer=cfg.crop_header_footer,
            pad_px=cfg.margin_pad_px,
        )

    if cfg.upscale > 1.0:
        new_w = int(round(gray.shape[1] * cfg.upscale))
        new_h = int(round(gray.shape[0] * cfg.upscale))
        gray = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    if cfg.apply_clahe:
        gray = apply_clahe(
            gray,
            clip_limit=cfg.clahe_clip_limit,
            tile_grid_size=cfg.clahe_tile_grid,
        )

    if cfg.output_mode == "sauvola":
        processed = sauvola_binarize(
            gray,
            window_size=cfg.sauvola_window,
            k=cfg.sauvola_k,
        )
        if cfg.denoise_speckles:
            processed = remove_tiny_speckles(processed, min_area=cfg.min_speckle_area)
    elif cfg.output_mode == "niblack":
        processed = niblack_binarize(
            gray,
            window_size=cfg.niblack_window,
            k=cfg.niblack_k,
        )
        if cfg.denoise_speckles:
            processed = remove_tiny_speckles(processed, min_area=cfg.min_speckle_area)
    elif cfg.output_mode == "otsu":
        processed = otsu_binarize(gray)
        if cfg.denoise_speckles:
            processed = remove_tiny_speckles(processed, min_area=cfg.min_speckle_area)
    else:
        processed = gray

    return Image.fromarray(processed, mode="L")


def render_and_preprocess_page(
    pdf_path: Union[str, Path],
    page_index: int,
    config: Optional[PreprocessConfig] = None,
) -> Image.Image:
    """Convenience helper: render a PDF page at 300 DPI and preprocess it."""
    cfg = config or PreprocessConfig()
    pil_img = render_pdf_page(pdf_path, page_index, dpi=cfg.dpi)
    return preprocess_page_image(pil_img, config=cfg)


# ---------------------------------------------------------------------------
# 7. Stratified A/B Preprocessing Grid Runner (Golden Set)
# ---------------------------------------------------------------------------

def build_ab_grid(
    *,
    modes: Sequence[BinarizeMode] = ("grayscale", "otsu", "sauvola", "niblack"),
    clahe_options: Sequence[bool] = (False, True),
    upscale_options: Sequence[float] = (1.0, 2.0),
    psm_options: Sequence[int] = (6,),
) -> List[GridRecipe]:
    """Generate candidate preprocessing + PSM recipes for A/B benchmarking."""
    recipes: List[GridRecipe] = []
    for mode, use_clahe, scale, psm in product(modes, clahe_options, upscale_options, psm_options):
        clahe_tag = "clahe" if use_clahe else "noclahe"
        scale_tag = f"{scale:.0f}x"
        name = f"{mode}_{clahe_tag}_{scale_tag}_psm{psm}"
        cfg = PreprocessConfig(
            dpi=300,
            apply_clahe=use_clahe,
            upscale=scale,
            output_mode=mode,
        )
        recipes.append(GridRecipe(name=name, config=cfg, psm=psm))
    return recipes


def make_tesseract_ocr_fn(
    *,
    tessdata_dir: Union[str, Path],
    lang: str = "mya_dhamma_v9_balanced",
) -> Callable[[Image.Image, int], str]:
    """Build an OCR callable ``(pil_img, psm) -> text`` (tesserocr or CLI)."""
    tess_dir = str(tessdata_dir)

    def _run(pil_img: Image.Image, psm: int) -> str:
        try:
            import tesserocr  # type: ignore

            with tesserocr.PyTessBaseAPI(path=tess_dir, lang=lang, psm=psm) as api:
                api.SetImage(pil_img)
                return api.GetUTF8Text() or ""
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
                    tess_dir,
                    "-l",
                    lang,
                    "--psm",
                    str(psm),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            return proc.stdout

    return _run


def run_preprocessing_ab_grid(
    manifest_path: Union[str, Path],
    *,
    ocr_fn: Callable[[Image.Image, int], str],
    recipes: Sequence[GridRecipe],
    save_report_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Benchmark preprocessing recipes across a stratified Golden Set manifest.

    Manifest entries must include ``page_id``, ``book_id``, ``stratum``,
    ``ref_txt``, and either ``image_path`` or (``pdf_path`` + ``page_index``).

    Returns a dictionary containing per-recipe results and the winning recipe
    for each stratum (e.g., ``clean`` vs ``faded``).
    """
    try:
        from scripts.eval_golden import aggregate_strata, score_page_texts  # type: ignore
    except ImportError:
        from eval_golden import aggregate_strata, score_page_texts  # type: ignore

    mpath = Path(manifest_path)
    base_dir = mpath.parent
    entries = json.loads(mpath.read_text(encoding="utf-8")).get("pages", [])
    if not entries:
        raise ValueError(f"No 'pages' found in manifest: {mpath}")

    loaded_pages: List[Dict[str, Any]] = []
    for entry in entries:
        ref_path = Path(entry["ref_txt"])
        if not ref_path.is_absolute():
            ref_path = base_dir / ref_path
        ref_text = ref_path.read_text(encoding="utf-8")

        if "image_path" in entry and entry["image_path"]:
            img_path = Path(entry["image_path"])
            if not img_path.is_absolute():
                img_path = base_dir / img_path
            pil_img = Image.open(img_path).convert("RGB")
        else:
            pdf_path = Path(entry["pdf_path"])
            if not pdf_path.is_absolute():
                pdf_path = base_dir / pdf_path
            pil_img = render_pdf_page(pdf_path, int(entry["page_index"]), dpi=300)

        loaded_pages.append(
            {
                "page_id": str(entry["page_id"]),
                "book_id": str(entry.get("book_id", "unknown")),
                "stratum": str(entry.get("stratum", "default")),
                "ref_text": ref_text,
                "pil_img": pil_img,
            }
        )

    recipe_results: List[Dict[str, Any]] = []
    best_by_stratum: Dict[str, Dict[str, Any]] = {}

    for recipe in recipes:
        page_scores = []
        for page in loaded_pages:
            proc_img = preprocess_page_image(page["pil_img"], config=recipe.config)
            hyp_text = ocr_fn(proc_img, recipe.psm)
            page_scores.append(
                score_page_texts(
                    page_id=page["page_id"],
                    book_id=page["book_id"],
                    stratum=page["stratum"],
                    ref_text=page["ref_text"],
                    hyp_text=hyp_text,
                )
            )

        strata_map, overall = aggregate_strata(page_scores)
        strata_dict = {k: asdict(v) for k, v in strata_map.items()}

        recipe_entry = {
            "recipe_name": recipe.name,
            "psm": recipe.psm,
            "config": asdict(recipe.config),
            "overall": asdict(overall),
            "strata": strata_dict,
        }
        recipe_results.append(recipe_entry)

        for stratum_name, stats in strata_dict.items():
            prev_best = best_by_stratum.get(stratum_name)
            if prev_best is None or stats["cer_pct"] < prev_best["cer_pct"]:
                best_by_stratum[stratum_name] = {
                    "recipe_name": recipe.name,
                    "cer_pct": stats["cer_pct"],
                    "wer_pct": stats["wer_pct"],
                    "exact_line_pct": stats["exact_line_pct"],
                    "psm": recipe.psm,
                    "config": asdict(recipe.config),
                }

    report = {
        "manifest": str(mpath),
        "recipes_tested": len(recipe_results),
        "best_by_stratum": best_by_stratum,
        "all_results": recipe_results,
    }

    if save_report_path is not None:
        out = Path(save_report_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return report


# ---------------------------------------------------------------------------
# Self-Test Suite
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    rng = np.random.default_rng(42)

    h, w = 1200, 850
    y_grad = np.linspace(215, 245, h, dtype=np.float32)[:, None]
    x_grad = np.linspace(-35, 0, w, dtype=np.float32)[None, :]
    canvas = np.clip(y_grad + x_grad, 150, 250).astype(np.uint8)

    for line_idx in range(18):
        y = 140 + line_idx * 48
        for x in range(130, 720, 22):
            cv2.rectangle(canvas, (x, y), (x + 14, y + 16), 35, thickness=-1)

    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 2.4, 1.0)
    tilted = cv2.warpAffine(canvas, mat, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=225)
    tilted[:, :12] = 15
    tilted[:12, :] = 15

    dust_y = rng.integers(50, h - 50, size=80)
    dust_x = rng.integers(50, w - 50, size=80)
    tilted[dust_y, dust_x] = 20

    # 1. Verify skew angle detection
    stripped = strip_black_scanner_borders(tilted)
    norm_bg = normalize_background(stripped)
    detected_angle = estimate_skew_angle(norm_bg, max_angle=5.0)
    assert abs(detected_angle - (-2.4)) <= 0.35

    # 2. Verify all 4 modes (sauvola, niblack, otsu, grayscale) + CLAHE + 2x upscale
    pil_in = Image.fromarray(tilted, mode="L")
    for mode in ("sauvola", "niblack", "otsu", "grayscale"):
        cfg = PreprocessConfig(
            apply_clahe=True,
            upscale=1.0,
            output_mode=mode,  # type: ignore[arg-type]
            dynamic_crop=True,
            margin_pad_px=20,
        )
        out = np.array(preprocess_page_image(pil_in, config=cfg))
        assert out.shape[0] < h and out.shape[1] < w, f"{mode} failed crop: {out.shape}"
        if mode != "grayscale":
            assert set(np.unique(out).tolist()).issubset({0, 255})

    out_1x = preprocess_page_image(pil_in, PreprocessConfig(upscale=1.0, output_mode="sauvola"))
    out_2x = preprocess_page_image(pil_in, PreprocessConfig(upscale=2.0, output_mode="sauvola"))
    assert out_2x.size[0] > out_1x.size[0] * 1.8

    # 3. Verify stratified A/B Grid Runner selects separate winners per stratum
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        clean_img = root / "clean.png"
        faded_img = root / "faded.png"
        pil_in.save(clean_img)
        pil_in.save(faded_img)

        ref_clean = "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ။\n"
        ref_faded = "ကစ္စာယန သဒ္ဒါကျမ်း။\n"
        (root / "clean.txt").write_text(ref_clean, encoding="utf-8")
        (root / "faded.txt").write_text(ref_faded, encoding="utf-8")

        manifest = {
            "version": "1.0",
            "pages": [
                {"page_id": "p_clean", "book_id": "b1", "stratum": "clean", "ref_txt": "clean.txt", "image_path": "clean.png"},
                {"page_id": "p_faded", "book_id": "b2", "stratum": "faded", "ref_txt": "faded.txt", "image_path": "faded.png"},
            ],
        }
        mpath = root / "manifest.json"
        mpath.write_text(json.dumps(manifest), encoding="utf-8")

        recipes = [
            GridRecipe("otsu_noclahe_1x_psm6", PreprocessConfig(output_mode="otsu", apply_clahe=False), psm=6),
            GridRecipe("niblack_clahe_1x_psm6", PreprocessConfig(output_mode="niblack", apply_clahe=True), psm=6),
        ]

        call_idx = {"n": 0}
        def _mock_ocr(_img: Image.Image, _psm: int) -> str:
            idx = call_idx["n"]
            call_idx["n"] += 1
            if idx == 0:
                return ref_clean
            if idx == 1:
                return "ကစ္စာယန သဒ္ဒါကျမ်။\n"
            if idx == 2:
                return "ဗုဒ္ဓ ဓမ္မ သင်္ဃ။\n"
            return ref_faded

        rep_path = root / "grid_report.json"
        report = run_preprocessing_ab_grid(
            mpath,
            ocr_fn=_mock_ocr,
            recipes=recipes,
            save_report_path=rep_path,
        )
        assert rep_path.exists()
        assert report["best_by_stratum"]["clean"]["recipe_name"] == "otsu_noclahe_1x_psm6"
        assert report["best_by_stratum"]["faded"]["recipe_name"] == "niblack_clahe_1x_psm6"

    print(
        f"All image_preprocessor + CLAHE/Niblack + A/B grid tests PASSED "
        f"(detected skew={detected_angle:+.2f} deg)."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Myanmar OCR Image Preprocessor & A/B Grid Runner")
    parser.add_argument("--self-test", action="store_true", help="Run built-in self-tests")
    parser.add_argument("--ab-grid", action="store_true", help="Run stratified A/B preprocessing grid")
    parser.add_argument("--manifest", type=Path, help="Golden Set manifest.json with image_path or pdf_path")
    parser.add_argument("--tessdata", type=Path, default=Path("tessdata"), help="Path to tessdata directory")
    parser.add_argument("--lang", type=str, default="mya_dhamma_v9_balanced", help="Tesseract model name")
    parser.add_argument("--fast-grid", action="store_true", help="Run fast 8-recipe grid (1x scale, PSM 6)")
    parser.add_argument("--save-grid-report", type=Path, default=None, help="Output JSON path for grid results")
    args = parser.parse_args(argv)

    if args.self_test or not args.ab_grid:
        _run_self_tests()
        return 0

    if not args.manifest:
        parser.error("--manifest is required with --ab-grid.")

    recipes = (
        build_ab_grid(upscale_options=(1.0,), psm_options=(6,))
        if args.fast_grid
        else build_ab_grid(upscale_options=(1.0, 2.0), psm_options=(6, 4))
    )
    ocr_fn = make_tesseract_ocr_fn(tessdata_dir=args.tessdata, lang=args.lang)
    report = run_preprocessing_ab_grid(
        args.manifest,
        ocr_fn=ocr_fn,
        recipes=recipes,
        save_report_path=args.save_grid_report,
    )

    print("\n=== Best Preprocessing Recipe per Stratum ===")
    for stratum, info in sorted(report["best_by_stratum"].items()):
        print(
            f"  {stratum:<12} -> {info['recipe_name']:<28} "
            f"CER={info['cer_pct']:.2f}%  WER={info['wer_pct']:.2f}%  ExactLn={info['exact_line_pct']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())