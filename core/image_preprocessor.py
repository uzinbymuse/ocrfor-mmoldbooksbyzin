"""300 DPI Rendering, Deskewing, Sauvola Binarization & Dynamic Crop Module.

Drop-in preprocessing module designed for ``core/ocr_engine.py`` in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Replaces fixed 180 DPI rendering, basic grayscale/autocontrast, and hard-coded
4.5% / 8.5% header-footer cropping with a document-adaptive pipeline tuned for
historical Burmese/Pali printed books:

1. Exact 300 DPI PDF rendering (PyMuPDF primary, pypdfium2 fallback).
2. Uneven paper illumination / bleed-through background normalization.
3. Pre-rotation dark scanner-border removal so deskewing never tilts black edges.
4. Coarse-to-fine projection-profile deskewing (accurate on dense text lines).
5. Exact O(N) integral-image Sauvola adaptive binarization (no scikit-image
   dependency required).
6. Dynamic content-box margin cropping with optional gap-aware running
   header/footer trimming.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for Myanmar historical book page preprocessing."""

    dpi: int = 300
    remove_shadows: bool = True
    strip_scanner_borders: bool = True
    deskew: bool = True
    max_skew_angle: float = 5.0
    min_skew_angle: float = 0.15
    output_mode: Literal["sauvola", "grayscale"] = "sauvola"
    sauvola_window: int = 31
    sauvola_k: float = 0.20
    denoise_speckles: bool = True
    min_speckle_area: int = 3
    dynamic_crop: bool = True
    crop_header_footer: bool = False
    margin_pad_px: int = 24


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
# 2. Background Illumination Normalization & Scanner-Border Removal
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


def estimate_skew_angle(
    gray: np.ndarray,
    max_angle: float = 5.0,
) -> float:
    """Estimate page skew angle (degrees) via coarse-to-fine projection variance.

    Positive return value means the image should be rotated counter-clockwise
    by that angle (as accepted by ``cv2.getRotationMatrix2D``).
    """
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
# 4. Exact O(N) Integral-Image Sauvola Adaptive Binarization
# ---------------------------------------------------------------------------

def sauvola_binarize(
    gray: np.ndarray,
    *,
    window_size: int = 31,
    k: float = 0.20,
    r: float = 128.0,
) -> np.ndarray:
    """Apply Sauvola adaptive thresholding in O(N) time using OpenCV integrals.

    Formula:
        T(x, y) = mean(x, y) * (1 + k * (std(x, y) / R - 1))

    Preserves fine Burmese diacritics (လုံးကြီးတင်, အောက်ကမြစ်, ဝစ္စပေါက်)
    on uneven or aged scans far better than global Otsu thresholding.
    """
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
    std = np.sqrt(variance)

    thresh = mean * (1.0 + k * (std / r - 1.0))
    return np.where(gray > thresh, 255, 0).astype(np.uint8)


def remove_tiny_speckles(binary: np.ndarray, min_area: int = 3) -> np.ndarray:
    """Remove isolated dust specks while preserving Burmese dots (့, း, ံ).

    At 300 DPI, Burmese dot diacritics typically span 12–40+ px²; keeping
    ``min_area <= 4`` safely removes 1–2 px scan noise without touching dots.
    """
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
    """Dynamically crop around the actual text block with clean white padding.

    If ``crop_header_footer=True``, trims a thin isolated top/bottom band only
    when separated from the main text body by a clear whitespace gap (preventing
    accidental clipping of body lines on pages with shifted margins).
    """
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
    """Run full preprocessing pipeline on a rendered PIL page image.

    Returns a clean ``PIL.Image.Image`` (mode ``'L'``) ready for ``tesserocr``
    or Tesseract CLI.
    """
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

    if cfg.output_mode == "sauvola":
        processed = sauvola_binarize(
            gray,
            window_size=cfg.sauvola_window,
            k=cfg.sauvola_k,
        )
        if cfg.denoise_speckles:
            processed = remove_tiny_speckles(processed, min_area=cfg.min_speckle_area)
    else:
        processed = gray

    if cfg.dynamic_crop:
        processed = dynamic_margin_crop(
            processed,
            crop_header_footer=cfg.crop_header_footer,
            pad_px=cfg.margin_pad_px,
        )

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
# Self-Test Suite (Run directly: python3 image_preprocessor.py)
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

    # Add 12px black scanner border on left and top
    tilted[:, :12] = 15
    tilted[:12, :] = 15

    # Add light 1px speckle dust
    dust_y = rng.integers(50, h - 50, size=80)
    dust_x = rng.integers(50, w - 50, size=80)
    tilted[dust_y, dust_x] = 20

    # 1. Verify skew angle estimation accuracy
    stripped = strip_black_scanner_borders(tilted)
    norm_bg = normalize_background(stripped)
    detected_angle = estimate_skew_angle(norm_bg, max_angle=5.0)
    assert abs(detected_angle - (-2.4)) <= 0.35, (
        f"Expected skew angle near -2.4 deg, got {detected_angle}"
    )

    # 2. Verify full end-to-end preprocessing pipeline
    pil_in = Image.fromarray(tilted, mode="L")
    cfg = PreprocessConfig(dpi=300, output_mode="sauvola", dynamic_crop=True, margin_pad_px=20)
    pil_out = preprocess_page_image(pil_in, config=cfg)
    arr_out = np.array(pil_out)

    assert pil_out.mode == "L"
    unique_vals = set(np.unique(arr_out).tolist())
    assert unique_vals.issubset({0, 255}), f"Expected binary values, got {unique_vals}"
    assert arr_out.shape[0] < h and arr_out.shape[1] < w, (
        f"Expected cropped dimensions < ({h}, {w}), got {arr_out.shape}"
    )
    assert arr_out[0, :].mean() == 255.0 and arr_out[:, 0].mean() == 255.0

    print(
        f"All preprocessing tests PASSED "
        f"(detected skew={detected_angle:+.2f} deg, output shape={arr_out.shape})."
    )


if __name__ == "__main__":
    _run_self_tests()