"""Confidence Extraction, Web Studio Highlighting & HITL Ground Truth Exporter.

Drop-in module designed for ``core/`` and ``apps/web_app.py`` in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Connects OCR inference directly to the Tesseract 5 fine-tuning pipeline
(``scripts/train_mya_dhamma.sh``):

1. Extracts line & word bounding boxes + confidence scores from ``tesserocr``
   (with a ``tesseract ... tsv`` CLI fallback parser).
2. Builds JSON payloads and HTML highlight spans for ``apps/web_app.py``'s
   side-by-side Page Inspector (flagging low-confidence words for fast QC).
3. Exports human-verified lines as padded ``gt/*.png`` (12% vertical padding
   for Burmese diacritics & subscripts) + UTN #11 normalized ``gt/*.gt.txt``
   pairs, plus a ``manifest.jsonl`` audit log.
"""

from __future__ import annotations

import csv
import html
import io
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

# Optional integration with our UTN #11 normalizer when placed in core/
try:
    from core.myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
except ImportError:
    try:
        from myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
    except ImportError:
        def normalize_myanmar_unicode(text: str, **_: Any) -> str:  # type: ignore
            return unicodedata.normalize("NFC", text).strip()


BBox = Tuple[int, int, int, int]  # (x1, y1, x2, y2) in page pixels

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9_-]+")


@dataclass
class WordSpan:
    """Single OCR word/token with confidence score and bounding box."""

    text: str
    confidence: float
    bbox: BBox
    severity: str  # "ok" | "warn" | "low"


@dataclass
class LineResult:
    """Single OCR text line with bounding box, confidence, and word spans."""

    line_index: int
    text: str
    mean_confidence: float
    min_confidence: float
    bbox: BBox
    words: List[WordSpan]
    needs_review: bool


def classify_confidence(
    confidence: float,
    *,
    low_thresh: float = 70.0,
    warn_thresh: float = 85.0,
) -> str:
    """Map numeric Tesseract confidence (0-100) to UI severity tier."""
    if confidence < low_thresh:
        return "low"
    if confidence < warn_thresh:
        return "warn"
    return "ok"


# ---------------------------------------------------------------------------
# 1. Confidence & Bounding-Box Extraction (tesserocr + TSV Fallback)
# ---------------------------------------------------------------------------

def extract_lines_with_tesserocr(
    pil_img: Image.Image,
    *,
    tessdata_dir: Union[str, Path],
    lang: str = "mya_dhamma_v9_balanced",
    psm: int = 6,
    low_thresh: float = 70.0,
    warn_thresh: float = 85.0,
) -> List[LineResult]:
    """Run ``tesserocr`` in-memory and return structured line & word confidences."""
    import tesserocr  # type: ignore

    results: List[LineResult] = []
    with tesserocr.PyTessBaseAPI(
        path=str(tessdata_dir),
        lang=lang,
        psm=psm,
    ) as api:
        api.SetImage(pil_img)
        api.Recognize()
        ri = api.GetIterator()
        if ri is None:
            return results

        line_idx = 0
        line_level = tesserocr.RIL.TEXTLINE
        word_level = tesserocr.RIL.WORD

        while True:
            raw_line = (ri.GetUTF8Text(line_level) or "").strip()
            line_bbox = ri.BoundingBox(line_level)
            if raw_line and line_bbox is not None:
                words: List[WordSpan] = []
                # Walk words inside the current textline
                while True:
                    w_text = (ri.GetUTF8Text(word_level) or "").strip()
                    w_conf = float(ri.Confidence(word_level))
                    w_box = ri.BoundingBox(word_level)
                    if w_text and w_box is not None:
                        words.append(
                            WordSpan(
                                text=normalize_myanmar_unicode(w_text),
                                confidence=round(w_conf, 2),
                                bbox=(int(w_box[0]), int(w_box[1]), int(w_box[2]), int(w_box[3])),
                                severity=classify_confidence(
                                    w_conf,
                                    low_thresh=low_thresh,
                                    warn_thresh=warn_thresh,
                                ),
                            )
                        )
                    if ri.IsAtFinalElement(line_level, word_level):
                        break
                    if not ri.Next(word_level):
                        break

                confs = [w.confidence for w in words] or [float(ri.Confidence(line_level))]
                mean_c = round(sum(confs) / len(confs), 2)
                min_c = round(min(confs), 2)
                results.append(
                    LineResult(
                        line_index=line_idx,
                        text=normalize_myanmar_unicode(raw_line),
                        mean_confidence=mean_c,
                        min_confidence=min_c,
                        bbox=(
                            int(line_bbox[0]),
                            int(line_bbox[1]),
                            int(line_bbox[2]),
                            int(line_bbox[3]),
                        ),
                        words=words,
                        needs_review=(min_c < warn_thresh),
                    )
                )
                line_idx += 1

            if not ri.Next(line_level):
                break

    return results


def parse_tesseract_tsv(
    tsv_text: str,
    *,
    low_thresh: float = 70.0,
    warn_thresh: float = 85.0,
) -> List[LineResult]:
    """Parse ``tesseract <img.png> stdout --psm 6 tsv`` output into ``LineResult``s.

    Provides exact parity with ``extract_lines_with_tesserocr`` when running
    via Tesseract CLI subprocess fallback.
    """
    reader = csv.DictReader(io.StringIO(tsv_text), delimiter="\t")
    grouped: Dict[Tuple[int, int, int, int], List[WordSpan]] = {}

    for row in reader:
        if row.get("level") != "5":  # level 5 = word
            continue
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            conf = float(row["conf"])
            left = int(row["left"])
            top = int(row["top"])
            width = int(row["width"])
            height = int(row["height"])
            key = (
                int(row["page_num"]),
                int(row["block_num"]),
                int(row["par_num"]),
                int(row["line_num"]),
            )
        except (KeyError, ValueError):
            continue

        if conf < 0:
            continue

        span = WordSpan(
            text=normalize_myanmar_unicode(text),
            confidence=round(conf, 2),
            bbox=(left, top, left + width, top + height),
            severity=classify_confidence(conf, low_thresh=low_thresh, warn_thresh=warn_thresh),
        )
        grouped.setdefault(key, []).append(span)

    results: List[LineResult] = []
    for line_idx, (_, words) in enumerate(grouped.items()):
        x1 = min(w.bbox[0] for w in words)
        y1 = min(w.bbox[1] for w in words)
        x2 = max(w.bbox[2] for w in words)
        y2 = max(w.bbox[3] for w in words)
        confs = [w.confidence for w in words]
        mean_c = round(sum(confs) / len(confs), 2)
        min_c = round(min(confs), 2)
        joined = normalize_myanmar_unicode(" ".join(w.text for w in words))

        results.append(
            LineResult(
                line_index=line_idx,
                text=joined,
                mean_confidence=mean_c,
                min_confidence=min_c,
                bbox=(x1, y1, x2, y2),
                words=words,
                needs_review=(min_c < warn_thresh),
            )
        )

    return results


# ---------------------------------------------------------------------------
# 2. Web Studio Highlighting & JSON Serialization (apps/web_app.py)
# ---------------------------------------------------------------------------

def render_line_highlight_html(line: LineResult) -> str:
    """Render a line as safe HTML with per-word confidence highlight spans.

    CSS classes expected in ``apps/web_app.py``:
      - ``.conf-low``  -> soft red background (e.g. ``#fee2e2``)
      - ``.conf-warn`` -> soft amber background (e.g. ``#fef3c7``)
      - ``.conf-ok``   -> transparent / normal text
    """
    parts: List[str] = []
    for word in line.words:
        escaped = html.escape(word.text)
        parts.append(
            f'<span class="ocr-word conf-{word.severity}" '
            f'data-conf="{word.confidence:.1f}" '
            f'title="Confidence: {word.confidence:.1f}%">{escaped}</span>'
        )
    return " ".join(parts)


def serialize_page_qc(
    lines: Sequence[LineResult],
    *,
    page_no: int,
    image_size: Tuple[int, int],
) -> Dict[str, Any]:
    """Build a JSON-serializable QC payload for ``apps/web_app.py``."""
    all_words = [w for line in lines for w in line.words]
    avg_conf = (
        round(sum(w.confidence for w in all_words) / len(all_words), 2)
        if all_words
        else 0.0
    )

    return {
        "page_no": page_no,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "summary": {
            "total_lines": len(lines),
            "review_lines": sum(1 for ln in lines if ln.needs_review),
            "low_conf_words": sum(1 for w in all_words if w.severity == "low"),
            "warn_conf_words": sum(1 for w in all_words if w.severity == "warn"),
            "avg_confidence": avg_conf,
        },
        "lines": [
            {
                **asdict(ln),
                "highlight_html": render_line_highlight_html(ln),
            }
            for ln in lines
        ],
    }


# ---------------------------------------------------------------------------
# 3. Human-in-the-Loop Ground Truth Exporter (gt/*.png + gt/*.gt.txt)
# ---------------------------------------------------------------------------

def crop_line_with_padding(
    page_img: Image.Image,
    bbox: BBox,
    *,
    vertical_pad_ratio: float = 0.12,
    min_vertical_pad_px: int = 6,
    horizontal_pad_px: int = 10,
) -> Image.Image:
    """Crop a text line with 12% vertical padding to preserve Burmese diacritics."""
    w, h = page_img.size
    x1, y1, x2, y2 = bbox
    line_h = max(1, y2 - y1)
    v_pad = max(min_vertical_pad_px, int(round(line_h * vertical_pad_ratio)))

    cx1 = max(0, x1 - horizontal_pad_px)
    cy1 = max(0, y1 - v_pad)
    cx2 = min(w, x2 + horizontal_pad_px)
    cy2 = min(h, y2 + v_pad)

    return page_img.crop((cx1, cy1, cx2, cy2))


def export_verified_line_to_gt(
    page_img: Image.Image,
    *,
    bbox: BBox,
    corrected_text: str,
    gt_dir: Union[str, Path],
    book_id: str,
    page_no: int,
    line_index: int,
    original_ocr_text: str = "",
    original_confidence: Optional[float] = None,
    vertical_pad_ratio: float = 0.12,
) -> Dict[str, Any]:
    """Export a proofread line as ``<stem>.png`` + ``<stem>.gt.txt`` for training.

    Writes directly in the ``gt/*.png`` + ``gt/*.gt.txt`` format consumed by
    ``scripts/train_mya_dhamma.sh`` and appends an audit record to
    ``manifest.jsonl``.
    """
    clean_text = normalize_myanmar_unicode(corrected_text).replace("\n", " ").strip()
    if not clean_text:
        raise ValueError("corrected_text is empty after normalization.")

    safe_book = _SAFE_ID_RE.sub("_", book_id.strip()).strip("_") or "book"
    stem = f"{safe_book}_p{page_no:04d}_l{line_index:03d}"

    out_dir = Path(gt_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_crop = crop_line_with_padding(
        page_img,
        bbox,
        vertical_pad_ratio=vertical_pad_ratio,
    )

    png_path = out_dir / f"{stem}.png"
    txt_path = out_dir / f"{stem}.gt.txt"

    line_crop.save(png_path, format="PNG")
    txt_path.write_text(clean_text + "\n", encoding="utf-8")

    record = {
        "stem": stem,
        "book_id": safe_book,
        "page_no": page_no,
        "line_index": line_index,
        "bbox": list(bbox),
        "crop_size": list(line_crop.size),
        "original_ocr_text": original_ocr_text,
        "corrected_text": clean_text,
        "original_confidence": original_confidence,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = out_dir / "manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "saved",
        "stem": stem,
        "png_path": str(png_path),
        "gt_txt_path": str(txt_path),
        "normalized_text": clean_text,
        "crop_size": list(line_crop.size),
    }


def handle_save_gt_payload(
    payload: Dict[str, Any],
    *,
    page_img: Image.Image,
    gt_dir: Union[str, Path],
) -> Dict[str, Any]:
    """Helper for ``apps/web_app.py`` ``POST /api/save_gt`` JSON handler."""
    bbox_raw = payload["bbox"]
    if len(bbox_raw) != 4:
        raise ValueError("bbox must contain 4 integers: [x1, y1, x2, y2]")
    bbox: BBox = (int(bbox_raw[0]), int(bbox_raw[1]), int(bbox_raw[2]), int(bbox_raw[3]))

    return export_verified_line_to_gt(
        page_img,
        bbox=bbox,
        corrected_text=str(payload["corrected_text"]),
        gt_dir=gt_dir,
        book_id=str(payload.get("book_id", "dhamma")),
        page_no=int(payload["page_no"]),
        line_index=int(payload["line_index"]),
        original_ocr_text=str(payload.get("original_ocr_text", "")),
        original_confidence=(
            float(payload["original_confidence"])
            if payload.get("original_confidence") is not None
            else None
        ),
    )


# ---------------------------------------------------------------------------
# Self-Test Suite (Run directly: python3 hitl_confidence.py)
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    import tempfile

    sample_tsv = (
        "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        "5\t1\t1\t1\t1\t1\t100\t120\t120\t40\t95.4\tဗုဒ္ဓ\n"
        "5\t1\t1\t1\t1\t2\t230\t120\t140\t40\t78.2\tဓမ္မ\n"
        "5\t1\t1\t1\t1\t3\t380\t120\t130\t40\t62.0\tသင်္ဃာ\n"
    )

    lines = parse_tesseract_tsv(sample_tsv, low_thresh=70.0, warn_thresh=85.0)
    assert len(lines) == 1
    ln = lines[0]
    assert ln.needs_review is True
    assert ln.bbox == (100, 120, 510, 160)
    assert [w.severity for w in ln.words] == ["ok", "warn", "low"]

    # Check HTML highlighting & QC payload
    qc_payload = serialize_page_qc(lines, page_no=17, image_size=(1200, 1600))
    assert qc_payload["summary"]["low_conf_words"] == 1
    assert qc_payload["summary"]["warn_conf_words"] == 1
    assert 'class="ocr-word conf-low"' in qc_payload["lines"][0]["highlight_html"]

    # Check HITL export (gt/*.png + gt/*.gt.txt + manifest.jsonl)
    page_img = Image.new("L", (1200, 1600), color=255)
    with tempfile.TemporaryDirectory() as tmpdir:
        res = handle_save_gt_payload(
            {
                "book_id": "vol1_dhamma",
                "page_no": 17,
                "line_index": 0,
                "bbox": [100, 120, 510, 160],
                "original_ocr_text": "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ",
                "corrected_text": "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ။",
                "original_confidence": 78.5,
            },
            page_img=page_img,
            gt_dir=tmpdir,
        )
        png_file = Path(res["png_path"])
        txt_file = Path(res["gt_txt_path"])
        manifest_file = Path(tmpdir) / "manifest.jsonl"

        assert png_file.exists() and txt_file.exists() and manifest_file.exists()
        # 40px line height + 6px min vertical pad top & bottom -> 52px height
        with Image.open(png_file) as crop_im:
            assert crop_im.size == (430, 52), f"Unexpected crop size: {crop_im.size}"
        assert txt_file.read_text(encoding="utf-8") == "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ။\n"

    print("All HITL confidence & Ground Truth export tests PASSED.")


if __name__ == "__main__":
    _run_self_tests()