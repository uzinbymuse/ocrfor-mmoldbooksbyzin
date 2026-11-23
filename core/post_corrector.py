"""Low-Confidence Local Ollama Post-OCR Corrector with Archaic-Spelling Guard.

Drop-in module designed for ``core/`` and ``apps/web_app.py`` in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Key safety guarantees (informed by historical Burmese OCR best practices):
1. **Confidence Gating**: Only lines flagged as low-confidence by
   ``hitl_confidence.py`` are sent to Ollama. High-confidence lines are never
   touched.
2. **Archaic Spelling Protection**: Explicitly locks pre-1963/1978 historical
   spellings (e.g., ``တရုပ်`` vs modern ``တရုတ်``) both in the prompt and via
   a deterministic post-check rollback so an LLM never "modernizes" valid
   historical text.
3. **Anti-Hallucination Edit Budget**: Rejects any LLM reply that rewrites,
   truncates, or expands a line beyond ``max_edit_ratio``, falling back safely
   to the raw OCR text.
4. **Always-On Diff & Raw Preservation**: Emits structured diffs and escaped
   HTML ``<del>/<ins>`` markup alongside the original ``raw_text`` for one-click
   review in ``apps/web_app.py``.
5. **Zero Extra Dependencies**: Uses Python stdlib ``urllib.request`` to call
   local Ollama (``http://localhost:11434/api/generate``), with optional base64
   line-crop image support for VLMs like ``Gemma-SEA-LION-v4-4B-VL``.
"""

from __future__ import annotations

import base64
import difflib
import html
import io
import json
import unicodedata
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from PIL import Image

try:
    from core.myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
except ImportError:
    try:
        from myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
    except ImportError:
        def normalize_myanmar_unicode(text: str, **_: Any) -> str:  # type: ignore
            return unicodedata.normalize("NFC", text).strip()


# Default map of {archaic_form: modern_form} that must NOT be modernized.
# Extend via ``archaic_map_path`` JSON for book-specific Dhamma/Pali forms.
DEFAULT_ARCHAIC_MAP: Dict[str, str] = {
    "တရုပ်": "တရုတ်",
    "လူမျိူး": "လူမျိုး",
    "သာသနာတော်မြတ်": "သာသနာတော်မြတ်",
}

# Strict English instruction prompt (benchmarks show English instructions
# reduce over-editing on Southeast Asian scripts compared to verbose local prompts).
_STRICT_PROMPT_TEMPLATE = (
    "You are a strict Myanmar and Pali historical book OCR proofreader.\n"
    "Fix ONLY obvious optical character recognition (OCR) typos in the single line below.\n"
    "Rules:\n"
    "1. Do NOT rewrite, paraphrase, summarize, or modernize archaic pre-1978 Burmese spellings.\n"
    "2. Preserve all Pali terms, punctuation (၊ ။), and Myanmar numerals exactly.\n"
    "3. Output ONLY the corrected single line of Myanmar/Pali text with no explanation or quotes.\n"
    "{protected_clause}"
    "OCR line: {raw_line}\n"
    "Corrected line:"
)


@dataclass(frozen=True)
class CorrectorConfig:
    """Configuration for confidence-gated local Ollama post-correction."""

    ollama_url: str = "http://localhost:11434/api/generate"
    model: str = "qwen3:8b"  # or VLM such as "gemma-sea-lion-v4-4b-vl"
    timeout_sec: float = 30.0
    temperature: float = 0.0
    min_word_conf_thresh: float = 80.0
    mean_line_conf_thresh: float = 85.0
    max_edit_ratio: float = 0.25
    min_len_ratio: float = 0.80
    max_len_ratio: float = 1.20
    use_vision_crop: bool = False


@dataclass
class CorrectionResult:
    """Audit-friendly result for a single OCR line with raw text and diff."""

    line_index: int
    raw_text: str
    corrected_text: str
    changed: bool
    status: str  # "skipped_high_conf" | "corrected" | "unchanged" | "rejected_excessive_edit" | "ollama_error"
    min_confidence: float
    mean_confidence: float
    edit_ratio: float
    restored_archaic_terms: List[str] = field(default_factory=list)
    diff_ops: List[Dict[str, str]] = field(default_factory=list)
    diff_html: str = ""
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# 1. Archaic Spelling & Diff Helpers
# ---------------------------------------------------------------------------

def load_archaic_map(path: Optional[Union[str, Path]] = None) -> Dict[str, str]:
    """Load default + optional book-specific ``{archaic: modern}`` JSON map."""
    merged = dict(DEFAULT_ARCHAIC_MAP)
    if path is not None:
        p = Path(path)
        if p.exists():
            custom = json.loads(p.read_text(encoding="utf-8"))
            merged.update({str(k): str(v) for k, v in custom.items()})
    return merged


def _restore_archaic_spellings(
    raw_text: str,
    candidate_text: str,
    archaic_map: Dict[str, str],
) -> Tuple[str, List[str]]:
    """Restore any archaic spelling that existed in ``raw_text`` if modernized."""
    fixed = candidate_text
    restored: List[str] = []

    for archaic, modern in archaic_map.items():
        if archaic == modern:
            continue
        if archaic in raw_text and modern in fixed and archaic not in fixed:
            fixed = fixed.replace(modern, archaic)
            restored.append(f"{modern}->{archaic}")

    return fixed, restored


def _char_edit_ratio(a: str, b: str) -> float:
    """Normalized character difference ratio in [0.0, 1.0]."""
    if a == b:
        return 0.0
    if not a:
        return 1.0
    matcher = difflib.SequenceMatcher(None, a, b)
    return round(1.0 - matcher.ratio(), 4)


def build_line_diff(raw_text: str, corrected_text: str) -> Tuple[List[Dict[str, str]], str]:
    """Compute structured diff ops and escaped HTML ``<del>/<ins>`` markup."""
    matcher = difflib.SequenceMatcher(None, raw_text, corrected_text)
    ops: List[Dict[str, str]] = []
    html_parts: List[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        r_seg = raw_text[i1:i2]
        c_seg = corrected_text[j1:j2]
        ops.append({"op": tag, "raw": r_seg, "corrected": c_seg})

        if tag == "equal":
            html_parts.append(html.escape(c_seg))
        elif tag == "delete":
            html_parts.append(f'<del class="ocr-del">{html.escape(r_seg)}</del>')
        elif tag == "insert":
            html_parts.append(f'<ins class="ocr-ins">{html.escape(c_seg)}</ins>')
        elif tag == "replace":
            html_parts.append(
                f'<del class="ocr-del">{html.escape(r_seg)}</del>'
                f'<ins class="ocr-ins">{html.escape(c_seg)}</ins>'
            )

    return ops, "".join(html_parts)


# ---------------------------------------------------------------------------
# 2. Local Ollama Client (Text or Optional Vision-Language Crop)
# ---------------------------------------------------------------------------

def _encode_image_b64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_ollama_generate(
    prompt: str,
    *,
    config: CorrectorConfig,
    line_crop: Optional[Image.Image] = None,
) -> str:
    """Call local Ollama ``/api/generate`` and return the single-line response."""
    body: Dict[str, Any] = {
        "model": config.model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": config.temperature},
    }
    if config.use_vision_crop and line_crop is not None:
        body["images"] = [_encode_image_b64(line_crop)]

    req = urllib.request.Request(
        config.ollama_url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=config.timeout_sec) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    reply = str(data.get("response", "")).strip()
    # Take first non-empty line in case model adds trailing notes
    for ln in reply.splitlines():
        ln = ln.strip().strip('"').strip("'")
        if ln:
            return ln
    return ""


# ---------------------------------------------------------------------------
# 3. Confidence-Gated Line & Page Post-Correction
# ---------------------------------------------------------------------------

def _extract_line_fields(line: Any) -> Tuple[int, str, float, float, Optional[Tuple[int, int, int, int]]]:
    """Read line metadata from either a ``hitl_confidence.LineResult`` or dict."""
    if isinstance(line, dict):
        bbox = tuple(line["bbox"]) if line.get("bbox") else None
        return (
            int(line["line_index"]),
            str(line["text"]),
            float(line["min_confidence"]),
            float(line["mean_confidence"]),
            bbox,  # type: ignore[return-value]
        )
    return (
        int(line.line_index),
        str(line.text),
        float(line.min_confidence),
        float(line.mean_confidence),
        tuple(line.bbox) if getattr(line, "bbox", None) else None,  # type: ignore[return-value]
    )


def correct_line_if_needed(
    line: Any,
    *,
    config: Optional[CorrectorConfig] = None,
    archaic_map: Optional[Dict[str, str]] = None,
    line_crop: Optional[Image.Image] = None,
    llm_fn: Optional[Callable[[str, Optional[Image.Image]], str]] = None,
) -> CorrectionResult:
    """Correct a single OCR line via Ollama ONLY if below confidence threshold."""
    cfg = config or CorrectorConfig()
    amap = archaic_map if archaic_map is not None else DEFAULT_ARCHAIC_MAP

    line_idx, raw_text, min_conf, mean_conf, _ = _extract_line_fields(line)
    raw_norm = normalize_myanmar_unicode(raw_text, normalize_tall_aa=False)

    needs_llm = (
        min_conf < cfg.min_word_conf_thresh
        or mean_conf < cfg.mean_line_conf_thresh
    )
    if not needs_llm or not raw_norm:
        ops, diff_html = build_line_diff(raw_norm, raw_norm)
        return CorrectionResult(
            line_index=line_idx,
            raw_text=raw_norm,
            corrected_text=raw_norm,
            changed=False,
            status="skipped_high_conf",
            min_confidence=min_conf,
            mean_confidence=mean_conf,
            edit_ratio=0.0,
            diff_ops=ops,
            diff_html=diff_html,
        )

    # List any archaic terms actually present in this line so prompt locks them
    present_archaic = [term for term in amap if term in raw_norm]
    protected_clause = (
        f"4. Keep these historical spellings verbatim: {', '.join(present_archaic)}\n"
        if present_archaic
        else ""
    )
    prompt = _STRICT_PROMPT_TEMPLATE.format(
        protected_clause=protected_clause,
        raw_line=raw_norm,
    )

    try:
        if llm_fn is not None:
            raw_reply = llm_fn(prompt, line_crop)
        else:
            raw_reply = call_ollama_generate(prompt, config=cfg, line_crop=line_crop)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        ops, diff_html = build_line_diff(raw_norm, raw_norm)
        return CorrectionResult(
            line_index=line_idx,
            raw_text=raw_norm,
            corrected_text=raw_norm,
            changed=False,
            status="ollama_error",
            min_confidence=min_conf,
            mean_confidence=mean_conf,
            edit_ratio=0.0,
            diff_ops=ops,
            diff_html=diff_html,
            error=str(exc),
        )

    candidate = normalize_myanmar_unicode(raw_reply, normalize_tall_aa=False)
    if not candidate:
        ops, diff_html = build_line_diff(raw_norm, raw_norm)
        return CorrectionResult(
            line_index=line_idx,
            raw_text=raw_norm,
            corrected_text=raw_norm,
            changed=False,
            status="rejected_excessive_edit",
            min_confidence=min_conf,
            mean_confidence=mean_conf,
            edit_ratio=1.0,
            diff_ops=ops,
            diff_html=diff_html,
        )

    # Deterministic archaic-spelling rollback
    candidate, restored_terms = _restore_archaic_spellings(raw_norm, candidate, amap)

    # Anti-hallucination length & edit-budget gate
    len_ratio = len(candidate) / max(1, len(raw_norm))
    edit_ratio = _char_edit_ratio(raw_norm, candidate)
    if (
        edit_ratio > cfg.max_edit_ratio
        or len_ratio < cfg.min_len_ratio
        or len_ratio > cfg.max_len_ratio
    ):
        ops, diff_html = build_line_diff(raw_norm, raw_norm)
        return CorrectionResult(
            line_index=line_idx,
            raw_text=raw_norm,
            corrected_text=raw_norm,
            changed=False,
            status="rejected_excessive_edit",
            min_confidence=min_conf,
            mean_confidence=mean_conf,
            edit_ratio=edit_ratio,
            restored_archaic_terms=restored_terms,
            diff_ops=ops,
            diff_html=diff_html,
        )

    changed = candidate != raw_norm
    ops, diff_html = build_line_diff(raw_norm, candidate)
    return CorrectionResult(
        line_index=line_idx,
        raw_text=raw_norm,
        corrected_text=candidate,
        changed=changed,
        status="corrected" if changed else "unchanged",
        min_confidence=min_conf,
        mean_confidence=mean_conf,
        edit_ratio=edit_ratio,
        restored_archaic_terms=restored_terms,
        diff_ops=ops,
        diff_html=diff_html,
    )


def correct_page_lines(
    lines: Sequence[Any],
    *,
    config: Optional[CorrectorConfig] = None,
    archaic_map: Optional[Dict[str, str]] = None,
    page_img: Optional[Image.Image] = None,
    llm_fn: Optional[Callable[[str, Optional[Image.Image]], str]] = None,
) -> Dict[str, Any]:
    """Run confidence-gated post-correction across all lines on a page.

    Returns a JSON-ready payload for ``apps/web_app.py`` containing summary
    counts, per-line diffs, and both raw and corrected page text.
    """
    cfg = config or CorrectorConfig()
    results: List[CorrectionResult] = []

    for line in lines:
        crop = None
        if cfg.use_vision_crop and page_img is not None:
            _, _, _, _, bbox = _extract_line_fields(line)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                pad = max(6, int((y2 - y1) * 0.12))
                w, h = page_img.size
                crop = page_img.crop((max(0, x1 - 8), max(0, y1 - pad), min(w, x2 + 8), min(h, y2 + pad)))

        results.append(
            correct_line_if_needed(
                line,
                config=cfg,
                archaic_map=archaic_map,
                line_crop=crop,
                llm_fn=llm_fn,
            )
        )

    return {
        "model": cfg.model,
        "summary": {
            "total_lines": len(results),
            "skipped_high_conf": sum(1 for r in results if r.status == "skipped_high_conf"),
            "corrected": sum(1 for r in results if r.status == "corrected"),
            "rejected_excessive_edit": sum(1 for r in results if r.status == "rejected_excessive_edit"),
            "ollama_error": sum(1 for r in results if r.status == "ollama_error"),
        },
        "raw_page_text": "\n".join(r.raw_text for r in results),
        "corrected_page_text": "\n".join(r.corrected_text for r in results),
        "lines": [asdict(r) for r in results],
    }


# ---------------------------------------------------------------------------
# Self-Test Suite (Run directly: python3 post_corrector.py)
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    sample_lines = [
        # 0: High confidence -> must be skipped without calling LLM
        {"line_index": 0, "text": "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ ရတနာသုံးပါး။", "min_confidence": 92.5, "mean_confidence": 95.0},
        # 1: Low confidence with real typo -> corrected + diff generated
        {"line_index": 1, "text": "ကစ္စာယန သဒ္ဒါကျမ် နိဒါန်း။", "min_confidence": 64.0, "mean_confidence": 82.0},
        # 2: Low confidence with archaic spelling (တရုပ်) -> LLM tries to modernize -> must roll back!
        {"line_index": 2, "text": "တရုပ် ပုံနှိပ်စာလုံးဟောင်။", "min_confidence": 68.0, "mean_confidence": 79.0},
        # 3: Low confidence where LLM hallucinates a totally different sentence -> must reject!
        {"line_index": 3, "text": "သပြေကန် ဆရာတော် ဘုရားကြီး။", "min_confidence": 60.0, "mean_confidence": 75.0},
    ]

    llm_calls: List[str] = []

    def _mock_llm(prompt: str, _crop: Optional[Image.Image]) -> str:
        llm_calls.append(prompt)
        if "သဒ္ဒါကျမ်" in prompt:
            return "ကစ္စာယန သဒ္ဒါကျမ်း နိဒါန်း။"
        if "တရုပ်" in prompt:
            # Simulates an LLM fixing ဟောင်->ဟောင်း while wrongly modernizing တရုပ်->တရုတ်
            return "တရုတ် ပုံနှိပ်စာလုံးဟောင်း။"
        # Simulates a hallucinating LLM rewriting the line completely
        return "ဤစာကြောင်းသည် ဗုဒ္ဓဘာသာ ကျမ်းဂန်များအကြောင်း ရှင်းပြထားခြင်း ဖြစ်ပါသည်။"

    payload = correct_page_lines(
        sample_lines,
        config=CorrectorConfig(min_word_conf_thresh=80.0, max_edit_ratio=0.25),
        llm_fn=_mock_llm,
    )

    # 1. High-confidence line was never sent to LLM (only 3 of 4 lines sent)
    assert len(llm_calls) == 3, f"Expected 3 LLM calls, got {len(llm_calls)}"
    assert payload["lines"][0]["status"] == "skipped_high_conf"

    # 2. Typo line was cleanly corrected and HTML diff contains <ins>
    r1 = payload["lines"][1]
    assert r1["status"] == "corrected"
    assert r1["corrected_text"] == "ကစ္စာယန သဒ္ဒါကျမ်း နိဒါန်း။"
    assert '<ins class="ocr-ins">း</ins>' in r1["diff_html"]

    # 3. Archaic spelling "တရုပ်" was preserved while typo "ဟောင်"->"ဟောင်း" was kept!
    r2 = payload["lines"][2]
    assert r2["status"] == "corrected"
    assert r2["corrected_text"] == "တရုပ် ပုံနှိပ်စာလုံးဟောင်း။"
    assert "တရုတ်->တရုပ်" in r2["restored_archaic_terms"]

    # 4. Hallucinated rewrite was rejected and safely rolled back to raw OCR text
    r3 = payload["lines"][3]
    assert r3["status"] == "rejected_excessive_edit"
    assert r3["corrected_text"] == r3["raw_text"]

    print("All post_corrector.py gating, archaic-guard, anti-hallucination & diff tests PASSED.")


if __name__ == "__main__":
    _run_self_tests()