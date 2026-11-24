"""Stratified Golden-Set Evaluation Harness & Regression Gate for Myanmar OCR.

Drop-in evaluation script designed for ``scripts/eval_golden.py`` in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Features:
1. Locked NFC + UTN #11 Unicode normalization (via ``core.myanmar_normalizer``
   with automatic NFC fallback) and deterministic whitespace policy before
   scoring so encoding differences never inflate CER/WER.
2. Computes CER (with and without spaces), WER, Character Accuracy, and
   Exact-Match Line Rate using ``jiwer`` (with an exact built-in Levenshtein
   fallback if ``jiwer`` is not yet installed).
3. Stratified reporting by page type (``clean``, ``faded``, ``preface``,
   ``two_column``, ``footnote``, etc.) showing mean AND min–max range so
   worst-case strata are never hidden behind a global average.
4. Automated per-stratum Regression Gate comparing against a saved baseline
   JSON report (exits with status code 2 if any stratum regresses beyond
   ``--max-regression-pp``).

Usage:
    # Run self-test suite
    python3 eval_golden.py --self-test

    # Evaluate a run and save baseline report
    python3 eval_golden.py \\
        --manifest data/golden/manifest.json \\
        --hyp-dir cache/v9_balanced \\
        --model mya_dhamma_v9_balanced \\
        --preprocess "300dpi_sauvola" \\
        --psm 6 \\
        --save-report reports/baseline_v9.json

    # Gate a new experiment against the frozen baseline
    python3 eval_golden.py \\
        --manifest data/golden/manifest.json \\
        --hyp-dir cache/v10_candidate \\
        --model mya_dhamma_v10 \\
        --baseline reports/baseline_v9.json \\
        --max-regression-pp 0.20 \\
        --save-report reports/v10_candidate.json
"""

from __future__ import annotations
import sys, os
from pathlib import Path
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Optional integration with core/myanmar_normalizer.py (UTN #11)
try:
    from core.myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
except ImportError:
    try:
        from myanmar_normalizer import normalize_myanmar_unicode  # type: ignore
    except ImportError:
        def normalize_myanmar_unicode(text: str, **_: Any) -> str:  # type: ignore
            return unicodedata.normalize("NFC", text)

# Optional jiwer integration (exact built-in Levenshtein fallback if absent)
try:
    import jiwer as _jiwer  # type: ignore
except ImportError:
    _jiwer = None


_MULTI_SPACE_RE = re.compile(r"[ \t]+")


# ---------------------------------------------------------------------------
# 1. Locked Normalization Policy
# ---------------------------------------------------------------------------

def normalize_for_eval(
    text: str,
    *,
    normalize_tall_aa: bool = False,
    strip_blank_lines: bool = True,
) -> List[str]:
    """Normalize text lines identically for reference and hypothesis.

    Note: ``normalize_tall_aa`` defaults to ``False`` during evaluation so the
    scorer strictly respects ground-truth spellings without silently rewriting
    either reference or hypothesis.
    """
    lines: List[str] = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        cleaned = normalize_myanmar_unicode(
            raw_line,
            fix_visual_e=False,
            deduplicate_marks=False,
            normalize_tall_aa=normalize_tall_aa,
            clean_punctuation=False,
        )
        cleaned = unicodedata.normalize("NFC", cleaned)
        cleaned = _MULTI_SPACE_RE.sub(" ", cleaned).strip()
        if strip_blank_lines and not cleaned:
            continue
        lines.append(cleaned)
    return lines


# ---------------------------------------------------------------------------
# 2. Edit Distance, CER, WER & Exact-Line Metrics
# ---------------------------------------------------------------------------

def _levenshtein(seq_a: Sequence[Any], seq_b: Sequence[Any]) -> int:
    """Exact O(min(m, n)) space Wagner-Fischer edit distance."""
    if seq_a == seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    if not seq_b:
        return len(seq_a)

    prev = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        curr = [i] + [0] * len(seq_b)
        for j, item_b in enumerate(seq_b, start=1):
            cost = 0 if item_a == item_b else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[-1]


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (0.0 .. 1.0+)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    if _jiwer is not None:
        return float(_jiwer.cer(ref, hyp))
    return _levenshtein(ref, hyp) / len(ref)


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate on whitespace-tokenized text (0.0 .. 1.0+)."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    if _jiwer is not None:
        return float(_jiwer.wer(ref, hyp))
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def compute_exact_line_rate(ref_lines: List[str], hyp_lines: List[str]) -> Tuple[int, int, float]:
    """Compute exact-match line rate after locked line normalization."""
    total = len(ref_lines)
    if total == 0:
        return 0, 0, 1.0 if not hyp_lines else 0.0
    exact = sum(1 for r, h in zip(ref_lines, hyp_lines) if r == h)
    return exact, total, exact / total


# ---------------------------------------------------------------------------
# 3. Data Structures & Page/Stratum Evaluation
# ---------------------------------------------------------------------------

@dataclass
class PageScore:
    page_id: str
    book_id: str
    stratum: str
    ref_chars: int
    ref_words: int
    ref_lines: int
    exact_lines: int
    cer_pct: float
    cer_nospace_pct: float
    char_accuracy_pct: float
    wer_pct: float
    exact_line_pct: float


@dataclass
class StratumSummary:
    stratum: str
    page_count: int
    total_ref_chars: int
    cer_pct: float
    cer_min_pct: float
    cer_max_pct: float
    cer_nospace_pct: float
    char_accuracy_pct: float
    wer_pct: float
    exact_line_pct: float


def score_page_texts(
    *,
    page_id: str,
    book_id: str,
    stratum: str,
    ref_text: str,
    hyp_text: str,
) -> PageScore:
    """Evaluate one page's hypothesis text against its frozen reference."""
    ref_lines = normalize_for_eval(ref_text)
    hyp_lines = normalize_for_eval(hyp_text)

    ref_joined = "\n".join(ref_lines)
    hyp_joined = "\n".join(hyp_lines)

    ref_flat = " ".join(ref_lines)
    hyp_flat = " ".join(hyp_lines)

    ref_nospace = re.sub(r"\s+", "", ref_flat)
    hyp_nospace = re.sub(r"\s+", "", hyp_flat)

    cer = compute_cer(ref_joined, hyp_joined)
    cer_ns = compute_cer(ref_nospace, hyp_nospace)
    wer = compute_wer(ref_flat, hyp_flat)
    exact, total_lines, exact_rate = compute_exact_line_rate(ref_lines, hyp_lines)

    return PageScore(
        page_id=page_id,
        book_id=book_id,
        stratum=stratum,
        ref_chars=len(ref_joined),
        ref_words=len(ref_flat.split()),
        ref_lines=total_lines,
        exact_lines=exact,
        cer_pct=round(cer * 100.0, 3),
        cer_nospace_pct=round(cer_ns * 100.0, 3),
        char_accuracy_pct=round(max(0.0, (1.0 - cer) * 100.0), 3),
        wer_pct=round(wer * 100.0, 3),
        exact_line_pct=round(exact_rate * 100.0, 2),
    )


def aggregate_strata(pages: Sequence[PageScore]) -> Tuple[Dict[str, StratumSummary], StratumSummary]:
    """Compute character-weighted stratum summaries and overall summary."""
    by_stratum: Dict[str, List[PageScore]] = {}
    for p in pages:
        by_stratum.setdefault(p.stratum, []).append(p)

    def _summarize(name: str, items: Sequence[PageScore]) -> StratumSummary:
        total_chars = sum(x.ref_chars for x in items) or 1
        total_words = sum(x.ref_words for x in items) or 1
        total_lines = sum(x.ref_lines for x in items) or 1

        # Character-weighted CER & word-weighted WER for accurate micro-averages
        w_cer = sum(x.cer_pct * x.ref_chars for x in items) / total_chars
        w_cer_ns = sum(x.cer_nospace_pct * x.ref_chars for x in items) / total_chars
        w_wer = sum(x.wer_pct * x.ref_words for x in items) / total_words
        exact_pct = (sum(x.exact_lines for x in items) / total_lines) * 100.0

        cers = [x.cer_pct for x in items]
        return StratumSummary(
            stratum=name,
            page_count=len(items),
            total_ref_chars=sum(x.ref_chars for x in items),
            cer_pct=round(w_cer, 3),
            cer_min_pct=round(min(cers), 3),
            cer_max_pct=round(max(cers), 3),
            cer_nospace_pct=round(w_cer_ns, 3),
            char_accuracy_pct=round(max(0.0, 100.0 - w_cer), 3),
            wer_pct=round(w_wer, 3),
            exact_line_pct=round(exact_pct, 2),
        )

    strata_map = {
        name: _summarize(name, items)
        for name, items in sorted(by_stratum.items())
    }
    overall = _summarize("OVERALL", pages)
    return strata_map, overall


def evaluate_manifest(
    manifest_path: Path,
    *,
    hyp_dir: Optional[Path] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load a Golden Set manifest JSON and score every held-out page.

    Manifest schema:
        {
          "version": "1.0",
          "pages": [
            {
              "page_id": "vol1_p0017",
              "book_id": "vol1",
              "stratum": "clean",
              "ref_txt": "refs/vol1_p0017.txt",
              "hyp_txt": "vol1_p0017.txt"   // optional if matches page_id.txt in hyp_dir
            }
          ]
        }
    """
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    entries = raw.get("pages", [])
    if not entries:
        raise ValueError(f"No 'pages' found in manifest: {manifest_path}")

    page_scores: List[PageScore] = []
    for entry in entries:
        page_id = str(entry["page_id"])
        book_id = str(entry.get("book_id", "unknown"))
        stratum = str(entry.get("stratum", "default"))

        ref_path = Path(entry["ref_txt"])
        if not ref_path.is_absolute():
            ref_path = base_dir / ref_path

        if hyp_dir is not None:
            hyp_name = entry.get("hyp_txt") or f"{page_id}.txt"
            hyp_path = hyp_dir / Path(hyp_name).name
        else:
            hyp_name = entry.get("hyp_txt") or f"{page_id}.txt"
            hyp_path = Path(hyp_name)
            if not hyp_path.is_absolute():
                hyp_path = base_dir / hyp_path

        ref_text = ref_path.read_text(encoding="utf-8")
        hyp_text = hyp_path.read_text(encoding="utf-8") if hyp_path.exists() else ""

        page_scores.append(
            score_page_texts(
                page_id=page_id,
                book_id=book_id,
                stratum=stratum,
                ref_text=ref_text,
                hyp_text=hyp_text,
            )
        )

    strata_map, overall = aggregate_strata(page_scores)

    return {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "scorer": "jiwer" if _jiwer is not None else "builtin_levenshtein",
        "normalization_policy": "UTN11_NFC_collapse_spaces_strip_blank_lines",
        "metadata": metadata or {},
        "overall": asdict(overall),
        "strata": {k: asdict(v) for k, v in strata_map.items()},
        "pages": [asdict(p) for p in page_scores],
    }


# ---------------------------------------------------------------------------
# 4. Per-Stratum Regression Gate
# ---------------------------------------------------------------------------

def check_regression_gate(
    current_report: Dict[str, Any],
    baseline_report: Dict[str, Any],
    *,
    max_regression_pp: float = 0.25,
) -> Tuple[bool, List[str]]:
    """Verify no stratum (nor overall) regressed in CER beyond ``max_regression_pp``.

    Returns ``(passed, messages)``.
    """
    messages: List[str] = []
    passed = True

    def _compare(label: str, curr_cer: float, base_cer: float) -> None:
        nonlocal passed
        delta = round(curr_cer - base_cer, 3)
        sign = "+" if delta > 0 else ""
        if delta > max_regression_pp:
            passed = False
            messages.append(
                f"FAIL [{label}]: CER {base_cer:.3f}% -> {curr_cer:.3f}% "
                f"({sign}{delta:.3f} pp > allowed +{max_regression_pp:.3f} pp)"
            )
        else:
            messages.append(
                f"PASS [{label}]: CER {base_cer:.3f}% -> {curr_cer:.3f}% ({sign}{delta:.3f} pp)"
            )

    _compare(
        "OVERALL",
        float(current_report["overall"]["cer_pct"]),
        float(baseline_report["overall"]["cer_pct"]),
    )

    base_strata = baseline_report.get("strata", {})
    curr_strata = current_report.get("strata", {})
    for stratum, base_stats in sorted(base_strata.items()):
        if stratum not in curr_strata:
            passed = False
            messages.append(f"FAIL [{stratum}]: stratum missing from current report")
            continue
        _compare(
            stratum,
            float(curr_strata[stratum]["cer_pct"]),
            float(base_stats["cer_pct"]),
        )

    return passed, messages


def print_summary(report: Dict[str, Any]) -> None:
    """Print a compact stratified evaluation table to stdout."""
    print(f"\n=== Golden Set Evaluation ({report['scorer']}) ===")
    if report.get("metadata"):
        meta_str = " | ".join(f"{k}={v}" for k, v in report["metadata"].items() if v)
        if meta_str:
            print(f"Config: {meta_str}")

    header = (
        f"{'Stratum':<14} {'Pages':>5} {'CER%':>8} {'Range (min-max)':>17} "
        f"{'CER(ns)%':>9} {'WER%':>8} {'ExactLn%':>9}"
    )
    print("\n" + header)
    print("-" * len(header))

    for name, s in report["strata"].items():
        rng = f"{s['cer_min_pct']:.2f}-{s['cer_max_pct']:.2f}%"
        print(
            f"{name:<14} {s['page_count']:>5} {s['cer_pct']:>7.2f}% {rng:>17} "
            f"{s['cer_nospace_pct']:>8.2f}% {s['wer_pct']:>7.2f}% {s['exact_line_pct']:>8.1f}%"
        )

    ov = report["overall"]
    rng = f"{ov['cer_min_pct']:.2f}-{ov['cer_max_pct']:.2f}%"
    print("-" * len(header))
    print(
        f"{'OVERALL':<14} {ov['page_count']:>5} {ov['cer_pct']:>7.2f}% {rng:>17} "
        f"{ov['cer_nospace_pct']:>8.2f}% {ov['wer_pct']:>7.2f}% {ov['exact_line_pct']:>8.1f}%\n"
    )


# ---------------------------------------------------------------------------
# 5. Self-Test Suite
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    import tempfile

    # 1. Verify NFC + UTN #11 equivalence yields 0.0% CER (zero false penalty)
    p_equiv = score_page_texts(
        page_id="equiv_1",
        book_id="vol1",
        stratum="clean",
        ref_text="ငို ကိုယ် န\u1037\u103a ဦး\n",
        hyp_text="င\u102f\u102d က\u102f\u102dယ် န\u103a\u1037 \u1025\u102eး\n",
    )
    assert p_equiv.cer_pct == 0.0, f"Expected 0.0% CER on canonical equivalents, got {p_equiv.cer_pct}"
    assert p_equiv.exact_line_pct == 100.0

    # 2. End-to-end manifest + regression gate test (pass & fail cases)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "refs").mkdir()
        (root / "hyp_v9").mkdir()
        (root / "hyp_v10_good").mkdir()
        (root / "hyp_v10_regress").mkdir()

        manifest = {
            "version": "1.0",
            "pages": [
                {"page_id": "b1_p01", "book_id": "book1", "stratum": "clean", "ref_txt": "refs/b1_p01.txt"},
                {"page_id": "b2_p11", "book_id": "book2", "stratum": "faded", "ref_txt": "refs/b2_p11.txt"},
            ],
        }
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        ref_clean = "ဗုဒ္ဓ ဓမ္မ သင်္ဃာ ရတနာသုံးပါး။\nသပြေကန် ဆရာတော် ဘုရားကြီး။\n"
        ref_faded = "ကစ္စာယန သဒ္ဒါကျမ်း နိဒါန်း။\nတရုပ် ပုံနှိပ်စာလုံးဟောင်း။\n"
        (root / "refs/b1_p01.txt").write_text(ref_clean, encoding="utf-8")
        (root / "refs/b2_p11.txt").write_text(ref_faded, encoding="utf-8")

        # Baseline v9: 1 typo on clean, 1 typo on faded
        (root / "hyp_v9/b1_p01.txt").write_text(ref_clean.replace("ကြီး", "ကြီ"), encoding="utf-8")
        (root / "hyp_v9/b2_p11.txt").write_text(ref_faded.replace("နိဒါန်း", "နိဒါန်"), encoding="utf-8")

        # Candidate v10 good: improves faded, keeps clean
        (root / "hyp_v10_good/b1_p01.txt").write_text(ref_clean.replace("ကြီး", "ကြီ"), encoding="utf-8")
        (root / "hyp_v10_good/b2_p11.txt").write_text(ref_faded, encoding="utf-8")

        # Candidate v10 bad: perfect on clean, but regresses on faded
        (root / "hyp_v10_regress/b1_p01.txt").write_text(ref_clean, encoding="utf-8")
        (root / "hyp_v10_regress/b2_p11.txt").write_text(ref_faded.replace("တရုပ်", "တရုတ်").replace("နိဒါန်း", "နိဒန်"), encoding="utf-8")

        base_rep = evaluate_manifest(root / "manifest.json", hyp_dir=root / "hyp_v9")
        good_rep = evaluate_manifest(root / "manifest.json", hyp_dir=root / "hyp_v10_good")
        bad_rep = evaluate_manifest(root / "manifest.json", hyp_dir=root / "hyp_v10_regress")

        ok_good, _ = check_regression_gate(good_rep, base_rep, max_regression_pp=0.25)
        ok_bad, bad_msgs = check_regression_gate(bad_rep, base_rep, max_regression_pp=0.25)

        assert ok_good is True, "Improved candidate should pass regression gate"
        assert ok_bad is False, "Stratum regression must fail gate even if clean improved"
        assert any("FAIL [faded]" in m for m in bad_msgs)

    print("All eval_golden.py normalization, stratification & regression-gate tests PASSED.")


# ---------------------------------------------------------------------------
# 6. CLI Entry Point
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stratified Myanmar OCR Golden-Set Evaluator")
    parser.add_argument("--self-test", action="store_true", help="Run built-in verification tests")
    parser.add_argument("--manifest", type=Path, help="Path to golden set manifest.json")
    parser.add_argument("--hyp-dir", type=Path, default=None, help="Directory with OCR hypothesis .txt files")
    parser.add_argument("--model", type=str, default="", help="Model name/version to record in report")
    parser.add_argument("--preprocess", type=str, default="", help="Preprocessing recipe tag")
    parser.add_argument("--psm", type=str, default="", help="Tesseract PSM setting")
    parser.add_argument("--save-report", type=Path, default=None, help="Write full JSON report to path")
    parser.add_argument("--baseline", type=Path, default=None, help="Baseline JSON report for regression gate")
    parser.add_argument(
        "--max-regression-pp",
        type=float,
        default=0.25,
        help="Max allowed CER regression per stratum in percentage points (default: 0.25)",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        _run_self_tests()
        return 0

    if not args.manifest:
        parser.error("--manifest is required unless --self-test is passed.")

    report = evaluate_manifest(
        args.manifest,
        hyp_dir=args.hyp_dir,
        metadata={"model": args.model, "preprocess": args.preprocess, "psm": args.psm},
    )
    print_summary(report)

    if args.save_report:
        args.save_report.parent.mkdir(parents=True, exist_ok=True)
        args.save_report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved report to {args.save_report}")

    if args.baseline:
        baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
        passed, messages = check_regression_gate(
            report,
            baseline_report,
            max_regression_pp=args.max_regression_pp,
        )
        print("=== Regression Gate ===")
        for msg in messages:
            print(" ", msg)
        if not passed:
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())