"""UTN #11 Myanmar Unicode Canonical Normalizer for OCR Post-Processing.

Drop-in module designed for ``core/ocr_engine.py`` (``clean_line()``) in
``uzinbymuse/ocrfor-mmoldbooksbyzin``.

Implements Unicode Technical Note #11 (UTN #11) canonical diacritic storage
ordering for Burmese and Pali text in Myanmar script (Unicode 5.1+ / 18.0),
plus safe OCR-specific cleanup (zero-width removal, visual-order E-vowel
repair, Tall-AA normalization, and duplicate diacritic collapsing).

Reference:
    Unicode Technical Note #11: Representing Myanmar in Unicode
    https://www.unicode.org/notes/tn11/
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Unicode Character Definitions (Myanmar Block U+1000..U+109F)
# ---------------------------------------------------------------------------

# Invisible / zero-width characters commonly leaked by OCR engines or copy-paste
_INVISIBLE_CHARS_RE = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")

# Base characters that can start a Myanmar syllable cluster:
# Consonants (U+1000..U+1021), Independent Vowels (U+1023..U+1027, U+1029..U+102A),
# Great Sa (U+103F), Pali/Sanskrit extensions (U+1050..U+1055)
_BASE_CHAR = r"[\u1000-\u1021\u1023-\u1027\u1029\u102a\u103f\u1050-\u1055]"
_CONSONANT = r"[\u1000-\u1021\u103f\u1050-\u1055]"

# Kinzi prefix: Nga/Ra/Mon-Nga + Asat + Virama (e.g., င်္ in သင်္ဃာ)
_KINZI = r"(?:[\u1004\u101b\u105a]\u103a\u1039)"

# Subjoined stacked consonant: Virama (U+1039) + Consonant (e.g., ္ဓ in ဗုဒ္ဓ)
_STACK = r"(?:\u1039" + _CONSONANT + r")"

# Combining marks / diacritics governed by UTN #11 storage order
# Medials: 103B (ျ), 103C (ြ), 103D (ွ), 103E (ှ)
# Vowels:  1031 (ေ), 102D (ိ), 102E (ီ), 1032 (ဲ), 102F (ု), 1030 (ူ),
#          102B (ါ), 102C (ာ)
# Signs:   1036 (ံ), 1037 (့), 103A (်), 1038 (း)
_DIACRITIC_CLASS = r"[\u102b-\u1032\u1036-\u1038\u103a-\u103e]"

# Full Myanmar syllable cluster pattern:
#   Optional Kinzi + Base + Optional Stacked Consonants + 1+ Diacritics
# Note: Kinzi's own \u103a\u1039 is captured separately so it is never reordered.
_SYLLABLE_RE = re.compile(
    rf"(?P<kinzi>{_KINZI}?)"
    rf"(?P<base>{_BASE_CHAR})"
    rf"(?P<stack>{_STACK}*)"
    rf"(?P<marks>{_DIACRITIC_CLASS}+)"
)

# Visual-order Vowel Sign E (U+1031 ေ) misplaced BEFORE a consonant by OCR:
# Matches U+1031 preceded by start-of-string, whitespace, punctuation, or
# another syllable's closing sign/vowel (not a bare consonant that legitimately
# owns the U+1031), followed by a consonant (+ optional stack & medials).
_VISUAL_E_RE = re.compile(
    rf"(^|[\s(\[\"'\u104a\u104b\u1038\u103a\u102b\u102c\u102d\u102e\u102f\u1030\u1032\u1036\u1037])"
    rf"\u1031"
    rf"({_KINZI}?{_CONSONANT}{_STACK}*[\u103b-\u103e]*)"
)

# Consonants that require Tall AA (U+102B ါ) instead of regular AA (U+102C ာ)
# when no medial (ျ ြ ွ ှ) is present on the cluster.
_TALL_AA_CONSONANTS = frozenset("ခဂငဒပဝ")  # U+1001, U+1002, U+1004, U+1012, U+1015, U+101D

# UTN #11 Canonical Storage Order Weights for Burmese / Pali diacritics.
# Lower weight comes earlier in the Unicode codepoint sequence.
# Key guarantees:
#   1. Medials order:  ျ (103B) -> ြ (103C) -> ွ (103D) -> ှ (103E)
#   2. Vowel E (1031) precedes upper/lower/AA vowels
#   3. Upper vowel I (102D) precedes lower vowel U (102F) -> ို = 102D 102F
#   4. Lower vowels (102F/1030) precede Anusvara (1036) -> ုံ = 102F 1036
#   5. Dot Below (1037) precedes Asat (103A) -> Unicode NFC canonical order
#   6. Visarga (1038) is always last
_UTN11_ORDER = {
    "\u103b": 10,  # Medial Ya (ျ)
    "\u103c": 11,  # Medial Ra (ြ)
    "\u103d": 12,  # Medial Wa (ွ)
    "\u103e": 13,  # Medial Ha (ှ)
    "\u1031": 20,  # Vowel Sign E (ေ)
    "\u102d": 30,  # Vowel Sign I (ိ)
    "\u102e": 30,  # Vowel Sign II (ီ)
    "\u1032": 31,  # Vowel Sign AI (ဲ)
    "\u102f": 40,  # Vowel Sign U (ု)
    "\u1030": 40,  # Vowel Sign UU (ူ)
    "\u102b": 50,  # Vowel Sign Tall AA (ါ)
    "\u102c": 50,  # Vowel Sign AA (ာ)
    "\u1036": 60,  # Sign Anusvara (ံ)
    "\u1037": 70,  # Sign Dot Below (့) — ccc=7, precedes Asat in NFC
    "\u103a": 80,  # Sign Asat (်) — ccc=9
    "\u1038": 90,  # Sign Visarga (း)
}

# Punctuation & whitespace cleanup
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_DUP_PUDMA_RE = re.compile(r"(\u104b){2,}")   # ။။ -> ။
_DUP_PUDHPYAT_RE = re.compile(r"(\u104a){2,}")  # ၊၊ -> ၊

# Suspicious / non-canonical patterns for QA validation
_ORPHAN_MARK_RE = re.compile(
    rf"(?:^|[\s\u104a\u104b(\[\"']){_DIACRITIC_CLASS}+"
)
_CONFLICT_UPPER_RE = re.compile(r"[\u102d\u102e]{2,}")
_CONFLICT_LOWER_RE = re.compile(r"[\u102f\u1030]{2,}")
_CONFLICT_AA_RE = re.compile(r"[\u102b\u102c]{2,}")


@dataclass(frozen=True)
class ValidationIssue:
    """Represents a suspicious Unicode sequence left after normalization."""

    kind: str
    span: Tuple[int, int]
    text: str


def _reorder_cluster_marks(
    match: re.Match[str],
    *,
    deduplicate_marks: bool = True,
    normalize_tall_aa: bool = True,
) -> str:
    """Reorder diacritics on a single Myanmar syllable cluster per UTN #11."""
    kinzi = match.group("kinzi")
    base = match.group("base")
    stack = match.group("stack")
    marks = match.group("marks")

    # Sort marks strictly by UTN #11 weight (Python's Timsort is stable)
    sorted_marks = sorted(marks, key=lambda ch: _UTN11_ORDER.get(ch, 99))

    # Collapse identical consecutive marks (common OCR artifact, e.g. ာာ -> ာ)
    if deduplicate_marks:
        deduped: List[str] = []
        for ch in sorted_marks:
            if not deduped or deduped[-1] != ch:
                deduped.append(ch)
        sorted_marks = deduped

    # Normalize Tall AA (ါ) vs Regular AA (ာ) based on active consonant & medials
    if normalize_tall_aa and ("\u102b" in sorted_marks or "\u102c" in sorted_marks):
        has_medial = any(ch in "\u103b\u103c\u103d\u103e" for ch in sorted_marks)
        # In a stacked cluster (e.g. ဒ္ဓါ), the vowel attaches visually to the
        # final subjoined consonant; otherwise to the base consonant.
        active_consonant = stack[-1] if stack else base
        if active_consonant in _TALL_AA_CONSONANTS and not has_medial:
            sorted_marks = ["\u102b" if ch == "\u102c" else ch for ch in sorted_marks]
        elif active_consonant not in _TALL_AA_CONSONANTS or has_medial:
            sorted_marks = ["\u102c" if ch == "\u102b" else ch for ch in sorted_marks]

    return f"{kinzi}{base}{stack}{''.join(sorted_marks)}"


def normalize_myanmar_unicode(
    text: str,
    *,
    fix_visual_e: bool = True,
    deduplicate_marks: bool = True,
    normalize_tall_aa: bool = True,
    clean_punctuation: bool = True,
) -> str:
    """Normalize a Myanmar/Pali Unicode string to UTN #11 canonical order.

    Designed to be called inside ``clean_line()`` in ``core/ocr_engine.py``
    before or after regex ``WORD_FIXES``.

    Args:
        text: Raw OCR text line or paragraph.
        fix_visual_e: Repair OCR visual-order ``ေ`` placed before its consonant
            (e.g., ``ေက`` -> ``ကေ``).
        deduplicate_marks: Collapse duplicate combining marks within a syllable
            (e.g., ``ကာာ`` -> ``ကာ``, ``က့့`` -> ``က့``).
        normalize_tall_aa: Enforce ``ါ`` (U+102B) vs ``ာ`` (U+102C) according
            to consonant shape and presence of medials (e.g., ``ပာ`` -> ``ပါ``,
            ``ပြါ`` -> ``ပြာ``).
        clean_punctuation: Collapse repeated Myanmar section marks (``။။`` ->
            ``။``, ``၊၊`` -> ``၊``) and redundant spaces.

    Returns:
        Canonical UTN #11 NFC-normalized Myanmar Unicode string.
    """
    if not text:
        return text

    # 1. Strip zero-width & invisible OCR artifacts
    out = _INVISIBLE_CHARS_RE.sub("", text)

    # 2. Repair common visual-order Vowel Sign E (U+1031) leaked before consonant
    if fix_visual_e and "\u1031" in out:
        out = _VISUAL_E_RE.sub(r"\1\2" + "\u1031", out)

    # 3. Reorder syllable diacritics per UTN #11 (preserving Kinzi & stacks)
    out = _SYLLABLE_RE.sub(
        lambda m: _reorder_cluster_marks(
            m,
            deduplicate_marks=deduplicate_marks,
            normalize_tall_aa=normalize_tall_aa,
        ),
        out,
    )

    # 4. Standard Unicode NFC pass (composes U+1025 U+102E -> U+1026 ဦ and
    #    guarantees U+1037 precedes U+103A everywhere)
    out = unicodedata.normalize("NFC", out)

    # 5. Optional punctuation & whitespace polish
    if clean_punctuation:
        out = _DUP_PUDMA_RE.sub("\u104b", out)
        out = _DUP_PUDHPYAT_RE.sub("\u104a", out)
        out = _MULTI_SPACE_RE.sub(" ", out).strip()

    return out


def validate_myanmar_line(text: str) -> List[ValidationIssue]:
    """Detect remaining anomalous Myanmar sequences for Web Studio QC flagging.

    Returns a list of ``ValidationIssue`` items (empty list if clean).
    """
    issues: List[ValidationIssue] = []

    for pattern, kind in (
        (_ORPHAN_MARK_RE, "orphan_diacritic"),
        (_CONFLICT_UPPER_RE, "conflicting_upper_vowel"),
        (_CONFLICT_LOWER_RE, "conflicting_lower_vowel"),
        (_CONFLICT_AA_RE, "conflicting_aa_vowel"),
    ):
        for m in pattern.finditer(text):
            issues.append(ValidationIssue(kind=kind, span=m.span(), text=m.group(0)))

    return issues


# ---------------------------------------------------------------------------
# Self-Test Suite (Run directly: python3 myanmar_normalizer.py)
# ---------------------------------------------------------------------------

def _run_self_tests() -> None:
    cases = [
        # 1. Lower/Upper vowel swap (U+102F U+102D -> U+102D U+102F)
        ("င\u102f\u102d", "ငို", "UTN11 i+u ordering (ငို)"),
        ("က\u102f\u102dယ်", "ကိုယ်", "UTN11 i+u ordering (ကိုယ်)"),
        # 2. Dot below + Asat NFC order (U+103A U+1037 -> U+1037 U+103A)
        ("န\u103a\u1037", "န\u1037\u103a", "NFC dot-below before asat (န့်)"),
        ("ကော\u103a\u1037", "ကော\u1037\u103a", "NFC dot-below before asat (ကော့်)"),
        # 3. Medials ordering (Wa/Ha before Ra -> Ra + Wa + Ha)
        ("မ\u103e\u103d\u103cာ", "မြွှာ", "Medial canonical order (မြွှာ)"),
        # 4. Anusvara after lower vowel (U+1036 U+102F -> U+102F U+1036)
        ("သ\u1036\u102fး", "သုံး", "Anusvara after u (သုံး)"),
        # 5. Kinzi preservation (သင်္ဃာ)
        ("သင်္ဃာ", "သင်္ဃာ", "Kinzi preserved (သင်္ဃာ)"),
        # 6. Stacked Pali consonants preserved (ဗုဒ္ဓ, ပဋ္ဌာန်း)
        ("ဗုဒ္ဓ", "ဗုဒ္ဓ", "Pali stack preserved (ဗုဒ္ဓ)"),
        ("ပဋ္ဌာန်း", "ပဋ္ဌာန်း", "Pali stack with final asat (ပဋ္ဌာန်း)"),
        # 7. Independent vowel UU NFC composition (U+1025 U+102E -> U+1026)
        ("\u1025\u102eး", "ဦး", "NFC independent vowel uu (ဦး)"),
        # 8. Visual-order Vowel E repair
        ("\u1031ကျာင်း", "ကျောင်း", "Visual-order E repair (ကျောင်း)"),
        # 9. Tall AA normalization (ပာ -> ပါ, ပြါ -> ပြာ)
        ("ပာသည်", "ပါသည်", "Tall AA without medial (ပါသည်)"),
        ("ပြါသည်", "ပြာသည်", "Regular AA with medial (ပြာသည်)"),
        # 10. Zero-width & duplicate diacritic cleanup
        ("ဓမ္မ\u200bစာာအုပ်။။", "ဓမ္မစာအုပ်။", "Zero-width + dup cleanup"),
    ]

    for raw, expected, label in cases:
        actual = normalize_myanmar_unicode(raw)
        assert actual == expected, (
            f"FAIL [{label}]: expected {expected!r} "
            f"({[hex(ord(c)) for c in expected]}), "
            f"got {actual!r} ({[hex(ord(c)) for c in actual]})"
        )

    # Validation checks
    assert validate_myanmar_line("ဗုဒ္ဓ ဓမ္မ သင်္ဃာ") == []
    issues = validate_myanmar_line("က\u102d\u102e သည်")
    assert len(issues) == 1 and issues[0].kind == "conflicting_upper_vowel"

    print(f"All {len(cases)} normalization tests and validation checks PASSED.")


if __name__ == "__main__":
    _run_self_tests()