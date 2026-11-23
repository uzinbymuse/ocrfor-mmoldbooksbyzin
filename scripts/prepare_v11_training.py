#!/usr/bin/env python3
"""Step 3 Unified Dataset Expansion & v11 Training Preparation Pipeline.

Implements:
1. Hugging Face `alexbeatson/burmese_ocr_data` ingestion with UTN #11 + Virama/Conjunct filter
   (with smooth automatic fallback to Degraded Lead-Type Synthetic Dhamma/Pali Generator when offline).
2. Degraded Lead-Type Synthetic Dhamma/Pali Generator (bleed-through, Gaussian blur 0.4-0.8,
   speckle noise, ±1.5 deg slant, variable stroke thickness, 12% vertical padding).
3. Unified Manifest (`data/training_manifest_v11.json`) with:
   - 3x Oversampled Real Human GT (Batches 1-7 or all verified JSONs in data/ground_truth/)
   - Beatson + Degraded Synthetic lines to reach 6,000 - 9,000 total training entries
   - Frozen 4-Page Evaluation Set (ZERO data leakage into training):
       * Vol 1 Preface p.11 (v1_p11_*)
       * Vol 2 p.23 (v2_p23_*)
       * Vol 1 p.17 (v1_p17_*)
       * Basic Kaccayana p.5 (kaccayana_p5_* / basic_kaccayana_p5_*)
4. Generates `.box`, `.gt.txt`, `.lstmf` files and `train_list_v11.txt` / `eval_list_v11.txt`.
"""
import os
import sys
import glob
import json
import time
import random
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from core.myanmar_normalizer import normalize_myanmar_unicode, validate_myanmar_line
from core.synthetic_dhamma_gen import (
    discover_myanmar_fonts,
    build_synthetic_text_pool,
    render_degraded_line,
)

# Frozen 4-page Evaluation Set prefixes (STRICTLY EXCLUDED from training!)
FROZEN_EVAL_PREFIXES = (
    "v1_p11_",
    "v2_p23_",
    "v1_p17_",
    "kaccayana_p5_",
    "basic_kaccayana_p5_",
)


def is_frozen_eval_id(line_id: str) -> bool:
    lid = (line_id or "").lower()
    return any(lid.startswith(p) for p in FROZEN_EVAL_PREFIXES)


def load_human_gt(gt_dirs, img_search_dirs):
    """Load all human-verified JSON batches, normalize via UTN #11, and split frozen eval vs train."""
    seen_ids = set()
    train_human = []
    eval_human = []

    json_files = []
    for d in gt_dirs:
        if os.path.isdir(d):
            json_files.extend(sorted(glob.glob(os.path.join(d, "*verified*.json"))))

    # Prefer *_corrected.json over un-corrected batch with same prefix
    by_stem = {}
    for jf in json_files:
        bn = os.path.basename(jf)
        key = bn.replace("_corrected.json", ".json")
        if key not in by_stem or "_corrected.json" in bn:
            by_stem[key] = jf

    for jf in sorted(by_stem.values()):
        try:
            rows = json.load(open(jf, encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skipping unreadable GT file {jf}: {e}")
            continue
        if not isinstance(rows, list):
            continue
        for row in rows:
            lid = row.get("id", "").strip()
            raw_txt = row.get("text", "").strip()
            if not lid or not raw_txt or lid in seen_ids:
                continue
            seen_ids.add(lid)
            norm_txt = normalize_myanmar_unicode(raw_txt)
            # Find matching PNG image
            img_path = None
            for idir in img_search_dirs:
                cand = os.path.join(idir, lid + ".png")
                if os.path.exists(cand):
                    img_path = cand
                    break
            rec = {
                "id": lid,
                "text": norm_txt,
                "source": "human_gt",
                "origin_file": os.path.basename(jf),
                "image_path": img_path or "",
            }
            if is_frozen_eval_id(lid):
                eval_human.append(rec)
            else:
                train_human.append(rec)

    return train_human, eval_human


def try_load_beatson_dataset(out_img_dir, max_lines=4000):
    """Attempt to load Hugging Face `alexbeatson/burmese_ocr_data` and filter via validate_myanmar_line()."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[INFO] Hugging Face `datasets` library not installed -> switching smoothly to Synthetic Fallback.")
        return []

    print(f"[INFO] Attempting to fetch alexbeatson/burmese_ocr_data (target up to {max_lines} clean lines)...")
    try:
        ds = load_dataset("alexbeatson/burmese_ocr_data", split="train", streaming=True)
        os.makedirs(out_img_dir, exist_ok=True)
        kept = []
        rejected = 0
        for idx, sample in enumerate(ds):
            if len(kept) >= max_lines:
                break
            raw_txt = (sample.get("text") or sample.get("transcription") or "").strip()
            norm_txt = normalize_myanmar_unicode(raw_txt)
            ok, _ = validate_myanmar_line(norm_txt, min_chars=8, max_chars=140)
            if not ok:
                rejected += 1
                continue
            img = sample.get("image")
            if img is None:
                continue
            lid = f"beatson_{len(kept)+1:05d}"
            img_path = os.path.join(out_img_dir, lid + ".png")
            # Enforce 12% vertical padding on Beatson line images too
            if hasattr(img, "convert"):
                g = img.convert("L")
                pad_y = max(8, int(g.height * 0.12))
                padded = Image.new("L", (g.width + 24, g.height + 2 * pad_y), 255)
                padded.paste(g, (12, pad_y))
                padded.save(img_path)
                kept.append({
                    "id": lid,
                    "text": norm_txt,
                    "source": "beatson_hf",
                    "image_path": img_path,
                })
        print(f"[INFO] Beatson loaded: {len(kept)} clean lines kept ({rejected} malformed filtered out).")
        return kept
    except Exception as e:
        print(f"[INFO] Beatson HF download unavailable ({type(e).__name__}: {e}) -> using Synthetic Fallback.")
        return []


def collect_extra_clean_corpus(corpus_json_paths, exclude_pages=None):
    """Extract clean, UTN #11 validated Myanmar/Pali lines from local OCR JSONs (excluding frozen eval pages)."""
    exclude_pages = exclude_pages or {("v1", 11), ("v1", 17), ("v2", 23)}
    lines = []
    for tag, path in corpus_json_paths:
        if not os.path.exists(path):
            continue
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception:
            continue
        for p_str, txt in data.items():
            try:
                pno = int(p_str)
            except ValueError:
                continue
            if (tag, pno) in exclude_pages:
                continue
            for raw_l in txt.splitlines():
                norm = normalize_myanmar_unicode(raw_l)
                ok, _ = validate_myanmar_line(norm, min_chars=18, max_chars=95)
                # Strict filter: skip lines with running headers or any Latin/ASCII digits
                if not ok:
                    continue
                if any(k in norm for k in ("တရားပေါင်းချုပ်", "သပြေကန်ဆရာတော်")):
                    continue
                if any(ord(ch) < 128 and ch.isalnum() for ch in norm):
                    continue
                lines.append(norm)
    return lines


def _build_lstmf_worker(args):
    img_path, text, tessdata_dir, lang = args
    base = os.path.splitext(img_path)[0]
    lstmf = base + ".lstmf"
    if os.path.exists(lstmf) and os.path.getsize(lstmf) > 0:
        return lstmf, True
    try:
        w, h = Image.open(img_path).size
    except Exception:
        return lstmf, False
    with open(base + ".gt.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open(base + ".box", "w", encoding="utf-8") as f:
        f.write(f"WordStr 0 0 {w} {h} 0 #{text}\n\t 0 0 {w} {h} 0\n")
    env = os.environ.copy()
    env["OMP_THREAD_LIMIT"] = "1"
    subprocess.run(
        ["tesseract", img_path, base, "--tessdata-dir", tessdata_dir, "-l", lang, "--psm", "6", "lstm.train"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    ok = os.path.exists(lstmf) and os.path.getsize(lstmf) > 0
    return lstmf, ok


def main():
    ap = argparse.ArgumentParser(description="Prepare v11 expanded dataset (>=6,000 lines), manifest & .lstmf lists.")
    ap.add_argument("--work-dir", default=REPO_ROOT, help="Repository root directory")
    ap.add_argument("--gt-dir", action="append", default=[], help="Extra directory containing verified GT JSONs")
    ap.add_argument("--img-dir", action="append", default=[], help="Extra directory containing GT PNG line images")
    ap.add_argument("--corpus-v1", default="", help="Optional Vol 1 corpus JSON path for synthetic text seeding")
    ap.add_argument("--corpus-v2", default="", help="Optional Vol 2 corpus JSON path for synthetic text seeding")
    ap.add_argument("--tessdata-dir", default="", help="Path to tessdata folder containing mya.traineddata")
    ap.add_argument("--target-total", type=int, default=6800, help="Target total manifest lines (>=6000)")
    ap.add_argument("--human-oversample", type=int, default=3, help="Oversample multiplier for human-verified GT")
    ap.add_argument("--render-limit", type=int, default=0, help="Limit synthetic PNG/.lstmf rendering for fast validation (0 = render all)")
    ap.add_argument("--skip-beatson", action="store_true", help="Skip Hugging Face network call (offline mode)")
    ap.add_argument("--workers", type=int, default=2, help="Parallel workers for .lstmf generation")
    args = ap.parse_args()

    t0 = time.time()
    work_dir = os.path.abspath(args.work_dir)
    data_dir = os.path.join(work_dir, "data")
    syn_dir = os.path.join(data_dir, "v11_synthetic")
    beatson_dir = os.path.join(data_dir, "v11_beatson")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(syn_dir, exist_ok=True)

    tessdata_dir = args.tessdata_dir or os.path.join(work_dir, "tessdata")
    gt_dirs = [os.path.join(data_dir, "ground_truth")] + args.gt_dir
    img_dirs = [os.path.join(data_dir, "ground_truth"), os.path.join(data_dir, "gt_human")] + args.img_dir

    # 1. Load Human-Verified GT & split Frozen 4-page Eval Set
    train_human, eval_human = load_human_gt(gt_dirs, img_dirs)
    print(f"[STEP 1] Loaded Human GT: train_unique={len(train_human)}, frozen_eval={len(eval_human)}")

    # 2. Try Beatson HF Dataset (3,000 ~ 5,000 lines) with smooth offline fallback
    beatson_rows = []
    if not args.skip_beatson:
        beatson_rows = try_load_beatson_dataset(beatson_dir, max_lines=4000)

    # 3. Compute how many Degraded Synthetic Dhamma/Pali lines are needed to reach target_total (>=6,000)
    oversampled_human_count = len(train_human) * args.human_oversample
    needed_syn = max(1200, args.target_total - oversampled_human_count - len(beatson_rows))

    extra_corpus = collect_extra_clean_corpus([
        ("v1", args.corpus_v1),
        ("v2", args.corpus_v2),
    ])
    syn_texts = build_synthetic_text_pool(target_count=needed_syn, extra_corpus_texts=extra_corpus, seed=1101)
    fonts = discover_myanmar_fonts()
    assert fonts, "No Myanmar fonts found on system (install Noto Serif Myanmar / Padauk / Pyidaungsu)."

    render_n = len(syn_texts) if args.render_limit <= 0 else min(len(syn_texts), args.render_limit)
    print(f"[STEP 2] Building {len(syn_texts)} Degraded Synthetic Dhamma/Pali entries (rendering {render_n} PNGs with {len(fonts)} font(s))...")

    rng = random.Random(1101)
    syn_rows = []
    for idx, txt in enumerate(syn_texts, 1):
        lid = f"syn_v11_{idx:05d}"
        img_path = os.path.join(syn_dir, lid + ".png")
        font_path = fonts[(idx - 1) % len(fonts)]
        pt_size = rng.choice([30, 34, 38])
        rendered = False
        if idx <= render_n:
            if not os.path.exists(img_path):
                ghost = syn_texts[(idx + 17) % len(syn_texts)]
                im = render_degraded_line(txt, font_path, font_size=pt_size, vertical_pad_ratio=0.12, rng=rng, ghost_text=ghost)
                im.save(img_path)
            rendered = True
        syn_rows.append({
            "id": lid,
            "text": txt,
            "source": "synthetic_lead_type",
            "font": os.path.basename(font_path),
            "vertical_pad_ratio": 0.12,
            "image_path": img_path if rendered else "",
            "rendered": rendered,
        })

    # 4. Assemble Unified Training Manifest (3x oversampled human GT + Beatson + Synthetic)
    train_entries = []
    for rep in range(1, args.human_oversample + 1):
        for r in train_human:
            entry = dict(r)
            entry["oversample_copy"] = rep
            train_entries.append(entry)
    train_entries.extend(beatson_rows)
    train_entries.extend(syn_rows)

    # Strict Zero-Leakage & 100% UTN #11 Compliance Audit
    eval_ids = {r["id"] for r in eval_human}
    leak_ids = [r["id"] for r in train_entries if r["id"] in eval_ids or is_frozen_eval_id(r["id"])]
    assert not leak_ids, f"DATA LEAKAGE DETECTED in train_entries: {leak_ids[:5]}"

    non_utn = [r["id"] for r in (train_entries + eval_human) if normalize_myanmar_unicode(r["text"]) != r["text"]]
    assert not non_utn, f"Non-UTN#11 entries found: {non_utn[:5]}"

    manifest = {
        "model_target": "mya_dhamma_v11",
        "base_model": "mya_dhamma_v10",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "utn11_normalized_pct": 100.0,
        "vertical_padding_ratio": 0.12,
        "frozen_eval_pages": ["Vol 1 Preface p.11 (v1_p11)", "Vol 2 p.23 (v2_p23)", "Vol 1 p.17 (v1_p17)", "Basic Kaccayana p.5 (kaccayana_p5)"],
        "data_leakage_verified": True,
        "counts": {
            "human_gt_unique_train": len(train_human),
            "human_gt_oversample_factor": args.human_oversample,
            "human_gt_oversampled_total": oversampled_human_count,
            "beatson_hf_lines": len(beatson_rows),
            "synthetic_lead_type_lines": len(syn_rows),
            "synthetic_rendered_pngs": render_n,
            "total_train_entries": len(train_entries),
            "frozen_eval_entries": len(eval_human),
        },
        "train_entries": train_entries,
        "eval_entries": eval_human,
    }
    manifest_path = os.path.join(data_dir, "training_manifest_v11.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[STEP 3] Saved manifest -> {manifest_path} | total_train={len(train_entries)} | frozen_eval={len(eval_human)}")

    # 5. Generate .lstmf files & train_list_v11.txt / eval_list_v11.txt
    lang = "mya" if os.path.exists(os.path.join(tessdata_dir, "mya.traineddata")) else "mya_dhamma_v10"
    unique_jobs = {}
    for r in (train_entries + eval_human):
        ip = r.get("image_path")
        if ip and os.path.exists(ip) and ip not in unique_jobs:
            unique_jobs[ip] = (ip, r["text"], tessdata_dir, lang)

    built_map = {}
    if unique_jobs and os.path.isdir(tessdata_dir):
        print(f"[STEP 4] Generating .lstmf files for {len(unique_jobs)} unique rendered line images...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for (ip, _), (lstmf, ok) in zip(unique_jobs.items(), ex.map(_build_lstmf_worker, unique_jobs.values())):
                if ok:
                    built_map[ip] = lstmf

    train_lstmf = [built_map[r["image_path"]] for r in train_entries if r.get("image_path") in built_map]
    eval_lstmf = [built_map[r["image_path"]] for r in eval_human if r.get("image_path") in built_map]
    random.Random(1101).shuffle(train_lstmf)

    train_list_path = os.path.join(data_dir, "train_list_v11.txt")
    eval_list_path = os.path.join(data_dir, "eval_list_v11.txt")
    with open(train_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lstmf) + ("\n" if train_lstmf else ""))
    with open(eval_list_path, "w", encoding="utf-8") as f:
        f.write("\n".join(eval_lstmf) + ("\n" if eval_lstmf else ""))

    print(
        f"[DONE in {time.time() - t0:.1f}s] Manifest entries={len(train_entries)} (>=6000: {len(train_entries) >= 6000}) | "
        f"UTN#11=100% | Zero Leakage=True | train_list_v11.txt={len(train_lstmf)} | eval_list_v11.txt={len(eval_lstmf)}"
    )


if __name__ == "__main__":
    main()
