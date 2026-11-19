"""
hybrid_c_calibrated_naming.py - Phase 5, step 5c of the defect-inspector project.

Problem found in stage B: the VLM has a "default answer" per category
(e.g. "poke_insulation" for 92/92 cables), whatever the image shows.

Fix: contextual calibration ("Calibrate Before Use", Zhao et al., 2021).
  1. Ask the same question about CONTENT-FREE images (plain grey, black, white).
     Whatever the model answers there is pure bias: there is nothing to see.
  2. For every real image, divide each answer's probability by that bias,
     re-normalise, and take the highest.
No labels and no test images are used to measure the bias, so there is no leakage.

This script re-runs stage B's questions (same model, prompts and views, imported
from hybrid_b_vlm_name_defects.py), keeps ALL answer probabilities, and reports
raw vs calibrated accuracy side by side. Raw results should match stage B
exactly: that is a built-in consistency check.

Input : reports/hybrid_patchcore_predictions.csv   (from stage A)
Output: reports/hybrid_calibrated_images.csv       one row per image and view
        reports/hybrid_calibrated_summary.csv      raw vs calibrated per category and view


"""

import csv
import string
import time
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw

# Re-use stage B's code instead of copying it: one source of truth for
# the defect lists, prompts, views, model loading and scoring.
from hybrid_b_vlm_name_defects import (
    DATA_ROOT, DEFECTS, PREDICTIONS_CSV, REPORTS_DIR, VIEWS,
    build_prompt, letter_probabilities, load_model, make_view,
)

# Plain colours used as content-free images
BLANK_COLOURS = [(128, 128, 128), (0, 0, 0), (255, 255, 255)]


# ---------------------------------------------------------------------------
# Bias measurement
# ---------------------------------------------------------------------------

def content_free_images(view: str) -> list[Image.Image]:
    """Blank images shaped like each view, so only the CONTENT is missing."""
    images = []
    for colour in BLANK_COLOURS:
        if view == "crop":
            image = Image.new("RGB", (400, 400), colour)        # crops are smaller
        else:
            image = Image.new("RGB", (1024, 1024), colour)
            if view == "box":                                    # same red box, nothing inside
                ImageDraw.Draw(image).rectangle([300, 300, 700, 700],
                                                outline=(255, 0, 0), width=7)
        images.append(image)
    return images


def measure_bias(processor, model, prompt: str, view: str, letter_ids: list[int]) -> list[float]:
    """Average answer probabilities over the content-free images = the model's bias."""
    all_probs = [letter_probabilities(processor, model, image, prompt, letter_ids)
                 for image in content_free_images(view)]
    n = len(all_probs)
    return [sum(p[i] for p in all_probs) / n for i in range(len(letter_ids))]


def calibrate(probs: list[float], bias: list[float]) -> list[float]:
    """Divide each probability by its bias, then re-normalise to sum to 1."""
    scaled = [p / max(b, 1e-9) for p, b in zip(probs, bias)]
    total = sum(scaled)
    return [s / total for s in scaled]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    with PREDICTIONS_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "1"]
    print(f"{len(rows)} defective images, {len(VIEWS)} views")

    processor, model = load_model()
    tokenizer = processor.tokenizer

    image_rows = []
    for view in VIEWS:
        print(f"\n=== view: {view} ===")
        start = time.perf_counter()

        for category in DEFECTS:
            prompt, folders = build_prompt(category, view)
            letters = string.ascii_uppercase[:len(folders)]
            letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]

            # 1. Bias of this exact question, measured on blank images
            bias = measure_bias(processor, model, prompt, view, letter_ids)
            favourite = folders[max(range(len(bias)), key=bias.__getitem__)]
            print(f"  {category:<11} bias favours '{favourite}' "
                  f"({max(bias):.0%} on a blank image)")

            # 2. Every real defective image: raw and calibrated answers
            for r in (r for r in rows if r["category"] == category):
                image_path = DATA_ROOT / category / "test" / r["image"]
                box = (float(r["box_x0"]), float(r["box_y0"]),
                       float(r["box_x1"]), float(r["box_y1"]))
                image = make_view(image_path, box, view)

                raw = letter_probabilities(processor, model, image, prompt, letter_ids)
                cal = calibrate(raw, bias)
                raw_pred = folders[max(range(len(raw)), key=raw.__getitem__)]
                cal_pred = folders[max(range(len(cal)), key=cal.__getitem__)]

                image_rows.append({
                    "view": view,
                    "category": category,
                    "image": r["image"],
                    "true_defect": r["defect_type"],
                    "raw_pred": raw_pred,
                    "raw_correct": int(raw_pred == r["defect_type"]),
                    "cal_pred": cal_pred,
                    "cal_correct": int(cal_pred == r["defect_type"]),
                })
        print(f"  done in {time.perf_counter() - start:.0f} s")

    # Summary per view and category: raw vs calibrated
    summary_rows = []
    for view in VIEWS:
        for category, info in DEFECTS.items():
            items = [x for x in image_rows if x["view"] == view and x["category"] == category]
            n = len(items)
            raw_top = Counter(x["raw_pred"] for x in items).most_common(1)[0][1]
            cal_top = Counter(x["cal_pred"] for x in items).most_common(1)[0][1]
            summary_rows.append({
                "view": view,
                "category": category,
                "n_images": n,
                "chance": round(1 / len(info["options"]), 4),
                "raw_correct": sum(x["raw_correct"] for x in items),
                "raw_accuracy": round(sum(x["raw_correct"] for x in items) / n, 4),
                "cal_correct": sum(x["cal_correct"] for x in items),
                "cal_accuracy": round(sum(x["cal_correct"] for x in items) / n, 4),
                "raw_top_share": round(raw_top / n, 4),   # 1.0 = always the same answer
                "cal_top_share": round(cal_top / n, 4),
            })

    for name, table in [("hybrid_calibrated_images.csv", image_rows),
                        ("hybrid_calibrated_summary.csv", summary_rows)]:
        with (REPORTS_DIR / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    # Print raw vs calibrated per view (all categories together)
    print("\nNAMING ACCURACY, ALL CATEGORIES (random guessing is about 20%)")
    print(f"  {'view':<5} {'raw':>12} {'calibrated':>12}")
    for view in VIEWS:
        items = [x for x in image_rows if x["view"] == view]
        n = len(items)
        raw = sum(x["raw_correct"] for x in items)
        cal = sum(x["cal_correct"] for x in items)
        print(f"  {view:<5} {raw:>4}/{n} {raw / n:>5.1%} {cal:>4}/{n} {cal / n:>5.1%}")

    print("\nPER CATEGORY (raw -> calibrated, share of the most common answer)")
    for s in summary_rows:
        print(f"  {s['view']:<5} {s['category']:<11} "
              f"{s['raw_accuracy']:>5.1%} -> {s['cal_accuracy']:>5.1%}   "
              f"top answer {s['raw_top_share']:>4.0%} -> {s['cal_top_share']:>4.0%}")
    print("\nSaved: reports/hybrid_calibrated_images.csv and reports/hybrid_calibrated_summary.csv")


if __name__ == "__main__":
    main()