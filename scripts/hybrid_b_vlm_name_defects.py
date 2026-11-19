"""
hybrid_b_vlm_name_defects.py - Phase 5, stage B of the defect-inspector project.

Stage A (PatchCore) said WHERE the defect is. Here the vision-language model
(Qwen3.5-2B) says WHICH defect it is, as a multiple-choice question built from
the category's defect types. MVTec's folder names give the true answer, so the
naming accuracy can be measured exactly.

Three ways of showing the part to the VLM are compared:
    full : the whole image (VLM alone, no help from PatchCore)
    box  : the whole image with PatchCore's box drawn in red
    crop : a close-up of PatchCore's box only

Only truly defective images are used: this isolates the NAMING question.

Input : reports/hybrid_patchcore_predictions.csv   (from stage A)
Output: reports/hybrid_naming_images.csv           one row per image and view
        reports/hybrid_naming_summary.csv          accuracy per category and view

Run from the project root (E:\\defect-inspector):
    python scripts\\hybrid_b_vlm_name_defects.py
"""

import csv
import string
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForMultimodalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_CSV = REPORTS_DIR / "hybrid_patchcore_predictions.csv"

MODEL_ID = "Qwen/Qwen3.5-2B"   # best and fastest VLM from Steps 4d/4e
VIEWS = ["full", "box", "crop"]

# Plain-English description of every MVTec defect folder (the answer options).
# Order = order of the letters A, B, C, ...
DEFECTS = {
    "screw": {
        "object": "screw",
        "options": {
            "manipulated_front": "the tip of the screw is damaged or deformed",
            "scratch_head": "a scratch on the screw head",
            "scratch_neck": "a scratch on the neck, between the head and the thread",
            "thread_side": "damaged thread along the side of the screw",
            "thread_top": "damaged thread near the top of the screw",
        },
    },
    "metal_nut": {
        "object": "metal nut",
        "options": {
            "bent": "one edge of the nut is bent or deformed",
            "color": "a discoloured stain on the nut",
            "flip": "the nut is upside down (flipped)",
            "scratch": "a scratch on the nut surface",
        },
    },
    "cable": {
        "object": "cable",
        "options": {
            "bent_wire": "a wire is bent out of its insulation",
            "cable_swap": "two of the three wires are in the wrong positions",
            "combined": "several different defects at the same time",
            "cut_inner_insulation": "the insulation of one inner wire is cut",
            "cut_outer_insulation": "the outer insulation of the cable is cut",
            "missing_cable": "one of the three wires is completely missing",
            "missing_wire": "the copper is missing inside one wire",
            "poke_insulation": "the insulation is pierced or poked",
        },
    },
    "transistor": {
        "object": "transistor",
        "options": {
            "bent_lead": "a lead (leg) of the transistor is bent",
            "cut_lead": "a lead (leg) of the transistor is cut",
            "damaged_case": "the plastic case of the transistor is damaged",
            "misplaced": "the transistor is missing or not in its correct position",
        },
    },
}


# ---------------------------------------------------------------------------
# Images for each view
# ---------------------------------------------------------------------------

def make_view(image_path: Path, box: tuple[float, float, float, float], view: str) -> Image.Image:
    """Return the image to show the VLM for one view (full, box or crop)."""
    image = Image.open(image_path).convert("RGB")
    if view == "full":
        return image

    w, h = image.size
    x0, y0, x1, y1 = box[0] * w, box[1] * h, box[2] * w, box[3] * h

    if view == "box":
        # Draw a thick red rectangle; line width scales with image size
        drawn = image.copy()
        ImageDraw.Draw(drawn).rectangle([x0, y0, x1, y1], outline=(255, 0, 0),
                                        width=max(3, w // 150))
        return drawn

    # view == "crop": close-up of the suspicious area
    return image.crop((int(x0), int(y0), int(x1), int(y1)))


def build_prompt(category: str, view: str) -> tuple[str, list[str]]:
    """Build the multiple-choice question and return it with the defect folder order."""
    info = DEFECTS[category]
    folders = list(info["options"].keys())
    letters = string.ascii_uppercase[:len(folders)]

    if view == "full":
        intro = f"This {info['object']} is defective."
    elif view == "box":
        intro = f"This {info['object']} is defective. The red box marks the suspicious area."
    else:
        intro = f"This is a close-up of the suspicious area of a defective {info['object']}."

    lines = [intro, "Which defect is it?"]
    lines += [f"{letter}. {info['options'][folder]}" for letter, folder in zip(letters, folders)]
    lines.append("Answer with the letter only.")
    return "\n".join(lines), folders


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Load the processor and the 2B model on the GPU in bfloat16."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return processor, model


def letter_probabilities(processor, model, image: Image.Image, prompt: str,
                         letter_ids: list[int]) -> list[float]:
    """Probability of each answer letter as the model's first word (32-bit, as in Step 4d)."""
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},   # a PIL image this time, not a path
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=False,
    ).to(model.device)

    with torch.no_grad():
        hidden = model.model(**inputs).last_hidden_state[0, -1]
        rows = model.lm_head.weight[letter_ids].float()       # one row per letter
        logits = rows @ hidden.float()                        # 32-bit scores
        probs = torch.softmax(logits, dim=-1)                 # among the letters only
    return probs.tolist()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Stage A output: keep only truly defective images
    with PREDICTIONS_CSV.open() as f:
        rows = [r for r in csv.DictReader(f) if r["label"] == "1"]
    print(f"{len(rows)} defective images to name, {len(VIEWS)} views each")

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
            assert len(set(letter_ids)) == len(letter_ids), "two letters share a token"

            for r in (r for r in rows if r["category"] == category):
                image_path = DATA_ROOT / category / "test" / r["image"]
                box = (float(r["box_x0"]), float(r["box_y0"]),
                       float(r["box_x1"]), float(r["box_y1"]))
                image = make_view(image_path, box, view)

                probs = letter_probabilities(processor, model, image, prompt, letter_ids)
                predicted = folders[max(range(len(probs)), key=probs.__getitem__)]
                true_index = folders.index(r["defect_type"])

                image_rows.append({
                    "view": view,
                    "category": category,
                    "image": r["image"],
                    "true_defect": r["defect_type"],
                    "predicted_defect": predicted,
                    "correct": int(predicted == r["defect_type"]),
                    "p_true": round(probs[true_index], 4),
                    "box_hits_defect": r["box_hits_defect"],
                })
        print(f"  done in {time.perf_counter() - start:.0f} s")

    # Summary: accuracy per category and view, compared with random guessing
    summary_rows = []
    for view in VIEWS:
        for category, info in DEFECTS.items():
            items = [x for x in image_rows if x["view"] == view and x["category"] == category]
            correct = sum(x["correct"] for x in items)
            # Share of the single most frequent answer: high = the model keeps
            # picking the same option whatever the image (answer bias)
            top_answer, top_count = Counter(x["predicted_defect"] for x in items).most_common(1)[0]
            summary_rows.append({
                "view": view,
                "category": category,
                "n_images": len(items),
                "correct": correct,
                "accuracy": round(correct / len(items), 4),
                "chance": round(1 / len(info["options"]), 4),
                "most_common_answer": top_answer,
                "most_common_share": round(top_count / len(items), 4),
            })

    # Save
    for name, table in [("hybrid_naming_images.csv", image_rows),
                        ("hybrid_naming_summary.csv", summary_rows)]:
        with (REPORTS_DIR / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)

    # Print the comparison
    print("\nDEFECT NAMING ACCURACY (chance = random guess)")
    print(f"  {'category':<11} {'chance':>6} " + " ".join(f"{v:>13}" for v in VIEWS))
    for category, info in DEFECTS.items():
        cells = []
        for view in VIEWS:
            s = next(x for x in summary_rows if x["view"] == view and x["category"] == category)
            cells.append(f"{s['correct']:>3}/{s['n_images']:<3} {s['accuracy']:>5.2f}")
        print(f"  {category:<11} {1 / len(info['options']):>6.2f} " + " ".join(f"{c:>13}" for c in cells))
    print("\nSaved: reports/hybrid_naming_images.csv and reports/hybrid_naming_summary.csv")


if __name__ == "__main__":
    main()