"""
vlm_zero_shot_all.py - Step 4c of the defect-inspector project.

Zero-shot evaluation of Qwen3.5-2B on all four project categories with the
neutral prompt, plus a 95% confidence interval on every AUROC (bootstrap),
and a side-by-side comparison with the PatchCore baseline.

Outputs (committed to git):
    reports/vlm_zero_shot_images.csv    one row per image
    reports/vlm_zero_shot_summary.csv   one row per category (+ MEAN)

"""

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMultimodalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
PATCHCORE_CSV = REPORTS_DIR / "patchcore_baseline.csv"

MODEL_ID = "Qwen/Qwen3.5-2B"

# Folder name -> words used in the prompt
CATEGORIES = {
    "screw": "screw",
    "metal_nut": "metal nut",
    "cable": "cable",
    "transistor": "transistor",
}

# Neutral prompt (best in Step 4b). {object_name} is filled in per category,
# so for "screw" it is exactly the prompt used in Step 4b (consistency check).
PROMPT_TEMPLATE = (
    "Describe the condition of this {object_name}. "
    "On the first line answer only GOOD or DEFECTIVE."
)

# Bootstrap settings for the confidence intervals
N_BOOTSTRAP = 1000
SEED = 42


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_test_images(category: str) -> list[dict]:
    """Every test image of a category with its label (0 = good, 1 = defective)."""
    items = []
    for folder in sorted((DATA_ROOT / category / "test").iterdir()):
        if not folder.is_dir():
            continue
        label = 0 if folder.name == "good" else 1
        for image in sorted(folder.glob("*.png")):
            items.append({"path": image, "defect_type": folder.name, "label": label})
    return items


def read_patchcore_auroc() -> dict:
    """Read the PatchCore image AUROC per category from Step 3's CSV."""
    if not PATCHCORE_CSV.exists():
        return {}
    with PATCHCORE_CSV.open() as f:
        return {row["category"]: float(row["image_AUROC"]) for row in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Load the processor and the model on the GPU in bfloat16 (done ONCE)."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return processor, model


def first_token_id(processor, word: str) -> int:
    """Id of the first token of a word (enough to know which answer starts)."""
    return processor.tokenizer.encode(word, add_special_tokens=False)[0]


def defect_score(processor, model, image_path: Path, prompt: str,
                 good_id: int, defective_id: int) -> tuple[float, float]:
    """Return (score, mass) for one image. Same method as Step 4b.

    score = P(DEFECTIVE) / (P(DEFECTIVE) + P(GOOD)) for the first answer token
    mass  = P(DEFECTIVE) + P(GOOD) (how much the model wanted one of the two words)
    """
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "path": str(image_path)},
            {"type": "text", "text": prompt},
        ],
    }]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    probs = torch.softmax(logits.float(), dim=-1)

    p_good = probs[good_id].item()
    p_defective = probs[defective_id].item()
    mass = p_good + p_defective
    return (p_defective / mass if mass > 0 else 0.5), mass


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """95% confidence interval of the AUROC by bootstrap.

    Idea: re-draw the test set many times WITH replacement (same size),
    recompute the AUROC each time, and keep the middle 95% of the values.
    A wide interval = the test set is too small to be precise.
    """
    rng = np.random.default_rng(SEED)
    y, s = np.array(labels), np.array(scores)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        # A resample needs both classes, otherwise AUROC is undefined
        if len(np.unique(y[idx])) < 2:
            continue
        values.append(roc_auc_score(y[idx], s[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    # Load the model ONCE: loading takes seconds and 4 GB of GPU memory,
    # so re-loading it per category would only waste time.
    processor, model = load_model()
    good_id = first_token_id(processor, "GOOD")
    defective_id = first_token_id(processor, "DEFECTIVE")
    assert good_id != defective_id, "GOOD and DEFECTIVE share their first token"

    image_rows, summary_rows = [], []

    for category, object_name in CATEGORIES.items():
        prompt = PROMPT_TEMPLATE.format(object_name=object_name)
        images = list_test_images(category)
        labels = [item["label"] for item in images]
        print(f"\n=== {category}: {len(images)} images "
              f"({labels.count(0)} good, {labels.count(1)} defective) ===")

        scores, masses = [], []
        start = time.perf_counter()
        for i, item in enumerate(images, start=1):
            score, mass = defect_score(processor, model, item["path"], prompt,
                                       good_id, defective_id)
            scores.append(score)
            masses.append(mass)
            image_rows.append({
                "category": category,
                "image": f"{item['defect_type']}/{item['path'].name}",
                "defect_type": item["defect_type"],
                "label": item["label"],
                "score": round(score, 4),
                "mass": round(mass, 4),
            })
            if i % 50 == 0:
                print(f"  {i}/{len(images)} images done")
        seconds = time.perf_counter() - start

        # Ranking quality + its uncertainty
        auroc = roc_auc_score(labels, scores)
        ci_low, ci_high = bootstrap_auroc_ci(labels, scores)

        # The model's own decisions at score > 0.5
        predicted = [1 if s > 0.5 else 0 for s in scores]
        tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels))
        tn = sum(p == 0 and y == 0 for p, y in zip(predicted, labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))

        summary_rows.append({
            "category": category,
            "image_AUROC": round(auroc, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "mean_mass": round(sum(masses) / len(masses), 4),
            "ms_per_image": round(seconds / len(images) * 1000, 1),
        })
        print(f"  AUROC {auroc:.4f}  (95% CI {ci_low:.3f}-{ci_high:.3f}) | "
              f"TP {tp} TN {tn} FP {fp} FN {fn}")

    # MEAN row (AUROC only: averaging confidence intervals is not meaningful)
    mean_auroc = sum(r["image_AUROC"] for r in summary_rows) / len(summary_rows)
    summary_rows.append({"category": "MEAN", "image_AUROC": round(mean_auroc, 4)})

    # Save both tables
    images_csv = REPORTS_DIR / "vlm_zero_shot_images.csv"
    with images_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    summary_csv = REPORTS_DIR / "vlm_zero_shot_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)   # missing MEAN fields are left empty

    # Side-by-side comparison with PatchCore
    patchcore = read_patchcore_auroc()
    print("\nIMAGE AUROC: PatchCore (trained on good parts) vs VLM (zero-shot)")
    print(f"  {'category':<11} {'PatchCore':>9} {'VLM':>7}   VLM 95% CI")
    for r in summary_rows:
        pc = patchcore.get(r["category"])
        pc_text = f"{pc:.4f}" if pc is not None else "-"
        ci_text = (f"{r['ci95_low']:.3f}-{r['ci95_high']:.3f}"
                   if "ci95_low" in r else "")
        print(f"  {r['category']:<11} {pc_text:>9} {r['image_AUROC']:>7.4f}   {ci_text}")

    print(f"\nSaved: {images_csv}\nSaved: {summary_csv}")


if __name__ == "__main__":
    main()