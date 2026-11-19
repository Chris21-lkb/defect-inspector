"""
vlm_zero_shot_all.py - Steps 4c/4d/4e of the defect-inspector project.

Zero-shot evaluation of a vision-language model (VLM) on the four project
categories, scored with image AUROC + 95% bootstrap confidence interval,
broken down by defect type, and compared with the PatchCore baseline.

Step 4d change: the two answer scores (GOOD / DEFECTIVE) are computed in
32-bit instead of 16-bit, which removes the artificial ties between images.
Step 4e will reuse this file with a bigger model (change MODEL_ID / LOAD_IN_4BIT).

Outputs (committed to git), one set per RUN_NAME:
    reports/vlm_<RUN_NAME>_images.csv      one row per image
    reports/vlm_<RUN_NAME>_summary.csv     one row per category (+ MEAN)
    reports/vlm_<RUN_NAME>_by_defect.csv   AUROC per defect type

"""

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMultimodalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Settings: the only part to edit between runs
# ---------------------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3.5-2B"
LOAD_IN_4BIT = False                 # Step 4e: True for the 4B model
RUN_NAME = "qwen3.5-2b"              # used in the output file names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"
PATCHCORE_CSV = REPORTS_DIR / "patchcore_baseline.csv"

# Folder name -> words used in the prompt
CATEGORIES = {
    "screw": "screw",
    "metal_nut": "metal nut",
    "cable": "cable",
    "transistor": "transistor",
}

# Neutral prompt (best in Step 4b), filled in per category
PROMPT_TEMPLATE = (
    "Describe the condition of this {object_name}. "
    "On the first line answer only GOOD or DEFECTIVE."
)

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
    """PatchCore image AUROC per category, read from Step 3's CSV."""
    if not PATCHCORE_CSV.exists():
        return {}
    with PATCHCORE_CSV.open() as f:
        return {row["category"]: float(row["image_AUROC"]) for row in csv.DictReader(f)}


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write a list of dicts to CSV (columns taken from the first row)."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Load the processor and the model on the GPU (done ONCE)."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    extra = {}
    if LOAD_IN_4BIT:
        # Imported only when needed: requires `pip install bitsandbytes`
        from transformers import BitsAndBytesConfig
        extra["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",              # standard 4-bit format for LLM weights
            bnb_4bit_compute_dtype=torch.bfloat16,  # maths still done in 16-bit
        )

    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda", **extra
    )
    model.eval()
    return processor, model


def first_token_id(processor, word: str) -> int:
    """Id of the first token of a word (enough to know which answer starts)."""
    return processor.tokenizer.encode(word, add_special_tokens=False)[0]


def defect_score(processor, model, image_path: Path, prompt: str,
                 good_id: int, defective_id: int) -> tuple[float, float]:
    """Return (score, mass) for one image.

    score = P(DEFECTIVE) / (P(DEFECTIVE) + P(GOOD)), computed in 32-bit
    mass  = P(DEFECTIVE) + P(GOOD) over the whole vocabulary (diagnostic only)
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
        # Run the model body only (no output layer yet) and keep the
        # hidden state of the LAST position: it decides the next word.
        hidden = model.model(**inputs).last_hidden_state[0, -1]      # 16-bit vector

        # --- Step 4d fix -----------------------------------------------------
        # The output layer turns this vector into one score per vocabulary word.
        # In 16-bit those scores are rounded into coarse steps, so many images
        # got exactly the same score (ties). We only need TWO words, so we take
        # their two rows of the output layer and do this small product in 32-bit.
        rows = model.lm_head.weight[[good_id, defective_id]].float()  # (2, hidden)
        two_logits = rows @ hidden.float()                             # 32-bit
        logit_good, logit_defective = two_logits[0], two_logits[1]

        # P(D) / (P(D) + P(G)) is exactly sigmoid(logit_D - logit_G)
        score = torch.sigmoid(logit_defective - logit_good).item()

        # Mass needs the full vocabulary: 16-bit is fine for this diagnostic
        all_logits = model.lm_head(hidden)
        probs = torch.softmax(all_logits.float(), dim=-1)
        mass = (probs[good_id] + probs[defective_id]).item()

    return score, mass


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def bootstrap_auroc_ci(labels: list[int], scores: list[float]) -> tuple[float, float]:
    """95% confidence interval of the AUROC by bootstrap (see Step 4c)."""
    rng = np.random.default_rng(SEED)
    y, s = np.array(labels), np.array(scores)
    values = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:   # a resample needs both classes
            continue
        values.append(roc_auc_score(y[idx], s[idx]))
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def auroc_by_defect(rows: list[dict]) -> list[dict]:
    """AUROC of each defect type against the good images of its category."""
    out = []
    by_category = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)

    for category, items in by_category.items():
        good = [r for r in items if r["label"] == 0]
        defect_types = sorted({r["defect_type"] for r in items if r["label"] == 1})
        for defect in defect_types:
            bad = [r for r in items if r["defect_type"] == defect]
            subset = good + bad
            auroc = roc_auc_score([r["label"] for r in subset], [r["score"] for r in subset])
            out.append({"category": category, "defect_type": defect,
                        "n_images": len(bad), "image_AUROC": round(auroc, 4)})
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    processor, model = load_model()
    good_id = first_token_id(processor, "GOOD")
    defective_id = first_token_id(processor, "DEFECTIVE")
    assert good_id != defective_id, "GOOD and DEFECTIVE share their first token"

    image_rows, summary_rows = [], []

    for category, object_name in CATEGORIES.items():
        prompt = PROMPT_TEMPLATE.format(object_name=object_name)
        images = list_test_images(category)
        labels = [item["label"] for item in images]
        print(f"\n=== {category}: {len(images)} images ===")

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
                "score": round(score, 6),     # more decimals: scores are now finer
                "mass": round(mass, 4),
            })
            if i % 50 == 0:
                print(f"  {i}/{len(images)} images done")
        seconds = time.perf_counter() - start

        auroc = roc_auc_score(labels, scores)
        ci_low, ci_high = bootstrap_auroc_ci(labels, scores)
        n_unique = len(set(round(s, 6) for s in scores))

        summary_rows.append({
            "category": category,
            "image_AUROC": round(auroc, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "unique_scores": n_unique,
            "mean_mass": round(sum(masses) / len(masses), 4),
            "ms_per_image": round(seconds / len(images) * 1000, 1),
        })
        print(f"  AUROC {auroc:.4f}  (95% CI {ci_low:.3f}-{ci_high:.3f}) | "
              f"{n_unique} distinct scores for {len(images)} images")

    mean_auroc = sum(r["image_AUROC"] for r in summary_rows) / len(summary_rows)
    summary_rows.append({"category": "MEAN", "image_AUROC": round(mean_auroc, 4)})

    defect_rows = auroc_by_defect(image_rows)

    # Save the three tables
    write_csv(REPORTS_DIR / f"vlm_{RUN_NAME}_images.csv", image_rows)
    write_csv(REPORTS_DIR / f"vlm_{RUN_NAME}_summary.csv", summary_rows)
    write_csv(REPORTS_DIR / f"vlm_{RUN_NAME}_by_defect.csv", defect_rows)

    # Comparison with PatchCore
    patchcore = read_patchcore_auroc()
    print(f"\nIMAGE AUROC: PatchCore vs {RUN_NAME} (zero-shot)")
    print(f"  {'category':<11} {'PatchCore':>9} {'VLM':>7}   VLM 95% CI")
    for r in summary_rows:
        pc = patchcore.get(r["category"])
        pc_text = f"{pc:.4f}" if pc is not None else "-"
        ci_text = f"{r['ci95_low']:.3f}-{r['ci95_high']:.3f}" if "ci95_low" in r else ""
        print(f"  {r['category']:<11} {pc_text:>9} {r['image_AUROC']:>7.4f}   {ci_text}")

    print(f"\nPeak GPU memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Saved 3 tables in {REPORTS_DIR} with prefix vlm_{RUN_NAME}_")


if __name__ == "__main__":
    main()