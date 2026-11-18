"""
vlm_zero_shot_eval.py - Step 4b of the defect-inspector project.

Evaluates the vision-language model (Qwen3.5-2B) on EVERY test image of one
category, with no training (zero-shot), and scores it with the same metric as
PatchCore (image AUROC), so the two can be compared fairly.

Key idea: instead of reading the model's text answer, we read its internal
probability of starting the answer with "DEFECTIVE" versus "GOOD".
That gives a score between 0 and 1 for every image, like PatchCore's anomaly score.

Two prompts are tested (leading vs neutral) to measure prompt sensitivity.

Outputs (small files, committed to git):
    reports/vlm_zero_shot_<category>_images.csv   one row per image and prompt
    reports/vlm_zero_shot_<category>_summary.csv  one row per prompt

"""

import csv
import time
from pathlib import Path

import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMultimodalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"

MODEL_ID = "Qwen/Qwen3.5-2B"
CATEGORY = "screw"

# The two prompts we compare. Both ask for one keyword first, so the first
# word the model produces is its decision.
PROMPTS = {
    "leading": (
        "You are a quality inspector on a production line. "
        "This image shows one metal screw photographed from above. "
        "Is the screw defective? "
        "On the first line answer only DEFECTIVE or GOOD."
    ),
    "neutral": (
        "Describe the condition of this screw. "
        "On the first line answer only GOOD or DEFECTIVE."
    ),
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def list_test_images(category: str) -> list[dict]:
    """Return every test image of a category with its true label.

    label = 0 for good parts, 1 for defective parts (any defect type).
    """
    test_dir = DATA_ROOT / category / "test"
    items = []
    for folder in sorted(test_dir.iterdir()):
        if not folder.is_dir():
            continue
        label = 0 if folder.name == "good" else 1
        for image in sorted(folder.glob("*.png")):
            items.append({"path": image, "defect_type": folder.name, "label": label})
    return items


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    """Load the processor and the model on the GPU in bfloat16."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()
    return processor, model


def first_token_id(processor, word: str) -> int:
    """Return the id of the FIRST token of a word.

    Long words are split into several tokens (e.g. "DEF" + "ECTIVE").
    The first token is enough to know which answer the model is starting.
    """
    return processor.tokenizer.encode(word, add_special_tokens=False)[0]


def defect_score(processor, model, image_path: Path, prompt: str,
                 good_id: int, defective_id: int) -> tuple[float, float]:
    """Return (score, mass) for one image.

    score = P(DEFECTIVE) / (P(DEFECTIVE) + P(GOOD)) for the first answer token.
            0 = sure it is good, 1 = sure it is defective.
    mass  = P(DEFECTIVE) + P(GOOD) out of the whole vocabulary.
            Close to 1 = the model really wanted to answer with one of the two
            words. Low = it wanted to say something else (format not followed).
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

    # ONE forward pass, no text generation: we only need the probabilities
    # of the very next token (the first word of the answer).
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]          # scores for the next token
    probs = torch.softmax(logits.float(), dim=-1)        # turn scores into probabilities

    p_good = probs[good_id].item()
    p_defective = probs[defective_id].item()
    mass = p_good + p_defective
    score = p_defective / mass if mass > 0 else 0.5
    return score, mass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    images = list_test_images(CATEGORY)
    labels = [item["label"] for item in images]
    print(f"{CATEGORY}: {len(images)} test images "
          f"({labels.count(0)} good, {labels.count(1)} defective)")

    processor, model = load_model()
    good_id = first_token_id(processor, "GOOD")
    defective_id = first_token_id(processor, "DEFECTIVE")
    # The two first tokens must differ, otherwise the score means nothing
    assert good_id != defective_id, "GOOD and DEFECTIVE share their first token"

    image_rows, summary_rows = [], []

    for prompt_name, prompt in PROMPTS.items():
        print(f"\n=== prompt: {prompt_name} ===")
        scores, masses = [], []
        start = time.perf_counter()

        for i, item in enumerate(images, start=1):
            score, mass = defect_score(processor, model, item["path"], prompt,
                                       good_id, defective_id)
            scores.append(score)
            masses.append(mass)
            image_rows.append({
                "prompt": prompt_name,
                "image": f"{item['defect_type']}/{item['path'].name}",
                "defect_type": item["defect_type"],
                "label": item["label"],
                "score": round(score, 4),
                "mass": round(mass, 4),
            })
            if i % 20 == 0:
                print(f"  {i}/{len(images)} images done")

        seconds = time.perf_counter() - start

        # AUROC: threshold-free, so a model that is simply biased towards one
        # answer is NOT punished as long as it ranks defective parts higher.
        auroc = roc_auc_score(labels, scores)

        # The model's own decision = whichever word is more likely (score > 0.5)
        predicted = [1 if s > 0.5 else 0 for s in scores]
        tp = sum(p == 1 and y == 1 for p, y in zip(predicted, labels))
        tn = sum(p == 0 and y == 0 for p, y in zip(predicted, labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(predicted, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(predicted, labels))

        summary = {
            "category": CATEGORY,
            "prompt": prompt_name,
            "image_AUROC": round(auroc, 4),
            "accuracy": round((tp + tn) / len(labels), 4),
            "said_defective": tp + fp,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "mean_mass": round(sum(masses) / len(masses), 4),
            "ms_per_image": round(seconds / len(images) * 1000, 1),
        }
        summary_rows.append(summary)
        print(f"  AUROC {auroc:.4f} | said DEFECTIVE to {tp + fp}/{len(labels)} | "
              f"TP {tp} TN {tn} FP {fp} FN {fn} | mean mass {summary['mean_mass']}")

    # Save both tables
    images_csv = REPORTS_DIR / f"vlm_zero_shot_{CATEGORY}_images.csv"
    with images_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(image_rows)

    summary_csv = REPORTS_DIR / f"vlm_zero_shot_{CATEGORY}_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nSaved: {images_csv}\nSaved: {summary_csv}")


if __name__ == "__main__":
    main()