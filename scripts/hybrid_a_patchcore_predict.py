"""
hybrid_a_patchcore_predict.py - Phase 5, stage A of the defect-inspector project.

The hybrid pipeline has two stages that run one after the other:
    A. PatchCore (this script): is the part defective, and WHERE is the defect?
    B. Vision-language model (hybrid_b_vlm_name_defects.py): WHICH defect is it?

This script trains PatchCore per category (same settings as the baseline),
predicts every test image, turns each anomaly heatmap into a box around the
most suspicious area, and checks whether that box really touches the defect
(using MVTec's ground-truth masks).

Output (committed to git):
    reports/hybrid_patchcore_predictions.csv   one row per test image


"""

import csv
from pathlib import Path

import numpy as np
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything
from PIL import Image

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"

CATEGORIES = ["screw", "metal_nut", "cable", "transistor"]
SEED = 42

# Box settings (all relative to the image size, 0-1)
HEAT_THRESHOLD = 0.5   # keep pixels hotter than 50% of the way from min to max heat
MARGIN = 0.15          # enlarge the box by 15% of the image on each side (context)
MIN_SIZE = 0.25        # a box is at least 25% of the image wide and high


# ---------------------------------------------------------------------------
# Heatmap -> box
# ---------------------------------------------------------------------------

def anomaly_box(anomaly_map: np.ndarray) -> tuple[float, float, float, float]:
    """Return a box (x0, y0, x1, y1), relative to image size, around the hottest area.

    1. Keep the pixels whose heat is in the top half of this map's range.
    2. Take the smallest box containing them.
    3. Add a margin and enforce a minimum size, so the VLM keeps some context.
    """
    h, w = anomaly_map.shape
    low, high = float(anomaly_map.min()), float(anomaly_map.max())
    threshold = low + HEAT_THRESHOLD * (high - low)
    ys, xs = np.where(anomaly_map >= threshold)

    # Relative coordinates of the tight box
    x0, x1 = xs.min() / w, (xs.max() + 1) / w
    y0, y1 = ys.min() / h, (ys.max() + 1) / h

    # Add the margin on every side
    x0, y0, x1, y1 = x0 - MARGIN, y0 - MARGIN, x1 + MARGIN, y1 + MARGIN

    # Enforce the minimum size around the box centre
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half_w, half_h = max((x1 - x0) / 2, MIN_SIZE / 2), max((y1 - y0) / 2, MIN_SIZE / 2)
    x0, x1, y0, y1 = cx - half_w, cx + half_w, cy - half_h, cy + half_h

    # Stay inside the image
    clip = lambda v: float(min(max(v, 0.0), 1.0))
    return clip(x0), clip(y0), clip(x1), clip(y1)


def box_hits_defect(image_path: Path, box: tuple[float, float, float, float]) -> int:
    """1 if the box contains at least one ground-truth defect pixel, else 0.

    Only meaningful for defective images (good images have no mask).
    MVTec rule: test/<type>/000.png -> ground_truth/<type>/000_mask.png
    """
    category_dir = image_path.parents[2]
    mask_path = (category_dir / "ground_truth" / image_path.parent.name
                 / f"{image_path.stem}_mask.png")
    mask = np.array(Image.open(mask_path).convert("L")) > 0
    h, w = mask.shape
    x0, y0, x1, y1 = box
    region = mask[int(y0 * h):int(np.ceil(y1 * h)), int(x0 * w):int(np.ceil(x1 * w))]
    return int(region.any())


# ---------------------------------------------------------------------------
# One category
# ---------------------------------------------------------------------------

def predict_category(category: str) -> list[dict]:
    """Train PatchCore on one category, predict every test image, return rows."""
    seed_everything(SEED, workers=True)

    datamodule = MVTecAD(root=DATA_ROOT, category=category,
                         train_batch_size=16, eval_batch_size=16, num_workers=0)
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1)
    engine = Engine(default_root_dir=RESULTS_DIR, accelerator="gpu", devices=1)

    engine.fit(model=model, datamodule=datamodule)

    # predict() runs on the test images and returns one result batch per step
    batches = engine.predict(model=model, datamodule=datamodule)

    rows = []
    for batch in batches:
        for i, path_str in enumerate(batch.image_path):
            image_path = Path(path_str)
            defect_type = image_path.parent.name
            label = 0 if defect_type == "good" else 1

            # Heatmap of this image as a 2-D numpy array (drop extra dimensions)
            heat = batch.anomaly_map[i].squeeze().detach().cpu().float().numpy()
            box = anomaly_box(heat)

            rows.append({
                "category": category,
                "image": f"{defect_type}/{image_path.name}",
                "defect_type": defect_type,
                "label": label,
                "pc_score": round(float(batch.pred_score[i]), 6),
                "pc_pred": int(batch.pred_label[i]),   # PatchCore's own decision
                "box_x0": round(box[0], 4), "box_y0": round(box[1], 4),
                "box_x1": round(box[2], 4), "box_y1": round(box[3], 4),
                "box_hits_defect": box_hits_defect(image_path, box) if label else "",
            })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    all_rows = []

    for category in CATEGORIES:
        print(f"\n######## {category} ########")
        rows = predict_category(category)
        all_rows.extend(rows)

        # Quick summary for this category
        defective = [r for r in rows if r["label"] == 1]
        detected = sum(r["pc_pred"] == 1 for r in defective)
        false_alarms = sum(r["pc_pred"] == 1 for r in rows if r["label"] == 0)
        hits = sum(r["box_hits_defect"] for r in defective)
        print(f"  defective detected: {detected}/{len(defective)} | "
              f"false alarms: {false_alarms}/{len(rows) - len(defective)} | "
              f"box touches the real defect: {hits}/{len(defective)}")

    out_path = REPORTS_DIR / "hybrid_patchcore_predictions.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()