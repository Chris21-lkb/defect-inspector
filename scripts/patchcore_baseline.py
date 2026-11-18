"""
patchcore_baseline.py - Step 3 of the defect-inspector project.

Runs PatchCore on all four project categories with the SAME settings and
saves one results table. This table is the baseline that every later model
(zero-shot VLM, hybrid) will be compared against.

Output: reports/patchcore_baseline.csv  (committed to GitHub)

"""

import csv
import time
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"   # heavy outputs (ignored by git)
REPORTS_DIR = PROJECT_ROOT / "reports"   # small result tables (committed to git)

CATEGORIES = ["screw", "metal_nut", "cable", "transistor"]

# Fixed random seed: the coreset sampling is random, so without a seed
# two runs can give slightly different numbers. With it, results are repeatable.
SEED = 42

# Metric names exactly as anomalib returns them
METRICS = ["image_AUROC", "image_F1Score", "pixel_AUROC", "pixel_F1Score"]


# ---------------------------------------------------------------------------
# One category
# ---------------------------------------------------------------------------

def run_category(category: str) -> dict:
    """Train + test PatchCore on one category and return one results row."""
    # Re-seed before every category so each run is independent and repeatable
    seed_everything(SEED, workers=True)

    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=category,
        train_batch_size=16,
        eval_batch_size=16,
        num_workers=0,
    )

    # Same model settings for every category: a fair comparison needs
    # identical settings everywhere
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    engine = Engine(default_root_dir=RESULTS_DIR, accelerator="gpu", devices=1)

    # Train (build the memory bank) and time it
    start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - start

    # Test and time it
    start = time.perf_counter()
    metrics = engine.test(model=model, datamodule=datamodule)[0]
    test_seconds = time.perf_counter() - start

    # Rough time per test image (includes image loading; clean latency comes later)
    n_test = len(datamodule.test_data)

    # Build one row of the results table
    row = {"category": category}
    for name in METRICS:
        row[name] = round(float(metrics[name]), 4)
    row["fit_s"] = round(fit_seconds, 1)
    row["test_images"] = n_test
    row["ms_per_image"] = round(test_seconds / n_test * 1000, 1)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)

    # Run every category and collect one row each
    rows = []
    for category in CATEGORIES:
        print(f"\n######## {category} ########")
        rows.append(run_category(category))

    # Add a MEAN row: the single headline number for the README and CV
    mean_row = {"category": "MEAN"}
    for name in METRICS:
        mean_row[name] = round(sum(r[name] for r in rows) / len(rows), 4)
    rows.append(mean_row)

    # Save the table as CSV (DictWriter writes one dict per line)
    out_path = REPORTS_DIR / "patchcore_baseline.csv"
    columns = ["category", *METRICS, "fit_s", "test_images", "ms_per_image"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)   # missing keys in the MEAN row are left empty

    # Print a readable summary
    print("\nPATCHCORE BASELINE")
    print(f"  {'category':<11} {'img_AUROC':>9} {'img_F1':>7} {'px_AUROC':>9} {'px_F1':>6}")
    for r in rows:
        print(f"  {r['category']:<11} {r['image_AUROC']:>9.4f} {r['image_F1Score']:>7.4f} "
              f"{r['pixel_AUROC']:>9.4f} {r['pixel_F1Score']:>6.4f}")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()