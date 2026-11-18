"""
patchcore_one_category.py - Step 2 of the defect-inspector project.

First real model run: trains PatchCore on ONE MVTec AD category and prints
how well it detects (image level) and localises (pixel level) defects.

"""

import time
from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Same trick as explore_dataset.py: paths built from this file's location
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"

# One category for this first run
CATEGORY = "screw"


def main() -> None:
    # 1. DATA ---------------------------------------------------------------
    # The datamodule finds the images, resizes them and builds the
    # train set (good only) and the test set (good + defective + masks).
    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=CATEGORY,
        train_batch_size=16,   # small batches keep GPU memory low (6 GB card)
        eval_batch_size=16,
        num_workers=0,         # 0 = load data in the main process; safest on Windows
    )

    # 2. MODEL --------------------------------------------------------------
    # PatchCore with its standard settings:
    #   - a WideResNet-50 pre-trained on ImageNet extracts features
    #   - features from layer2 + layer3 = mid-level textures and shapes
    #   - the memory bank keeps 10% of all "normal" patch features (coreset)
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
    )

    # 3. ENGINE -------------------------------------------------------------
    # The engine runs training and testing on the GPU and saves outputs
    # (checkpoint + heatmap images) under results/.
    engine = Engine(
        default_root_dir=RESULTS_DIR,
        accelerator="gpu",
        devices=1,
    )

    # 4. TRAIN --------------------------------------------------------------
    # For PatchCore "training" = pass every good image through the backbone
    # once and store its patch features in the memory bank. No gradients.
    start = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - start

    # 5. TEST ---------------------------------------------------------------
    # Every test image is compared with the memory bank. Patches far from
    # any stored normal patch get a high anomaly score -> heatmap.
    start = time.perf_counter()
    results = engine.test(model=model, datamodule=datamodule)
    test_seconds = time.perf_counter() - start

    # 6. REPORT -------------------------------------------------------------
    # engine.test() returns a list with one dictionary of metric values.
    metrics = results[0]

    print(f"\n=== PatchCore on '{CATEGORY}' ===")
    for name, value in metrics.items():
        print(f"  {name:<15} {value:.4f}")
    print(f"  fit time        {fit_seconds:.1f} s")
    print(f"  test time       {test_seconds:.1f} s")


# Required on Windows: protects the code from being re-run by worker processes
if __name__ == "__main__":
    main()