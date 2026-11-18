"""
explore_dataset.py - Step 1 of the defect-inspector project.

Looks at the MVTec AD dataset before any model touches it:
  1. Counts the training images per category (all defect-free).
  2. Counts the test images per defect type.
  3. Checks that every defective test image has its ground-truth mask.

"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

# Project root = the folder above /scripts. Built from this file's own location,
# so the script works no matter which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Folder that contains the 15 MVTec AD category folders (screw, cable, ...)
DATA_ROOT = PROJECT_ROOT / "datasets"

# The four categories used in this project (closest to real factory parts)
CATEGORIES = ["screw", "metal_nut", "cable", "transistor"]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def count_pngs(folder: Path) -> int:
    """Return the number of .png images directly inside a folder."""
    # glob() yields matching files one by one, so wrap it in list() to count them
    return len(list(folder.glob("*.png")))


def find_missing_masks(category_dir: Path, defect_type: str) -> list[str]:
    """Return the names of defective test images that have NO mask file.

    MVTec naming rule:
        test/<defect_type>/000.png  ->  ground_truth/<defect_type>/000_mask.png
    """
    test_dir = category_dir / "test" / defect_type
    mask_dir = category_dir / "ground_truth" / defect_type

    missing = []
    for image in sorted(test_dir.glob("*.png")):
        # image.stem is the file name without its extension: "000.png" -> "000"
        mask = mask_dir / f"{image.stem}_mask.png"
        if not mask.exists():
            missing.append(image.name)
    return missing


def explore_category(category: str) -> dict:
    """Print a table for one category and return its totals."""
    category_dir = DATA_ROOT / category

    # Training set: only "good" images exist here
    train_good = count_pngs(category_dir / "train" / "good")

    # Test set: one sub-folder per defect type, plus "good"
    test_dir = category_dir / "test"
    defect_types = sorted(d.name for d in test_dir.iterdir()
                          if d.is_dir() and d.name != "good")
    test_good = count_pngs(test_dir / "good")

    # Header of the table for this category
    print(f"\n{category.upper()}")
    print(f"  {'split':<6} {'folder':<20} {'images':>6}  masks")
    print(f"  {'train':<6} {'good':<20} {train_good:>6}  -")
    print(f"  {'test':<6} {'good':<20} {test_good:>6}  -")

    # One row per defect type, with the mask check
    test_defective = 0
    all_missing = []
    for defect in defect_types:
        n_images = count_pngs(test_dir / defect)
        test_defective += n_images

        missing = find_missing_masks(category_dir, defect)
        all_missing.extend(f"{defect}/{name}" for name in missing)
        mask_status = "OK" if not missing else f"{len(missing)} MISSING"

        print(f"  {'test':<6} {defect:<20} {n_images:>6}  {mask_status}")

    # Short summary line for the category
    print(f"  -> {train_good} train | {test_good + test_defective} test "
          f"({test_good} good, {test_defective} defective, "
          f"{len(defect_types)} defect types)")
    if all_missing:
        print(f"  -> Images without a mask: {all_missing}")

    return {
        "train": train_good,
        "test_good": test_good,
        "test_defective": test_defective,
        "defect_types": len(defect_types),
        "missing_masks": len(all_missing),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Stop early with a clear message if the dataset is not where we expect it
    if not DATA_ROOT.exists():
        raise SystemExit(f"Dataset folder not found: {DATA_ROOT}")

    print(f"Dataset root: {DATA_ROOT}")

    # Explore each category and keep its totals
    totals = {category: explore_category(category) for category in CATEGORIES}

    # Overall summary across the four categories
    print("\nSUMMARY")
    print(f"  {'category':<12} {'train':>6} {'test':>6} {'defect types':>13} {'missing masks':>14}")
    for category, t in totals.items():
        n_test = t["test_good"] + t["test_defective"]
        print(f"  {category:<12} {t['train']:>6} {n_test:>6} "
              f"{t['defect_types']:>13} {t['missing_masks']:>14}")


# Run main() only when the file is executed directly, not when imported
if __name__ == "__main__":
    main()