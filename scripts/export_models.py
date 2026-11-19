"""
export_models.py - Phase 6, step 6a of the defect-inspector project.

Trains PatchCore once per category (same settings and seed as every
evaluation) and EXPORTS each model to a single .pt file. The demo app then
just loads these files: no training when the app starts.

Output (ignored by git: model files are large):
    models/<category>/weights/torch/patchcore.pt

Run from the project root (E:\\defect-inspector):
    python scripts\\export_models.py
"""

from pathlib import Path

from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lightning.pytorch import seed_everything

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"

CATEGORIES = ["screw", "metal_nut", "cable", "transistor"]
SEED = 42


def export_category(category: str) -> Path:
    """Train PatchCore on one category and export it as a torch .pt file."""
    seed_everything(SEED, workers=True)

    datamodule = MVTecAD(root=DATA_ROOT, category=category,
                         train_batch_size=16, eval_batch_size=16, num_workers=0)
    model = Patchcore(backbone="wide_resnet50_2", layers=["layer2", "layer3"],
                      coreset_sampling_ratio=0.1)
    engine = Engine(default_root_dir=RESULTS_DIR, accelerator="gpu", devices=1)

    engine.fit(model=model, datamodule=datamodule)

    # Testing once also sets the decision threshold (good vs defective)
    # that gets saved inside the exported file.
    engine.test(model=model, datamodule=datamodule)

    # Export the whole model (pre-processing + PatchCore + post-processing)
    return engine.export(
        model=model,
        export_type="torch",
        export_root=MODELS_DIR / category,
        model_file_name="patchcore",
    )


def main() -> None:
    for category in CATEGORIES:
        print(f"\n######## {category} ########")
        path = export_category(category)
        print(f"Exported: {path}")


if __name__ == "__main__":
    main()