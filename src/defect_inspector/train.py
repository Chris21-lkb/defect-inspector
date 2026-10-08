"""Train and evaluate one anomaly-detection model on one MVTec AD category.

Anomaly detection models learn what a *good* part looks like from defect-free
images only, then flag anything that deviates from it. No defect labels or
masks are needed for training; the labeled defects in the test split are only
used to measure how well the model works.

Example (run from the repository root):

    python -m defect_inspector.train --model padim --category cable
    python -m defect_inspector.train --model patchcore --category metal_nut

Results (metrics JSON, checkpoint, heatmap images) are written to
``results/<model>/<category>/``.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Padim, Patchcore

# Models supported in Phase 1. EfficientAD is left out on purpose: it needs an
# extra ImageNet-subset download (``imagenet_dir``) and is added in a later phase.
MODELS = {
    "padim": Padim,
    "patchcore": Patchcore,
}


def parse_args() -> argparse.Namespace:
    """Read command line options."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="padim")
    parser.add_argument("--category", default="cable", help="MVTec AD category, e.g. cable, transistor, metal_nut")
    parser.add_argument("--data-root", default="datasets/MVTecAD", help="Where MVTec AD is (or will be) downloaded")
    parser.add_argument("--results-dir", default="results", help="Root folder for outputs")
    parser.add_argument("--batch-size", type=int, default=16, help="Lower this if you run out of GPU memory")
    parser.add_argument("--num-workers", type=int, default=4, help="Data loader workers (use 0 if Windows complains)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> dict:
    """Train, test, and return a dictionary describing the run."""
    out_dir = Path(args.results_dir) / args.model / args.category
    out_dir.mkdir(parents=True, exist_ok=True)

    # The datamodule downloads MVTec AD on first use and handles the
    # train (good only) / test (good + defective) split.
    datamodule = MVTecAD(
        root=args.data_root,
        category=args.category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = MODELS[args.model]()
    engine = Engine(default_root_dir=str(out_dir))

    # Track GPU memory so the README can report honest numbers for this hardware.
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    start = time.time()
    engine.fit(model=model, datamodule=datamodule)
    train_seconds = time.time() - start

    # engine.test returns a list with one dict of metric name -> value
    # (for example image and pixel AUROC; the exact keys come from anomalib).
    test_results = engine.test(model=model, datamodule=datamodule)
    metrics = {k: float(v) for k, v in test_results[0].items()}

    record = {
        "model": args.model,
        "category": args.category,
        "device": torch.cuda.get_device_name(0) if use_cuda else "cpu",
        "train_seconds": round(train_seconds, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 1e6) if use_cuda else None,
        "batch_size": args.batch_size,
        "metrics": metrics,
    }

    # Save a machine-readable record so benchmark.py can build the results table.
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    print(json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    # The __main__ guard matters on Windows: data loader workers re-import this
    # file, and without the guard they would start training again.
    run(parse_args())
