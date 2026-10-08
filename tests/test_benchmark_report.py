"""Tests for the results-table builder (no GPU, dataset or training needed)."""

import json
import sys
from pathlib import Path

import pytest

# Make src/ importable when running `pytest` from the repository root
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defect_inspector.benchmark import build_report  # noqa: E402


def _write_record(root: Path, model: str, category: str, metrics: dict) -> None:
    """Create a fake metrics.json the same way train.py would."""
    folder = root / model / category
    folder.mkdir(parents=True)
    record = {
        "model": model,
        "category": category,
        "device": "test-device",
        "train_seconds": 12.3,
        "peak_gpu_memory_mb": 456,
        "batch_size": 16,
        "metrics": metrics,
    }
    (folder / "metrics.json").write_text(json.dumps(record), encoding="utf-8")


def test_report_contains_all_runs(tmp_path):
    _write_record(tmp_path, "padim", "cable", {"image_AUROC": 0.9, "pixel_AUROC": 0.95})
    _write_record(tmp_path, "patchcore", "cable", {"image_AUROC": 0.99, "pixel_AUROC": 0.98})
    output = tmp_path / "docs" / "results.md"

    build_report(str(tmp_path), str(output))

    text = output.read_text(encoding="utf-8")
    assert "padim" in text and "patchcore" in text
    assert "0.990" in text  # metrics are formatted to 3 decimals
    assert "image_AUROC" in text and "pixel_AUROC" in text


def test_report_fails_clearly_when_no_results(tmp_path):
    with pytest.raises(SystemExit):
        build_report(str(tmp_path), str(tmp_path / "out.md"))
