# defect-inspector

Visual defect inspection for industrial parts using **anomaly detection**, with a FastAPI backend and a React UI (planned).

A model learns what a *good* part looks like from defect-free images only, then flags and highlights anything that deviates. This mirrors real production lines, where defect samples are rare and labeling them is expensive.

> **Status: Phase 1 (baseline benchmark).** Training and evaluation scripts are in place. The API and web UI come in later phases. Benchmark numbers will be added to this README once they have been measured; none are claimed yet.

## Roadmap

| Phase | Goal | Status |
|---|---|---|
| 1 | Baseline: train PaDiM and PatchCore on MVTec AD categories close to connector/electronic parts, compare accuracy, speed and GPU memory | Scripts written, results pending |
| 2 | FastAPI `/inspect` endpoint: image in, pass/fail + anomaly score + heatmap out | Planned |
| 3 | React UI: upload a part, see the verdict and highlighted defect region | Planned |
| 4 | Polish: tests, architecture diagram, failure-case analysis, optional EfficientAD | Planned |
| 5 | Deploy a live demo | Planned |

## Setup

Requires Python 3.10+ and, for reasonable training speed, an NVIDIA GPU.

```bash
# 1. Create an environment
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/macOS: source venv/bin/activate)

# 2. Install PyTorch with the CUDA build that matches your driver:
#    https://pytorch.org/get-started/locally/

# 3. Install the rest
pip install -r requirements.txt

# 4. Make the package importable
set PYTHONPATH=src               # Windows cmd   (PowerShell: $env:PYTHONPATH="src"; Linux/macOS: export PYTHONPATH=src)
```

## Train one model on one category

```bash
python -m defect_inspector.train --model padim --category cable
python -m defect_inspector.train --model patchcore --category metal_nut --batch-size 8
```

The MVTec AD dataset is downloaded automatically on first use into `datasets/MVTecAD`. Outputs go to `results/<model>/<category>/` (metrics JSON, checkpoint, heatmap images).

If you run out of GPU memory (the target laptop GPU has 6 GB), lower `--batch-size`. If Windows data loading hangs or errors, pass `--num-workers 0`.

## Run the full comparison

```bash
python -m defect_inspector.benchmark run        # trains every model on cable, transistor, metal_nut
python -m defect_inspector.benchmark report     # writes docs/results.md from the saved metrics
```

## Tests

```bash
pytest
```

The tests cover the results-table builder and do not need a GPU or the dataset.

## Dataset and licensing

This project uses the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset, released under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/): non-commercial use only. If you use it, cite the authors (Bergmann et al., IJCV 2021 and CVPR 2019). The dataset is not included in this repository.

Models are provided by [anomalib](https://github.com/openvinotoolkit/anomalib).
