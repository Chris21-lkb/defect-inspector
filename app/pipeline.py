"""
pipeline.py - Phase 6, step 6b of the defect-inspector project.

The inspection pipeline used by the demo app, built from the SAME code that
was evaluated in Phases 3-5 (imported from scripts/), so the demo behaves
exactly like the numbers in the README:

    1. PatchCore (exported in step 6a): good or defective + heatmap
    2. Box around the hottest area           (hybrid_a: anomaly_box)
    3. Close-up crop of that box             (hybrid_b: make_view logic)
    4. Qwen3.5-2B multiple-choice naming     (hybrid_b: build_prompt, letter_probabilities)
    5. Bias calibration on blank images      (hybrid_c: measure_bias, calibrate)
"""

import base64
import io
import os
import string
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# Make the evaluated scripts importable (they live in /scripts, not in /app)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from anomalib.deploy import TorchInferencer                      # noqa: E402
from hybrid_a_patchcore_predict import anomaly_box               # noqa: E402
from hybrid_b_vlm_name_defects import (                          # noqa: E402
    DEFECTS, build_prompt, letter_probabilities, load_model,
)
from hybrid_c_calibrated_naming import calibrate, measure_bias   # noqa: E402

CATEGORIES = list(DEFECTS.keys())
VIEW = "crop"   # best view in step 5c (46.8% naming accuracy after calibration)


def load_own_patchcore(path: Path) -> TorchInferencer:
    """Load one of OUR exported PatchCore files (kept on the CPU at first).

    anomalib refuses to load .pt files by default, because loading them runs
    Python's pickle, which can execute code hidden in a malicious file.
    These files are created on this machine by scripts/export_models.py, so
    we trust them: the safety switch is opened ONLY while loading them, then
    closed again. Never point this at a .pt file downloaded from elsewhere.
    """
    os.environ["TRUST_REMOTE_CODE"] = "1"
    try:
        return TorchInferencer(path=path, device="cpu")
    finally:
        os.environ.pop("TRUST_REMOTE_CODE", None)   # close the switch again


class Inspector:
    """Loads every model once, then inspects images on request."""

    def __init__(self) -> None:
        # --- PatchCore: one exported model per category, kept on the CPU.
        # Only the one in use is moved to the GPU (see _patchcore), so the
        # four PatchCore models + the VLM fit together in 6 GB.
        self.patchcore = {}
        for category in CATEGORIES:
            path = MODELS_DIR / category / "weights" / "torch" / "patchcore.pt"
            if not path.exists():
                raise FileNotFoundError(f"{path} missing: run scripts/export_models.py first")
            self.patchcore[category] = load_own_patchcore(path)
        self.on_gpu = None   # category whose PatchCore currently sits on the GPU

        # --- Vision-language model (stays on the GPU)
        self.processor, self.vlm = load_model()
        tokenizer = self.processor.tokenizer

        # --- Prompt, letter tokens and bias per category, computed once
        self.questions = {}
        for category in CATEGORIES:
            prompt, folders = build_prompt(category, VIEW)
            letters = string.ascii_uppercase[:len(folders)]
            letter_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in letters]
            bias = measure_bias(self.processor, self.vlm, prompt, VIEW, letter_ids)
            self.questions[category] = {"prompt": prompt, "folders": folders,
                                        "letter_ids": letter_ids, "bias": bias}

    # ------------------------------------------------------------------
    def _patchcore(self, category: str) -> TorchInferencer:
        """Return the category's PatchCore, moving it to the GPU if needed."""
        if self.on_gpu != category:
            if self.on_gpu is not None:                      # send the previous one back
                old = self.patchcore[self.on_gpu]
                old.model.to("cpu")
                old.device = torch.device("cpu")
            new = self.patchcore[category]
            new.model.to("cuda")
            new.device = torch.device("cuda")
            self.on_gpu = category
        return self.patchcore[category]

    # ------------------------------------------------------------------
    def inspect(self, image: Image.Image, category: str) -> dict:
        """Run the full pipeline on one image and return a JSON-ready dict."""
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category '{category}'. Choose one of {CATEGORIES}")
        image = image.convert("RGB")
        result = {"category": category}

        # 1. PatchCore: score, decision and heatmap
        start = time.perf_counter()
        with torch.no_grad():   # inference only: no gradients, less memory
            prediction = self._patchcore(category).predict(image)
        if torch.cuda.is_available():
            torch.cuda.synchronize()   # wait for the GPU so the timing is honest
        result["patchcore_ms"] = round((time.perf_counter() - start) * 1000, 1)

        heat = prediction.anomaly_map[0].squeeze().detach().cpu().float().numpy()
        result["anomaly_score"] = round(float(prediction.pred_score[0]), 4)
        result["defective"] = bool(prediction.pred_label[0])
        result["heatmap_png"] = overlay_heatmap(image, heat)

        # Good parts stop here: there is no defect to name
        if not result["defective"]:
            result["box"] = None
            result["suggestions"] = []
            result["vlm_ms"] = 0.0
            return result

        # 2-3. Box around the hottest area, then a close-up crop of it
        x0, y0, x1, y1 = anomaly_box(heat)
        w, h = image.size
        result["box"] = [round(v, 4) for v in (x0, y0, x1, y1)]
        crop = image.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))

        # 4-5. VLM multiple choice + calibration
        q = self.questions[category]
        start = time.perf_counter()
        raw = letter_probabilities(self.processor, self.vlm, crop, q["prompt"], q["letter_ids"])
        calibrated = calibrate(raw, q["bias"])
        result["vlm_ms"] = round((time.perf_counter() - start) * 1000, 1)

        # Top 3 suggestions, most likely first, with a readable description
        ranked = sorted(zip(q["folders"], calibrated), key=lambda x: x[1], reverse=True)[:3]
        options = DEFECTS[category]["options"]
        result["suggestions"] = [
            {"defect": folder, "description": options[folder], "probability": round(p, 3)}
            for folder, p in ranked
        ]
        return result


def overlay_heatmap(image: Image.Image, heat: np.ndarray) -> str:
    """Red overlay where the (normalised) anomaly map is above PatchCore's threshold.

    anomalib normalises anomaly maps so that 0.5 = the decision threshold:
    below 0.5 nothing is drawn, above it the red becomes stronger.
    Returned as a base64 PNG string the web page can display directly.
    """
    heat_img = Image.fromarray(np.clip(heat * 255, 0, 255).astype(np.uint8))
    heat_big = np.asarray(heat_img.resize(image.size, Image.BILINEAR)) / 255.0

    alpha = np.clip((heat_big - 0.5) * 2, 0, 1)[..., None] * 0.6   # 0 below threshold
    base = np.asarray(image).astype(np.float32)
    red = np.zeros_like(base)
    red[..., 0] = 255
    blended = (base * (1 - alpha) + red * alpha).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(blended).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")