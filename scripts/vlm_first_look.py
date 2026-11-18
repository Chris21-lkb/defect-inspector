"""
vlm_first_look.py - Step 4a of the defect-inspector project.

First contact with a vision-language model (VLM): Qwen3.5-2B looks at two
screws (one good, one defective) and says whether each one is defective.
No training at all: this is "zero-shot" inspection.

The first run downloads the model (about 4 GB) into your Hugging Face cache.
"""

import time
from pathlib import Path

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "datasets"

# Model name on Hugging Face
MODEL_ID = "Qwen/Qwen3.5-2B"

# Two test images: we already know the true answer for each one
IMAGES = [
    ("good", DATA_ROOT / "screw" / "test" / "good" / "000.png"),
    ("defective (scratch_head)", DATA_ROOT / "screw" / "test" / "scratch_head" / "000.png"),
]

# The instruction given to the model. A fixed answer format (first line =
# one keyword) makes the answer easy to read automatically later.
# PROMPT = (
#     "You are a quality inspector on a production line. "
#     "This image shows one metal screw photographed from above. "
#     "Is the screw defective? "
#     "On the first line answer only DEFECTIVE or GOOD. "
#     "On the second line, in one short sentence, describe the defect and where it is, "
#     "or write 'no defect'."
# )

PROMPT = (
    "Describe the condition of this screw. "
    "On the first line answer only GOOD or DEFECTIVE. "
    "On the second line explain why in one short sentence."
)


def load_model():
    """Load the processor (turns image + text into tensors) and the model."""
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,  # 16-bit weights: half the memory of 32-bit, same quality
        device_map="cuda",     # put the whole model on the GPU
    )
    model.eval()               # inference mode (no training behaviour)
    return processor, model


def ask(processor, model, image_path: Path) -> str:
    """Show one image to the model with PROMPT and return its text answer."""
    # Chat format: one user message containing an image and a text question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "path": str(image_path)},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]

    # The chat template turns the messages into model inputs (token ids + pixels)
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,  # adds the "assistant, your turn" marker
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    # Generate the answer. do_sample=False = greedy decoding: the model always
    # picks its most likely next word, so the same image gives the same answer
    # every run. That is what we want for a fair, repeatable evaluation.
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=60, do_sample=False)

    # The output contains the prompt + the answer: keep only the new tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    start = time.perf_counter()
    processor, model = load_model()
    print(f"Loaded in {time.perf_counter() - start:.1f} s")

    for truth, image_path in IMAGES:
        start = time.perf_counter()
        answer = ask(processor, model, image_path)
        seconds = time.perf_counter() - start

        print(f"\n--- {image_path.parent.name}/{image_path.name}  (truth: {truth})")
        print(answer)
        print(f"[{seconds:.1f} s]")

    # Peak GPU memory used by the model during this run
    peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\nPeak GPU memory: {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()