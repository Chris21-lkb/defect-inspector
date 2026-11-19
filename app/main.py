"""
main.py - Phase 6, step 6c of the defect-inspector project.

FastAPI web service around the inspection pipeline.

Endpoints:
    GET  /            the upload page (app/static/index.html)
    GET  /health      quick check that the models are loaded
    GET  /categories  the part types the service can inspect
    POST /inspect     image file + category -> verdict, heatmap, defect suggestions

Run from the project root (E:\\defect-inspector):
    uvicorn app.main:app --port 8000
Then open http://127.0.0.1:8000 in a browser.
"""

import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from app.pipeline import CATEGORIES, Inspector

STATIC_DIR = Path(__file__).resolve().parent / "static"

# The models are loaded ONCE when the server starts, not on every request
# (loading takes ~20 s; an inspection takes about a second).
inspector: Inspector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Runs at server start-up (before 'yield') and shutdown (after it)."""
    global inspector
    print("Loading models (PatchCore x4 + Qwen3.5-2B)...")
    inspector = Inspector()
    print("Models loaded. Open http://127.0.0.1:8000")
    yield


app = FastAPI(title="defect-inspector", lifespan=lifespan)


@app.get("/")
def home() -> FileResponse:
    """Serve the upload page."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Simple status check (useful for monitoring and for tests)."""
    return {"status": "ok" if inspector is not None else "loading"}


@app.get("/categories")
def categories() -> list[str]:
    """Part types with a trained PatchCore model."""
    return CATEGORIES


# Plain "def" (not "async def"): FastAPI then runs it in a worker thread,
# so a slow GPU call does not freeze the whole server.
@app.post("/inspect")
def inspect(file: UploadFile = File(...), category: str = Form(...)) -> dict:
    """Inspect one uploaded image of the given part type."""
    if inspector is None:
        raise HTTPException(status_code=503, detail="Models are still loading")
    if category not in CATEGORIES:
        raise HTTPException(status_code=400, detail=f"category must be one of {CATEGORIES}")

    # Read the upload and make sure it really is an image
    try:
        image = Image.open(io.BytesIO(file.file.read()))
        image.load()
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="The file is not a readable image")

    return inspector.inspect(image, category)