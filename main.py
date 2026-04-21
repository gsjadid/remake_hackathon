"""
main.py
FastAPI entry point for the ZEISS Smart Energy Assistant.

Run:
    python main.py

Server starts at http://localhost:8000
  GET  /         → ZEISS workflow energy dashboard
  POST /analyze  → upload CSV, returns segment analysis JSON
  GET  /health   → model readiness probe
"""

import io
import os
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.data_processor import (
    aggregate_to_segments,
    detect_and_load,
    load_training_data,
)
from app.hybrid_engine import analyze_segment
from app.ml_model import model as mlp_model

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TRAINING_DIR = BASE_DIR / "data" / "training"
TEMPLATES_DIR = BASE_DIR / "templates"


# ── Startup: train MLP from S1-S10 ───────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train the MLP model from training CSVs before accepting requests."""
    print("[ZEISS AI] Loading training data…")
    try:
        raw_df = load_training_data(str(TRAINING_DIR))
        segments_df = aggregate_to_segments(raw_df, has_label=True)
        stats = mlp_model.train(segments_df)
        print(
            f"[ZEISS AI] MLP trained: accuracy={stats['accuracy']:.2%}  "
            f"F1={stats['f1_macro']:.2%}  "
            f"({stats['n_train']} train / {stats['n_val']} val segments)"
        )
    except Exception as exc:
        print(f"[ZEISS AI] WARNING — training failed: {exc}")
        traceback.print_exc()
    yield
    # Nothing to clean up


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ZEISS Workflow Energy Intelligence",
    lifespan=lifespan,
)
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_ready": mlp_model.is_trained,
        "model_accuracy": round(mlp_model.accuracy, 4) if mlp_model.is_trained else None,
        "model_f1": round(mlp_model.f1_macro, 4) if mlp_model.is_trained else None,
    }


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Accept a workflow CSV (raw telemetry or pre-aggregated segments).
    Returns full analysis JSON with per-segment recommendations and summary stats.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a CSV file.")

    raw_bytes = await file.read()
    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    return _run_analysis(raw_bytes, scenario_name=Path(file.filename).stem)


# Allowed demo scenario codes exposed on the front-end
_SAMPLE_SCENARIOS = {
    "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S13",
}


@app.get("/analyze-sample")
async def analyze_sample(name: str = Query(..., description="Scenario code, e.g. S1, S4, S9")):
    """
    Run the same analysis pipeline on a bundled demo scenario taken from
    data/training/test_segments.csv (pre-aggregated).
    """
    code = name.strip().upper()
    if code not in _SAMPLE_SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Unknown sample scenario: {name}")

    segments_csv = TRAINING_DIR / "test_segments.csv"
    if not segments_csv.exists():
        raise HTTPException(status_code=500, detail="Demo data not available on server.")

    try:
        df = pd.read_csv(segments_csv, low_memory=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read demo data: {exc}")

    if "scenario_code" not in df.columns:
        raise HTTPException(status_code=500, detail="Demo CSV missing scenario_code column.")

    subset = df[df["scenario_code"].astype(str).str.upper() == code]
    if len(subset) == 0:
        raise HTTPException(status_code=404, detail=f"No demo segments for scenario {code}.")

    scenario_label = str(subset["scenario_name"].iloc[0]) if "scenario_name" in subset.columns else code
    buf = io.StringIO()
    subset.to_csv(buf, index=False)
    return _run_analysis(buf.getvalue().encode("utf-8"),
                         scenario_name=f"{code} · {scenario_label}")


def _run_analysis(raw_bytes: bytes, scenario_name: str) -> JSONResponse:
    """Shared analysis pipeline for uploaded and sample CSVs."""
    try:
        segments_df, was_raw = detect_and_load(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if len(segments_df) == 0:
        raise HTTPException(status_code=422, detail="No segments found in file.")

    results = []
    for _, row in segments_df.iterrows():
        results.append(analyze_segment(row.to_dict()))

    total_energy_wh = sum(r["total_energy_wh"] for r in results)
    total_savings_wh = sum(r["estimated_savings_wh"] for r in results)
    savings_pct = (total_savings_wh / total_energy_wh * 100) if total_energy_wh > 0 else 0.0
    spike_count = sum(1 for r in results if r["is_energy_spike"])
    issues_count = sum(1 for r in results if r["recommended_action"] != "no_action")

    return JSONResponse(content={
        "scenario_name": scenario_name,
        "was_raw_telemetry": was_raw,
        "segments": results,
        "total_segments": len(results),
        "issues_count": issues_count,
        "total_energy_wh": round(total_energy_wh, 4),
        "total_savings_wh": round(total_savings_wh, 4),
        "savings_pct": round(savings_pct, 2),
        "spike_count": spike_count,
        "model_accuracy": round(mlp_model.accuracy, 4) if mlp_model.is_trained else 0.0,
        "model_f1": round(mlp_model.f1_macro, 4) if mlp_model.is_trained else 0.0,
        "model_ready": mlp_model.is_trained,
    })


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=3000, reload=False)
