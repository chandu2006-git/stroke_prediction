"""
StrokeGuard FastAPI backend.

Endpoints:
  GET  /api/health    - health check + model status
  POST /api/predict   - patient risk prediction + SHAP/LIME explanations

The CatBoost model is loaded ONCE at startup (model_service.ModelService).
Model path is configurable via STROKEGUARD_MODEL_PATH env var.

CORS is enabled so the local frontend can call the API.
No patient data is stored. No authentication. No database.
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .schemas import (
    PatientInput,
    PredictionResponse,
    HealthResponse,
    ExplanationBlock,
)
from .model_service import ModelService, ModelNotReady
from .explanation_service import ExplanationService

logger = logging.getLogger("strokeguard.api")

app = FastAPI(
    title="StrokeGuard API",
    description="Explainable stroke risk prediction research prototype.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_service: Optional[ModelService] = None
explanation_service: Optional[ExplanationService] = None
MODEL_NAME = "CatBoost Classifier"


@app.on_event("startup")
def _startup() -> None:
    """Load model and prepare explanation service once at startup."""
    global model_service, explanation_service
    logging.basicConfig(level=logging.INFO)

    model_service = ModelService()
    try:
        model_service.load()
    except ModelNotReady:
        logger.warning("Model not ready at startup. See /api/health.")

    background_df = None
    if model_service.is_ready():
        import pandas as pd
        import numpy as np

        rng = np.random.default_rng(42)
        n_bg = 120
        background_df = pd.DataFrame({
            "gender": rng.choice(["Male", "Female", "Other"], n_bg),
            "age": rng.normal(50, 18, n_bg).clip(1, 100),
            "hypertension": rng.choice([0, 1], n_bg),
            "heart_disease": rng.choice([0, 1], n_bg),
            "ever_married": rng.choice(["Yes", "No"], n_bg),
            "work_type": rng.choice(
                ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"], n_bg
            ),
            "Residence_type": rng.choice(["Urban", "Rural"], n_bg),
            "avg_glucose_level": rng.normal(90, 30, n_bg).clip(50, 300),
            "bmi": rng.normal(28, 7, n_bg).clip(12, 60),
            "bmi_missing": rng.choice([0, 1], n_bg),
            "smoking_status": rng.choice(
                ["formerly smoked", "never smoked", "smokes", "Unknown"], n_bg
            ),
        })

    explanation_service = ExplanationService(
        model=model_service.model if model_service.is_ready() else None,
        background_data=background_df,
    )


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if model_service is not None:
        model_path = str(model_service.model_path)  # Path -> str for the JSON response
    else:
        model_path = str(DEFAULT_MODEL_PATH)
    return HealthResponse(
        status="ok" if (model_service and model_service.is_ready()) else "model_missing",
        model_loaded=bool(model_service and model_service.is_ready()),
        model_path=model_path,
        model_name=MODEL_NAME,
    )


@app.post("/api/predict", response_model=PredictionResponse)
def predict(payload: PatientInput) -> PredictionResponse:
    if not model_service or not model_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available. The trained CatBoost model "
                   "file (models/strokeguard_catboost.cbm) is missing. "
                   "Please add it and restart the server.",
        )

    try:
        data = payload.model_dump()
        result = model_service.predict(data)
        df = result["input_dataframe"]

        shap_block = explanation_service.shap_explain(df) if explanation_service else {
            "available": False, "message": "Explanation service unavailable.", "contributions": []
        }
        lime_block = explanation_service.lime_explain(df) if explanation_service else {
            "available": False, "message": "Explanation service unavailable.", "contributions": []
        }

        patient_summary = {
            "gender": data["gender"],
            "age": data["age"],
            "hypertension": "Yes" if data["hypertension"] == 1 else "No",
            "heart_disease": "Yes" if data["heart_disease"] == 1 else "No",
            "ever_married": data["ever_married"],
            "work_type": data["work_type"],
            "residence_type": data["residence_type"],
            "avg_glucose_level": data["avg_glucose_level"],
            "bmi": data["bmi"] if data["bmi"] is not None else "Not provided",
            "bmi_missing": data["bmi_missing"],
            "smoking_status": data["smoking_status"],
        }

        timestamp = datetime.now(timezone.utc).isoformat()

        return PredictionResponse(
            prediction=result["prediction"],
            prediction_label=result["prediction_label"],
            probability=result["probability"],
            model_name=MODEL_NAME,
            timestamp=timestamp,
            shap_explanation=ExplanationBlock(**shap_block),
            lime_explanation=ExplanationBlock(**lime_block),
            patient_summary=patient_summary,
        )

    except ModelNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction could not be completed.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False)