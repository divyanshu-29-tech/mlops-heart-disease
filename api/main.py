import logging
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"

FEATURE_NAMES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak",
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD MODEL
# ============================================================

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully from %s", MODEL_PATH)
except Exception:
    logger.exception("Failed to load model: %s")
    model = None


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps API for heart disease prediction",
    version="1.0.0",
)
Instrumentator().instrument(app).expose(app)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class HeartDiseaseInput(BaseModel):
    age: float
    trestbps: float
    chol: float
    thalach: float
    oldpeak: float

    sex: int
    cp: int
    fbs: int
    restecg: int
    exang: int
    slope: int
    ca: int
    thal: int


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    """
    Health check endpoint.
    """

    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(data: HeartDiseaseInput):
    """
    Predict heart disease risk.

    Returns:
        prediction:
            0 = No heart disease
            1 = Heart disease

        confidence:
            Probability associated with the predicted class.
    """

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    try:
        # ----------------------------------------------------
        # Convert request into a pandas DataFrame
        # ----------------------------------------------------

        input_data = {
            "age": data.age,
            "trestbps": data.trestbps,
            "chol": data.chol,
            "thalach": data.thalach,
            "oldpeak": data.oldpeak,
            "sex": data.sex,
            "cp": data.cp,
            "fbs": data.fbs,
            "restecg": data.restecg,
            "exang": data.exang,
            "slope": data.slope,
            "ca": data.ca,
            "thal": data.thal,
        }

        # IMPORTANT:
        # The model pipeline expects column names.
        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURE_NAMES,
        )

        logger.info(
            "Prediction request received: %s",
            input_data,
        )

        # ----------------------------------------------------
        # Make prediction
        # ----------------------------------------------------

        prediction = int(model.predict(input_df)[0])

        # ----------------------------------------------------
        # Calculate confidence
        # ----------------------------------------------------

        probabilities = model.predict_proba(input_df)[0]

        confidence = float(
            probabilities[prediction]
        )

        # ----------------------------------------------------
        # Human-readable result
        # ----------------------------------------------------

        if prediction == 1:
            result = "Heart disease detected"
        else:
            result = "No heart disease detected"

        logger.info(
            "Prediction=%s, confidence=%.4f",
            prediction,
            confidence,
        )

        return {
            "prediction": prediction,
            "result": result,
            "confidence": round(confidence, 4),
        }

    except Exception as exc:
        logger.exception(
            "Prediction failed: %s"
        )

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc!s}",
        )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Heart Disease Prediction API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }
