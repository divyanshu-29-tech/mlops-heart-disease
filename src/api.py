from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "logistic_regression.joblib"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Heart Disease Prediction API",
    description="MLOps API for heart disease classification",
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PatientData(BaseModel):

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: float
    thal: float


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "logistic_regression",
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(patient: PatientData):

    input_data = pd.DataFrame(
        [patient.model_dump()]
    )

    prediction = model.predict(
        input_data
    )[0]

    probability = model.predict_proba(
        input_data
    )[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "model": "logistic_regression",
    }
