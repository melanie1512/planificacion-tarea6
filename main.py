import os
import joblib
import pandas as pd
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

MODEL_PATH = os.getenv("MODEL_PATH", "heart_model.pkl")

ALLOWED_SEX = {"M", "F"}
ALLOWED_CHEST = {"TA", "ATA", "NAP", "ASY"}
ALLOWED_ECG = {"Normal", "ST", "LVH"}
ALLOWED_EXANG = {"Y", "N"}
ALLOWED_SLOPE = {"Up", "Flat", "Down"}

class HeartInput(BaseModel):
    Age: int = Field(..., ge=1, le=120)
    Sex: str
    ChestPainType: str
    RestingBP: int = Field(..., ge=0, le=300)
    Cholesterol: int = Field(..., ge=0, le=1000)
    FastingBS: int = Field(..., ge=0, le=1, description="0 o 1")
    RestingECG: str
    MaxHR: int = Field(..., ge=0, le=250)
    ExerciseAngina: str
    Oldpeak: float = Field(..., ge=-5, le=10)
    ST_Slope: str

    @field_validator("Sex")
    @classmethod
    def _val_sex(cls, v):
        v = str(v).strip().upper()
        if v not in ALLOWED_SEX:
            raise ValueError(f"Sex debe ser uno de {sorted(ALLOWED_SEX)}")
        return v

    @field_validator("ChestPainType")
    @classmethod
    def _val_chest(cls, v):
        v = str(v).strip().upper()
        if v not in ALLOWED_CHEST:
            raise ValueError(f"ChestPainType debe ser uno de {sorted(ALLOWED_CHEST)}")
        return v

    @field_validator("RestingECG")
    @classmethod
    def _val_ecg(cls, v):
        v = str(v).strip()
        mapping = {"normal":"Normal", "st":"ST", "lvh":"LVH", "LVH":"LVH", "ST":"ST", "Normal":"Normal"}
        v = mapping.get(v, v)
        if v not in ALLOWED_ECG:
            raise ValueError(f"RestingECG debe ser uno de {sorted(ALLOWED_ECG)}")
        return v

    @field_validator("ExerciseAngina")
    @classmethod
    def _val_exang(cls, v):
        v = str(v).strip().upper()
        if v not in ALLOWED_EXANG:
            raise ValueError(f"ExerciseAngina debe ser uno de {sorted(ALLOWED_EXANG)}")
        return v

    @field_validator("ST_Slope")
    @classmethod
    def _val_slope(cls, v):
        v = str(v).strip().capitalize()
        if v not in ALLOWED_SLOPE:
            raise ValueError(f"ST_Slope debe ser uno de {sorted(ALLOWED_SLOPE)}")
        return v

class PredictResponse(BaseModel):
    prediction: int

class BatchRequest(BaseModel):
    items: List[HeartInput]

app = FastAPI(title="Heart Failure Prediction API", version="1.0.0")

# Cargar modelo al iniciar
model = None

@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}. Ejecuta train_heart.py primero.")
    model = joblib.load(MODEL_PATH)

@app.get("/health")
def health():
    ok = model is not None
    return {"status": "ok" if ok else "not_ready", "model_loaded": ok}

_COLUMNS = ["Age","Sex","ChestPainType","RestingBP","Cholesterol","FastingBS","RestingECG","MaxHR","ExerciseAngina","Oldpeak","ST_Slope"]

def _predict_df(df: pd.DataFrame, threshold: float):
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        preds = model.predict(df[_COLUMNS])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en inferencia: {e}")
    return preds

@app.post("/predict", response_model=PredictResponse)
def predict_one(item: HeartInput, threshold: float = Query(0.5, ge=0, le=1)):
    df = pd.DataFrame([item.model_dump()])
    preds = _predict_df(df, threshold)
    return PredictResponse(prediction=int(preds[0]), threshold=threshold)

@app.post("/predict_batch")
def predict_batch(req: BatchRequest, threshold: float = Query(0.5, ge=0, le=1)):
    rows = [it.model_dump() for it in req.items]
    df = pd.DataFrame(rows)
    preds = _predict_df(df, threshold)
    return {"items": [
        {"prediction": int(p)} for p in preds
    ], "threshold": threshold}
