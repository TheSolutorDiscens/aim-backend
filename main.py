# COPYRIGHT (C) © HTTPS://WWW.COMPUTES.COM 2026 . ALL RIGHTS RESERVED.......
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
from sqlalchemy.orm import Session
import numpy as np
import joblib
import logging
from database import SessionLocal
from models import Prediction

# ------------------ APP SETUP ------------------

app = FastAPI(
    title="AIM Diabetes Prediction API",
    version="1.0.0",
    description="Machine Learning API for Diabetes Prediction"
)

# CORS (restrict later in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ LOGGING ------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AIM")


# -------------------------------------------------
# DATABASE DEPENDENCY
# -------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ------------------ LOAD MODEL ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model_v1.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler_v1.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    logging.info("Model and scaler loaded successfully")
except Exception as e:
    logging.error(f"Failed to load model/scaler: {e}")
    raise RuntimeError("Model initialization failed")


# ------------------ DATA SCHEMA ------------------

class DiabetesInput(BaseModel):
    Name: str = Field(..., min_length=1, max_length=100)
    Gender: str = Field(..., regex="^(Male|Female|Other)$")
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=100)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5)
    Age: float = Field(..., ge=0, le=120)

# ------------------ ROUTES ------------------

@app.get("/")
def root():
    return {"message": "AIM FastAPI Backend is Live"}

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AIM Backend",
        "model_loaded": True
    }

@app.post("/api/v1/predict")
def predict(data: DiabetesInput, db: Session = Depends(get_db)):
    try:
        features = np.array([[
            data.Pregnancies,
            data.Glucose,
            data.BloodPressure,
            data.SkinThickness,
            data.Insulin,
            data.BMI,
            data.DiabetesPedigreeFunction,
            data.Age
        ]])

        scaled = scaler.transform(features)
        prediction = int(model.predict(scaled)[0])
        proba = model.predict_proba(scaled)[0]

        confidence = round(max(proba) * 100, 2)

        record = Prediction(
            name=data.Name,
            gender=data.Gender,
            age=data.Age,
            pregnancies=data.Pregnancies,
            glucose=data.Glucose,
            blood_pressure=data.BloodPressure,
            skin_thickness=data.SkinThickness,
            insulin=data.Insulin,
            bmi=data.BMI,
            dpf=data.DiabetesPedigreeFunction,
            prediction=prediction,
            confidence=confidence,
            positive_prob=round(proba[1] * 100, 2),
            negative_prob=round(proba[0] * 100, 2),
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "positive": round(proba[1] * 100, 2),
                "negative": round(proba[0] * 100, 2)
            }
        }

    except Exception as e:
        logger.exception("Prediction error" + e)
        raise HTTPException(500, "Prediction failed")





@app.get("/history")
def history(db: Session = Depends(get_db)):
    records = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )

    
    return records

