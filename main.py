# COPYRIGHT (C) © HTTPS://WWW.COMPUTES.COM 2024 . ALL RIGHTS RESERVED.......
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
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

db = SessionLocal()

# ------------------ LOGGING ------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("AIM")

# ------------------ LOAD MODEL ------------------

try:
    model = joblib.load("model_v1.pkl")
    scaler = joblib.load("scaler_v1.pkl")
    logger.info("Model and scaler loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model/scaler: {e}")
    raise RuntimeError("Model loading failed")

# ------------------ DATA SCHEMA ------------------

class DiabetesInput(BaseModel):
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
def predict(data: DiabetesInput):
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

        scaled_features = scaler.transform(features)

        prediction = int(model.predict(scaled_features)[0])
        proba = model.predict_proba(scaled_features)[0]

        confidence = round(max(proba) * 100, 2)

        logger.info(
            f"Prediction made | Prediction={prediction} | Confidence={confidence:.2f}"
        )

        record = Prediction(
            name=data.Name,
            age=data.Age,
            gender=data.Gender,
            pregnancies=data.Pregnancies,
            glucose=data.Glucose,
            blood_pressure=data.BloodPressure,
            skin_thickness=data.SkinThickness,
            insulin=data.Insulin,
            bmi=data.BMI,
            dpf=data.DiabetesPedigreeFunction,
            prediction=prediction,
            confidence=confidence,
            positive_prob=proba[1] * 100,
            negative_prob=proba[0] * 100,
        )

        db.add(record)
        db.commit()
        db.refresh(record)

        return {
            "status": "success",
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "negative": round(proba[0] * 100, 2),
                "positive": round(proba[1] * 100, 2)
            }
        }

     

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Prediction failed")





@app.get("/history")
def history():
    db = SessionLocal()
    records = db.query(Prediction).order_by(Prediction.created_at.desc()).limit(20).all()
    db.close()

    
    return records

