# COPYRIGHT (C) © HTTPS://WWW.COMPUTES.COM 2026 . ALL RIGHTS RESERVED.......
from fastapi import FastAPI, HTTPException, Depends , Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field
import os
from sqlalchemy.orm import Session
from typing import Literal
import numpy as np
import joblib
import logging
from database import SessionLocal, engine
from models import Prediction, Base

# ------------------ APP SETUP ------------------

app = FastAPI(
    title="AIM Diabetes Prediction API",
    version="1.0.0",
    description="Machine Learning API for Diabetes Prediction"
)


# CORS (restrict later in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


Base.metadata.create_all(bind=engine)


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
    Age: float = Field(..., ge=0, le=120)
    Gender: Literal["Male", "Female", "Other"]
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=300)
    BloodPressure: float = Field(..., ge=0, le=200)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=100)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=2.5)

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

        confidence = float(round(max(proba) * 100, 2))
        positive_prob = float(round(proba[1] * 100, 2))
        negative_prob = float(round(proba[0] * 100, 2))

        record = Prediction(
            Name=data.Name,
            Age=int(data.Age),
            Gender=data.Gender,
            Pregnancies=int(data.Pregnancies),
            Glucose=float(data.Glucose),
            BloodPressure=float(data.BloodPressure),
            SkinThickness=float(data.SkinThickness),
            Insulin=float(data.Insulin),
            BMI=float(data.BMI),
            DiabetesPedigreeFunction=float(data.DiabetesPedigreeFunction),
            prediction=prediction,
            confidence=confidence,
            positive_prob=positive_prob,
            negative_prob=negative_prob,
        )

        try:
            db.add(record)
            db.commit()
            db.refresh(record)
        except Exception :
            db.rollback()  
            logger.error("Database insert failed", exc_info=True)
        finally:
            db.close()



        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": {
                "positive": round(proba[1] * 100, 2),
                "negative": round(proba[0] * 100, 2)
            }
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed")






@app.options("/{path:path}")
async def options_handler(path: str, request: Request):
    return PlainTextResponse("", status_code=200)



@app.get("/history")
def history(db: Session = Depends(get_db)):
    records = (
        db.query(Prediction)
        .order_by(Prediction.created_at.desc())
        .limit(20)
        .all()
    )


    return records

