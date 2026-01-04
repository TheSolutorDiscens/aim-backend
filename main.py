# COPYRIGHT (C) © HTTPS://WWW.COMPUTES.COM 2024 . ALL RIGHTS RESERVED.......
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="AIM Diabetes Prediction API")

# CORS (allow frontend calls)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# ------------------ DATA SCHEMA ------------------

class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float


# ------------------ ROUTES ------------------

@app.get("/")
def root():
    return {"message": "AIM FastAPI Backend is Live"}

@app.post("/predict")
def predict(data: DiabetesInput):
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

    return {
        "status": "success",
        "prediction": prediction
    }
