from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base
from datetime import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    Name = Column(String, nullable=False)
    Age = Column(Float, nullable=False)
    Gender = Column(String, nullable=False)

    Pregnancies = Column(Float, nullable=False)
    Glucose = Column(Float, nullable=False)
    BloodPressure = Column(Float, nullable=False)
    SkinThickness = Column(Float, nullable=False)
    Insulin = Column(Float, nullable=False)
    BMI = Column(Float, nullable=False)
    DiabetesPedigreeFunction = Column(Float, nullable=False)

    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    positive_prob = Column(Float, nullable=False)
    negative_prob = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
