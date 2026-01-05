from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base
from datetime import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    age = Column(Integer)
    gender = Column(String)

    pregnancies = Column(Integer)
    glucose = Column(Integer)
    blood_pressure = Column(Integer)
    skin_thickness = Column(Integer)
    insulin = Column(Integer)
    bmi = Column(Float)
    dpf = Column(Float)

    prediction = Column(Integer)
    confidence = Column(Float)
    positive_prob = Column(Float)
    negative_prob = Column(Float)

    created_at = Column(DateTime, default=datetime.utcnow)
