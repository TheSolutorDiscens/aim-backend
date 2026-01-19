from sqlalchemy import Column, Integer, Float, String, DateTime
from database import Base
from datetime import datetime

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String, nullable=False)
    age = Column(Float, nullable=False)
    gender = Column(String, nullable=False)

    pregnancies = Column(Float, nullable=False)
    glucose = Column(Float, nullable=False)
    blood_pressure = Column(Float, nullable=False)
    skin_thickness = Column(Float, nullable=False)
    insulin = Column(Float, nullable=False)
    bmi = Column(Float, nullable=False)
    dpf = Column(Float, nullable=False)

    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    positive_prob = Column(Float, nullable=False)
    negative_prob = Column(Float, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
