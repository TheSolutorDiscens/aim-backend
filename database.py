from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os

DATABASE_URL = os.getenv("postgresql://aim_database_6vn7_user:BhHE20vuZj3G6OM49mpxHoNo3gdYLvuR@dpg-d5dojru3jp1c73f0dir0-a.virginia-postgres.render.com/aim_database_6vn7")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()
