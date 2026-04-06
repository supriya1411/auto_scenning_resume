from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
from src.config import BASE_DIR

DATABASE_URL = f"sqlite:///{BASE_DIR / 'screening.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class FeedbackRecord(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, index=True, nullable=False)
    candidate_id = Column(String, index=True, nullable=False)
    recruiter_id = Column(String, nullable=False)
    decision = Column(String, nullable=False)          # shortlisted / rejected / on_hold
    model_version = Column(Integer, nullable=True)
    # Features at decision time (for retraining)
    tfidf_sim = Column(Float, nullable=True)
    semantic_sim = Column(Float, nullable=True)
    skill_match = Column(Float, nullable=True)
    years_exp = Column(Float, nullable=True)
    edu_level = Column(Float, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class ScreeningLog(Base):
    __tablename__ = "screening_log"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, index=True)
    candidate_id = Column(String)
    hybrid_score = Column(Float)
    hire_probability = Column(Float)
    tfidf_score = Column(Float)
    semantic_score = Column(Float)
    skill_match_score = Column(Float)
    explanation = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
