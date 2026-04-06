import logging
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, FeedbackRecord
from api.schemas import ModelInfoResponse, RetrainRequest, RetrainResponse
from api.routes.screen import get_scorer
from src.config import MODELS_DIR
from src.retrainer import retrain_from_feedback, FEATURE_NAMES
import joblib

logger = logging.getLogger(__name__)
router = APIRouter()


def _load_meta() -> dict:
    meta_path = MODELS_DIR / "model_meta.joblib"
    if not meta_path.exists():
        return {"name": "unknown", "version": 0, "validation_auc": 0.0, "trained_at": "N/A", "n_samples": 0, "flavor": "xgboost"}
    return joblib.load(meta_path)


@router.get("/model/info", response_model=ModelInfoResponse)
def model_info(db: Session = Depends(get_db)):
    meta = _load_meta()
    actionable_count = (
        db.query(FeedbackRecord)
        .filter(FeedbackRecord.decision.in_(["shortlisted", "rejected"]))
        .count()
    )
    return ModelInfoResponse(
        active_model=f"{meta.get('name', 'unknown')}_v{meta.get('version', 0)}",
        version=meta.get("version", 0),
        validation_auc=meta.get("validation_auc", 0.0),
        trained_at=str(meta.get("trained_at", "N/A")),
        n_samples=meta.get("n_samples", 0),
        samples_since_retrain=actionable_count,
    )


@router.post("/retrain", response_model=RetrainResponse)
def trigger_retrain(req: RetrainRequest, db: Session = Depends(get_db)):
    meta = _load_meta()
    current_auc = meta.get("validation_auc", 0.0)

    # Pull feedback with feature snapshots
    records = (
        db.query(FeedbackRecord)
        .filter(FeedbackRecord.decision.in_(["shortlisted", "rejected"]))
        .all()
    )

    if not records:
        raise HTTPException(status_code=400, detail="No feedback data available for retraining.")

    rows = []
    for r in records:
        if r.tfidf_sim is not None:
            rows.append({
                "tfidf_sim": r.tfidf_sim,
                "semantic_sim": r.semantic_sim or 0.0,
                "skill_match": r.skill_match or 0.0,
                "years_exp": r.years_exp or 0.1,
                "edu_level": r.edu_level or 0.0,
                "decision": r.decision,
            })

    if not rows:
        raise HTTPException(
            status_code=400,
            detail="Feedback records lack feature snapshots. Screen candidates first, then collect feedback.",
        )

    feedback_df = pd.DataFrame(rows)
    result = retrain_from_feedback(
        feedback_df=feedback_df,
        current_auc=current_auc,
        flavor=req.model_flavor,
        models_dir=MODELS_DIR,
    )

    # Reload scorer if model was promoted
    if result.get("promoted"):
        global_scorer = get_scorer()
        global_scorer._loaded = False
        global_scorer.load()

    return RetrainResponse(**result)
