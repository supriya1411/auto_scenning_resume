import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, FeedbackRecord, ScreeningLog
from api.schemas import FeedbackRequest, FeedbackResponse
from src.config import RETRAIN_SAMPLE_THRESHOLD

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    # Check candidate was screened
    log = (
        db.query(ScreeningLog)
        .filter(ScreeningLog.job_id == req.job_id, ScreeningLog.candidate_id == req.candidate_id)
        .order_by(ScreeningLog.created_at.desc())
        .first()
    )

    # Build feedback record (attach features from latest screening log if available)
    record = FeedbackRecord(
        job_id=req.job_id,
        candidate_id=req.candidate_id,
        recruiter_id=req.recruiter_id,
        decision=req.decision,
        tfidf_sim=log.tfidf_score if log else None,
        semantic_sim=log.semantic_score if log else None,
        skill_match=log.skill_match_score if log else None,
    )
    db.add(record)
    db.commit()

    # Count unprocessed feedback (shortlisted + rejected only — excludes on_hold)
    actionable = (
        db.query(FeedbackRecord)
        .filter(FeedbackRecord.decision.in_(["shortlisted", "rejected"]))
        .count()
    )

    return FeedbackResponse(
        success=True,
        message=f"Feedback recorded. Decision: {req.decision}",
        samples_since_retrain=actionable,
    )
