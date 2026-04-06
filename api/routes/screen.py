import json
import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.database import get_db, ScreeningLog, FeedbackRecord
from api.schemas import ScreenRequest, ScreenResponse, CandidateResult, ComponentScores, ExplanationOut
from src.scorer import HybridScorer
from src.explainer import Explainer
import joblib

logger = logging.getLogger(__name__)
router = APIRouter()

# Shared scorer instance (loaded once at startup via app state)
_scorer: HybridScorer = None
_explainer: Explainer = None


def get_scorer() -> HybridScorer:
    global _scorer
    if _scorer is None:
        _scorer = HybridScorer()
        _scorer.load()
    return _scorer


def get_explainer() -> Explainer:
    global _explainer
    if _explainer is None:
        scorer = get_scorer()
        _explainer = Explainer(model=scorer.model, model_name=scorer.model_name)
    return _explainer


@router.post("/screen", response_model=ScreenResponse)
def screen_candidates(req: ScreenRequest, db: Session = Depends(get_db)):
    scorer = get_scorer()
    explainer = get_explainer()
    weights = req.score_weights.model_dump() if req.score_weights else None

    results = []
    for resume_input in req.resumes:
        try:
            score_data = scorer.score(
                resume_text=resume_input.text,
                job_description=req.job_description,
                required_skills=req.required_skills,
                weights=weights,
            )
            explanation = explainer.explain(
                features=score_data["_features"],
                hire_probability=score_data["hire_probability"],
                required_skills=req.required_skills,
                resume_text=resume_input.text,
            )
            results.append({
                "candidate_id": resume_input.candidate_id,
                "hybrid_score": score_data["hybrid_score"],
                "hire_probability": score_data["hire_probability"],
                "component_scores": score_data["component_scores"],
                "explanation": explanation,
                "_features": score_data["_features"],
            })

            # Persist screening log
            log = ScreeningLog(
                job_id=req.job_id,
                candidate_id=resume_input.candidate_id,
                hybrid_score=score_data["hybrid_score"],
                hire_probability=score_data["hire_probability"],
                tfidf_score=score_data["component_scores"]["tfidf_score"],
                semantic_score=score_data["component_scores"]["semantic_score"],
                skill_match_score=score_data["component_scores"]["skill_match_score"],
                explanation=json.dumps(explanation),
            )
            db.add(log)
        except Exception as e:
            logger.error(f"Error scoring candidate {resume_input.candidate_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Scoring failed for {resume_input.candidate_id}: {str(e)}")

    db.commit()

    # Sort descending by hybrid_score
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)

    ranked = []
    for rank, r in enumerate(results, start=1):
        ranked.append(
            CandidateResult(
                rank=rank,
                candidate_id=r["candidate_id"],
                hybrid_score=r["hybrid_score"],
                hire_probability=r["hire_probability"],
                component_scores=ComponentScores(**r["component_scores"]),
                explanation=ExplanationOut(**r["explanation"]),
            )
        )

    return ScreenResponse(
        job_id=req.job_id,
        total_candidates=len(ranked),
        ranked_candidates=ranked,
    )
