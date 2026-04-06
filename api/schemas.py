from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ── Request schemas ──────────────────────────────────────────────
class ResumeInput(BaseModel):
    candidate_id: str
    text: str = Field(..., min_length=50, description="Resume plain text (min 50 chars)")


class ScoreWeights(BaseModel):
    tfidf: float = Field(0.20, ge=0, le=1)
    semantic: float = Field(0.30, ge=0, le=1)
    skill_match: float = Field(0.25, ge=0, le=1)
    hire_prob: float = Field(0.25, ge=0, le=1)


class ScreenRequest(BaseModel):
    job_id: str
    job_description: str = Field(..., min_length=20)
    required_skills: List[str] = Field(default_factory=list)
    resumes: List[ResumeInput] = Field(..., min_length=1)
    model_flavor: str = Field("xgboost", pattern="^(xgboost|logistic_regression)$")
    score_weights: Optional[ScoreWeights] = None


class FeedbackRequest(BaseModel):
    job_id: str
    candidate_id: str
    recruiter_id: str
    decision: str = Field(..., pattern="^(shortlisted|rejected|on_hold)$")


class RetrainRequest(BaseModel):
    model_flavor: str = Field("xgboost", pattern="^(xgboost|logistic_regression)$")
    force: bool = False


# ── Response schemas ─────────────────────────────────────────────
class ComponentScores(BaseModel):
    tfidf_score: float
    semantic_score: float
    skill_match_score: float


class ExplanationOut(BaseModel):
    top_positive_factors: List[str]
    top_negative_factors: List[str]
    model_contribution: Dict[str, float]


class CandidateResult(BaseModel):
    rank: int
    candidate_id: str
    hybrid_score: float
    hire_probability: float
    component_scores: ComponentScores
    explanation: ExplanationOut


class ScreenResponse(BaseModel):
    job_id: str
    total_candidates: int
    ranked_candidates: List[CandidateResult]


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    samples_since_retrain: int


class ModelInfoResponse(BaseModel):
    active_model: str
    version: int
    validation_auc: float
    trained_at: str
    n_samples: int
    samples_since_retrain: int


class RetrainResponse(BaseModel):
    success: bool
    promoted: bool
    new_auc: Optional[float] = None
    incumbent_auc: Optional[float] = None
    n_samples: Optional[int] = None
    reason: str
