import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import logging

from src.config import MODELS_DIR, DEFAULT_WEIGHTS
from src.features import FeatureEngineer

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["tfidf_sim", "semantic_sim", "skill_match", "years_exp", "edu_level"]


class HybridScorer:
    def __init__(self):
        self.feature_engineer: Optional[FeatureEngineer] = None
        self.model = None
        self.model_name: str = "unknown"
        self.model_version: int = 0
        self._loaded = False

    def load(self, models_dir: Path = MODELS_DIR) -> None:
        fe_path = models_dir / "feature_engineer.joblib"
        model_path = models_dir / "model.joblib"
        if not fe_path.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {models_dir}. Run: python scripts/train_model.py"
            )
        self.feature_engineer = FeatureEngineer.load(fe_path)
        self.model = joblib.load(model_path)
        meta_path = models_dir / "model_meta.joblib"
        if meta_path.exists():
            meta = joblib.load(meta_path)
            self.model_name = meta.get("name", "unknown")
            self.model_version = meta.get("version", 0)
        self._loaded = True
        logger.info(f"HybridScorer loaded: {self.model_name} v{self.model_version}")

    def _ensure_loaded(self):
        if not self._loaded:
            self.load()

    def score(
        self,
        resume_text: str,
        job_description: str,
        required_skills: List[str],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict:
        self._ensure_loaded()
        w = weights or DEFAULT_WEIGHTS
        features = self.feature_engineer.compute_features(resume_text, job_description, required_skills)
        X = np.array([[features[f] for f in FEATURE_NAMES]])
        hire_prob = float(self.model.predict_proba(X)[0][1])
        hybrid_score = (
            w["tfidf"] * features["tfidf_sim"]
            + w["semantic"] * features["semantic_sim"]
            + w["skill_match"] * features["skill_match"]
            + w["hire_prob"] * hire_prob
        )
        return {
            "hire_probability": round(hire_prob, 4),
            "hybrid_score": round(float(hybrid_score), 4),
            "component_scores": {
                "tfidf_score": round(features["tfidf_sim"], 4),
                "semantic_score": round(features["semantic_sim"], 4),
                "skill_match_score": round(features["skill_match"], 4),
            },
            "_features": features,
        }
