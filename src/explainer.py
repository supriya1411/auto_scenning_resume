import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["tfidf_sim", "semantic_sim", "skill_match", "years_exp", "edu_level"]

FEATURE_TEMPLATES = {
    "tfidf_sim": {
        "high": "Strong keyword overlap with job description ({value:.0%} TF-IDF match)",
        "med":  "Moderate keyword overlap with job description ({value:.0%} TF-IDF match)",
        "low":  "Weak keyword overlap with job description ({value:.0%} TF-IDF match)",
    },
    "semantic_sim": {
        "high": "Resume is semantically well-aligned with the job description (score: {value:.2f})",
        "med":  "Moderate semantic alignment with the job description (score: {value:.2f})",
        "low":  "Low semantic alignment with the job description (score: {value:.2f})",
    },
    "skill_match": {
        "high": "Excellent skill match — {value:.0%} of required skills found in resume",
        "med":  "Partial skill match — {value:.0%} of required skills found in resume",
        "low":  "Poor skill match — only {value:.0%} of required skills found in resume",
    },
    "years_exp": {
        "high": "Extensive relevant experience ({raw:.1f}+ years)",
        "med":  "Adequate experience ({raw:.1f} years)",
        "low":  "Limited experience ({raw:.1f} years) — minimum recommended is 3",
    },
    "edu_level": {
        "high": "Holds an advanced degree (Master's / PhD) relevant to the role",
        "med":  "Holds a Bachelor's degree relevant to the role",
        "low":  "Educational qualifications not clearly specified",
    },
}


def _level(value: float) -> str:
    if value >= 0.70:
        return "high"
    if value >= 0.40:
        return "med"
    return "low"


def _render(feature: str, value: float) -> str:
    level = _level(value)
    tmpl = FEATURE_TEMPLATES[feature][level]
    raw = value * 20 if feature == "years_exp" else value * 3
    return tmpl.format(value=value, raw=raw)


class Explainer:
    def __init__(self, model, model_name: str = "xgboost"):
        self.model = model
        self.model_name = model_name
        self._shap_explainer = None

    def _get_shap_explainer(self):
        if self._shap_explainer is None:
            try:
                import shap
                self._shap_explainer = shap.TreeExplainer(self.model)
            except Exception as e:
                logger.warning(f"SHAP init failed: {e}")
        return self._shap_explainer

    def _importances(self, features: Dict[str, float]) -> Optional[List[float]]:
        X = np.array([[features[f] for f in FEATURE_NAMES]])
        flavor = self.model_name.lower()
        try:
            if "xgb" in flavor or "xgboost" in flavor:
                exp = self._get_shap_explainer()
                if exp is None:
                    return None
                sv = exp.shap_values(X)
                arr = sv[1][0] if isinstance(sv, list) else sv[0]
                return arr.tolist()
            else:
                coefs = self.model.coef_[0]
                vals = np.array([features[f] for f in FEATURE_NAMES])
                return (coefs * vals).tolist()
        except Exception as e:
            logger.warning(f"Importance computation failed: {e}")
            return None

    def explain(
        self,
        features: Dict[str, float],
        hire_probability: float,
        required_skills: List[str],
        resume_text: str,
    ) -> Dict:
        importances = self._importances(features)

        if importances:
            pairs = sorted(zip(FEATURE_NAMES, importances), key=lambda x: abs(x[1]), reverse=True)
            pos_feats = [(f, i) for f, i in pairs if i > 0]
            neg_feats = [(f, i) for f, i in pairs if i <= 0]
        else:
            sorted_feats = sorted(features.items(), key=lambda x: x[1], reverse=True)
            pos_feats = [(f, v) for f, v in sorted_feats if v >= 0.4]
            neg_feats = [(f, v) for f, v in sorted_feats if v < 0.4]

        top_positive = [_render(f, features[f]) for f, _ in pos_feats[:3] if features[f] >= 0.35]
        top_negative = [_render(f, features[f]) for f, _ in neg_feats[:2] if features[f] < 0.55]

        # Add missing skills to negatives
        if required_skills and features.get("skill_match", 0) < 0.7:
            rl = resume_text.lower()
            missing = [s for s in required_skills if s.lower() not in rl][:3]
            if missing:
                top_negative.append(f"Missing required skills: {', '.join(missing)}")

        # Guarantee minimums
        while len(top_positive) < 2:
            top_positive.append("Candidate profile partially aligns with role requirements")
        if not top_negative:
            top_negative = ["No critical gaps detected"]

        return {
            "top_positive_factors": top_positive[:3],
            "top_negative_factors": top_negative[:3],
            "model_contribution": {
                "tfidf_score": round(features.get("tfidf_sim", 0), 4),
                "semantic_score": round(features.get("semantic_sim", 0), 4),
                "skill_match_score": round(features.get("skill_match", 0), 4),
                "hire_probability": round(hire_probability, 4),
            },
        }
