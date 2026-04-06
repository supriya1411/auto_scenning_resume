import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
import joblib
import logging
from datetime import datetime, timezone

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.config import MODELS_DIR, RETRAIN_AUC_THRESHOLD

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["tfidf_sim", "semantic_sim", "skill_match", "years_exp", "edu_level"]


def _build_model(flavor: str):
    if flavor == "xgboost":
        return XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    return LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42)


def train_model(X: np.ndarray, y: np.ndarray, flavor: str = "xgboost"):
    model = _build_model(flavor)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    cv_auc = float(scores.mean())
    model.fit(X, y)
    return model, cv_auc


def retrain_from_feedback(
    feedback_df: pd.DataFrame,
    current_auc: float,
    flavor: str = "xgboost",
    models_dir: Path = MODELS_DIR,
) -> Dict:
    if len(feedback_df) < 20:
        return {"success": False, "reason": "Need ≥ 20 labeled samples to retrain"}

    label_map = {"shortlisted": 1, "rejected": 0}
    df = feedback_df.copy()
    df["label"] = df["decision"].map(label_map)
    df = df.dropna(subset=["label"])

    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        return {"success": False, "reason": f"Missing feature columns: {missing}"}

    X = df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(int)

    if len(np.unique(y)) < 2:
        return {"success": False, "reason": "Need both positive (shortlisted) and negative (rejected) labels"}

    logger.info(f"Retraining {flavor} on {len(X)} samples...")
    new_model, new_auc = train_model(X, y, flavor=flavor)
    logger.info(f"New CV AUC: {new_auc:.4f} | Incumbent AUC: {current_auc:.4f}")

    promoted = new_auc > current_auc and new_auc >= RETRAIN_AUC_THRESHOLD
    if promoted:
        meta_path = models_dir / "model_meta.joblib"
        old_version = 0
        if meta_path.exists():
            old_version = joblib.load(meta_path).get("version", 0)
        new_version = old_version + 1
        joblib.dump(new_model, models_dir / "model.joblib")
        joblib.dump(
            {
                "name": flavor,
                "version": new_version,
                "validation_auc": new_auc,
                "trained_at": datetime.now(timezone.utc).isoformat(),
                "n_samples": len(X),
                "flavor": flavor,
            },
            meta_path,
        )
        logger.info(f"Model promoted to v{new_version}")

    return {
        "success": True,
        "new_auc": round(new_auc, 4),
        "incumbent_auc": round(current_auc, 4),
        "promoted": promoted,
        "n_samples": len(X),
        "reason": "Promoted" if promoted else "Did not beat incumbent AUC or below threshold",
    }
