"""
Train XGBoost & Logistic Regression models on the processed dataset.
Run AFTER download_data.py:
    python scripts/train_model.py
    python scripts/train_model.py --flavor logistic_regression
"""
import sys
import json
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import PROCESSED_DIR, MODELS_DIR, EMBEDDING_MODEL
from src.features import FeatureEngineer
from src.retrainer import train_model, FEATURE_NAMES
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main(flavor: str = "xgboost"):
    pairs_path = PROCESSED_DIR / "training_pairs.csv"
    if not pairs_path.exists():
        raise FileNotFoundError(
            f"training_pairs.csv not found. Run: python scripts/download_data.py"
        )

    df = pd.read_csv(pairs_path)
    logger.info(f"Loaded {len(df)} training pairs. Label balance:\n{df['label'].value_counts().to_string()}")

    # ── Feature engineering ──
    fe = FeatureEngineer(embedding_model_name=EMBEDDING_MODEL)

    corpus = df["resume"].tolist() + df["job_description"].tolist()
    logger.info("Fitting TF-IDF vectorizer...")
    fe.fit_tfidf(corpus)

    required_skills_list = [json.loads(s) for s in df["required_skills"]]

    feat_df = fe.compute_features_batch(
        resume_texts=df["resume"].tolist(),
        job_descriptions=df["job_description"].tolist(),
        required_skills_list=required_skills_list,
    )

    # Save features for inspection
    feat_df["label"] = df["label"].values
    feat_df.to_csv(PROCESSED_DIR / "features.csv", index=False)
    logger.info(f"Features saved to {PROCESSED_DIR / 'features.csv'}")

    X = feat_df[FEATURE_NAMES].values.astype(np.float32)
    y = df["label"].values.astype(int)

    # ── Train & evaluate ──
    logger.info(f"Training {flavor}...")
    model, cv_auc = train_model(X, y, flavor=flavor)
    logger.info(f"Cross-validation AUC: {cv_auc:.4f}")

    # ── Save artifacts ──
    fe.save(MODELS_DIR / "feature_engineer.joblib")
    joblib.dump(model, MODELS_DIR / "model.joblib")
    meta = {
        "name": flavor,
        "version": 1,
        "validation_auc": round(cv_auc, 4),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(X),
        "flavor": flavor,
        "features": FEATURE_NAMES,
    }
    joblib.dump(meta, MODELS_DIR / "model_meta.joblib")

    logger.info(f"Model saved → {MODELS_DIR / 'model.joblib'}")
    logger.info(f"Metadata   → {MODELS_DIR / 'model_meta.joblib'}")
    logger.info(f"Training complete. AUC = {cv_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train hire prediction model")
    parser.add_argument(
        "--flavor",
        choices=["xgboost", "logistic_regression"],
        default="xgboost",
        help="Model flavor to train (default: xgboost)",
    )
    args = parser.parse_args()
    main(flavor=args.flavor)
