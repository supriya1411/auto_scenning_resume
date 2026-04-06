"""
Download and preprocess the Kaggle Resume Dataset.
Dataset: gauravduttakiit/resume-dataset
Run: python scripts/download_data.py
"""
import sys
import json
import random
import logging
import pandas as pd
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import RAW_DIR, PROCESSED_DIR, KAGGLE_DATASET, JOB_DESCRIPTIONS, CATEGORY_SKILLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

random.seed(42)


def download_dataset() -> Path:
    csv_path = RAW_DIR / "UpdatedResumeDataSet.csv"
    if csv_path.exists():
        logger.info(f"Dataset already present: {csv_path}")
        return csv_path

    logger.info(f"Downloading: {KAGGLE_DATASET}")
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)
        logger.info("Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed: {e}\n"
            "Ensure ~/.kaggle/kaggle.json exists with your API credentials.\n"
            "Get it from https://www.kaggle.com/settings/account"
        )

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV found after download.")
    return sorted(csv_files)[0]


def build_training_pairs(df: pd.DataFrame, neg_ratio: int = 2) -> pd.DataFrame:
    categories = sorted(df["Category"].unique())
    rows = []
    for _, row in df.iterrows():
        cat = row["Category"]
        resume = str(row["Resume"])
        # positive pair
        rows.append({
            "resume": resume,
            "job_description": JOB_DESCRIPTIONS[cat],
            "required_skills": json.dumps(CATEGORY_SKILLS.get(cat, [])),
            "category": cat,
            "label": 1,
        })
        # negative pairs
        others = [c for c in categories if c != cat]
        for neg_cat in random.sample(others, min(neg_ratio, len(others))):
            rows.append({
                "resume": resume,
                "job_description": JOB_DESCRIPTIONS[neg_cat],
                "required_skills": json.dumps(CATEGORY_SKILLS.get(neg_cat, [])),
                "category": neg_cat,
                "label": 0,
            })
    return pd.DataFrame(rows)


def main():
    csv_path = download_dataset()

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df = df[["Category", "Resume"]].dropna()
    df["Category"] = df["Category"].str.strip()
    df["Resume"] = df["Resume"].str.strip()

    known = set(JOB_DESCRIPTIONS.keys())
    df = df[df["Category"].isin(known)].reset_index(drop=True)
    logger.info(f"Loaded {len(df)} resumes across {df['Category'].nunique()} categories")

    pairs = build_training_pairs(df, neg_ratio=2)
    out_path = PROCESSED_DIR / "training_pairs.csv"
    pairs.to_csv(out_path, index=False)
    logger.info(f"Saved {len(pairs)} training pairs → {out_path}")
    logger.info(f"Label distribution:\n{pairs['label'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
