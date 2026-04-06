import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import logging

logger = logging.getLogger(__name__)

FEATURE_NAMES = ["tfidf_sim", "semantic_sim", "skill_match", "years_exp", "edu_level"]


def extract_years_experience(text: str) -> float:
    patterns = [
        r"(\d+)\+?\s*years?\s+(?:of\s+)?experience",
        r"experience\s+of\s+(\d+)\+?\s*years?",
        r"(\d+)\s*years?\s+work(?:ing)?\s+experience",
        r"(\d+)\s*yrs?\s",
    ]
    years = []
    for pat in patterns:
        for m in re.findall(pat, text.lower()):
            try:
                y = int(m)
                if 0 < y < 50:
                    years.append(y)
            except ValueError:
                pass
    return float(min(years)) if years else 2.0


def extract_education_level(text: str) -> int:
    t = text.lower()
    if any(k in t for k in ["ph.d", "phd", "doctorate", "doctor of"]):
        return 3
    if any(k in t for k in ["master's", "masters", "m.tech", "m.sc", "mba", "m.e.", "mca"]):
        return 2
    if any(k in t for k in ["bachelor's", "b.tech", "b.e.", "bca", "b.sc", "btech", "undergraduate"]):
        return 1
    return 0


def compute_skill_match(resume_text: str, required_skills: List[str]) -> float:
    if not required_skills:
        return 0.0
    resume_lower = resume_text.lower()
    matched = sum(1 for skill in required_skills if skill.lower() in resume_lower)
    return matched / len(required_skills)


class FeatureEngineer:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        self.embedding_model_name = embedding_model_name
        self._encoder = None
        self.is_fitted = False

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformer: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def fit_tfidf(self, corpus: List[str]) -> None:
        self.tfidf.fit(corpus)
        self.is_fitted = True

    def tfidf_similarity(self, text1: str, text2: str) -> float:
        if not self.is_fitted:
            raise RuntimeError("Call fit_tfidf() first.")
        vecs = self.tfidf.transform([text1, text2])
        sim = cosine_similarity(vecs[0], vecs[1])[0][0]
        return float(np.clip(sim, 0.0, 1.0))

    def semantic_similarity(self, text1: str, text2: str) -> float:
        enc = self._get_encoder()
        embs = enc.encode([text1[:512], text2[:512]], normalize_embeddings=True)
        return float(np.clip(np.dot(embs[0], embs[1]), 0.0, 1.0))

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        enc = self._get_encoder()
        return enc.encode([t[:512] for t in texts], normalize_embeddings=True, show_progress_bar=False)

    def compute_features(self, resume_text: str, job_description: str, required_skills: List[str]) -> Dict[str, float]:
        return {
            "tfidf_sim": self.tfidf_similarity(resume_text[:5000], job_description[:2000]),
            "semantic_sim": self.semantic_similarity(resume_text, job_description),
            "skill_match": compute_skill_match(resume_text, required_skills),
            "years_exp": min(extract_years_experience(resume_text) / 20.0, 1.0),
            "edu_level": extract_education_level(resume_text) / 3.0,
        }

    def compute_features_batch(
        self,
        resume_texts: List[str],
        job_descriptions: List[str],
        required_skills_list: List[List[str]],
    ) -> pd.DataFrame:
        n = len(resume_texts)
        logger.info(f"Computing features for {n} pairs...")

        all_texts = resume_texts + job_descriptions
        tfidf_vecs = self.tfidf.transform(all_texts)
        tfidf_sims = [
            float(np.clip(cosine_similarity(tfidf_vecs[i], tfidf_vecs[n + i])[0][0], 0, 1))
            for i in range(n)
        ]

        enc = self._get_encoder()
        logger.info("Encoding resumes for semantic similarity...")
        res_embs = enc.encode([t[:512] for t in resume_texts], normalize_embeddings=True, show_progress_bar=True, batch_size=32)
        logger.info("Encoding job descriptions...")
        jd_embs = enc.encode([t[:512] for t in job_descriptions], normalize_embeddings=True, show_progress_bar=True, batch_size=32)
        semantic_sims = [float(np.clip(np.dot(res_embs[i], jd_embs[i]), 0, 1)) for i in range(n)]

        skill_matches = [compute_skill_match(resume_texts[i], required_skills_list[i]) for i in range(n)]
        years_exps = [min(extract_years_experience(t) / 20.0, 1.0) for t in resume_texts]
        edu_levels = [extract_education_level(t) / 3.0 for t in resume_texts]

        return pd.DataFrame({
            "tfidf_sim": tfidf_sims,
            "semantic_sim": semantic_sims,
            "skill_match": skill_matches,
            "years_exp": years_exps,
            "edu_level": edu_levels,
        })

    def save(self, path: Path) -> None:
        joblib.dump({"tfidf": self.tfidf, "embedding_model_name": self.embedding_model_name, "is_fitted": self.is_fitted}, path)
        logger.info(f"FeatureEngineer saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "FeatureEngineer":
        state = joblib.load(path)
        fe = cls(embedding_model_name=state["embedding_model_name"])
        fe.tfidf = state["tfidf"]
        fe.is_fitted = state["is_fitted"]
        return fe
