import logging
import sys
from pathlib import Path
import random
import json

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional
import io

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

try:
    import docx
    logger.info("DOCX extractor (python-docx) identified successfully.")
except ImportError:
    docx = None
    logger.warning("DOCX extractor (python-docx) not found. Uploading .docx files will fail.")

try:
    from pdfminer.high_level import extract_text as extract_pdf_text
    logger.info("PDF extractor (pdfminer.six) identified successfully.")
except ImportError:
    extract_pdf_text = None
    logger.warning("PDF extractor (pdfminer.six) not found. Uploading .pdf files will fail.")

# Setup mock memory states
GLOBAL_SETTINGS = {
    "threshold": 0.70,
    "mode": "balanced"
}

DASHBOARD_DATA = {
    "total_count": 0,
    "candidates": []
}

class SettingsUpdate(BaseModel):
    threshold: float
    mode: str
class ResumeInput(BaseModel):
    candidate_id: str
    text: str

class ScreenRequest(BaseModel):
    job_id: str
    job_description: str
    required_skills: List[str]
    resumes: List[ResumeInput]

class FeedbackRequest(BaseModel):
    job_id: str
    candidate_id: str
    recruiter_id: str
    decision: str

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

app = FastAPI(title="MOCKED AI Resume Auto-Screening API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all for local dev, including Port 3000
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("==================================================")
    logger.info("AI Resume Screening API is now ONLINE")
    logger.info("Backend: http://localhost:8000")
    logger.info("Frontend (if served by API): http://localhost:8000")
    logger.info("Frontend (if served separately): http://localhost:3000")
    logger.info("==================================================")


@app.post("/api/v1/screen")
def screen_candidates(req: ScreenRequest):
    results = []
    
    for idx, r in enumerate(req.resumes):
        hybrid_score = random.uniform(0.4, 0.95)
        hire_prob = hybrid_score + random.uniform(-0.1, 0.1)
        tfidf = random.uniform(0.3, 0.9)
        semantic = random.uniform(0.5, 0.9)
        skills = random.uniform(0.2, 0.9)
        
        pos_factors = [
            f"Strong keyword overlap with job description ({round(tfidf*100)}% TF-IDF match)",
            f"Resume is semantically well-aligned with the job description (score: {round(semantic, 2)})",
            f"Skill match — {round(skills*100)}% of required skills found"
        ]
        
        neg_factors = []
        if skills < 0.6:
            neg_factors.append("Missing critical skills requested in the job description")
        if hire_prob < 0.5:
            neg_factors.append("Overall profile indicates limited relevant experience")
            
        if not neg_factors:
            neg_factors = ["No critical gaps detected"]

        results.append({
            "candidate_id": r.candidate_id,
            "hybrid_score": round(hybrid_score, 4),
            "hire_probability": round(max(0.1, min(1.0, hire_prob)), 4),
            "component_scores": {
                "tfidf_score": round(tfidf, 4),
                "semantic_score": round(semantic, 4),
                "skill_match_score": round(skills, 4),
            },
            "explanation": {
                "top_positive_factors": pos_factors[:random.randint(2,3)],
                "top_negative_factors": neg_factors,
                "model_contribution": {"tfidf_score": tfidf, "semantic_score": semantic, "skill_match_score": skills, "hire_probability": hire_prob}
            }
        })
        
    results.sort(key=lambda x: x["hybrid_score"], reverse=True)
    
    for rank, r in enumerate(results, 1):
        r["rank"] = rank

    # Persist in memory for Dashboard
    DASHBOARD_DATA["total_count"] += len(results)
    
    # Store new candidates at top of list
    DASHBOARD_DATA["candidates"] = results + DASHBOARD_DATA["candidates"]

    return {
        "job_id": req.job_id,
        "total_candidates": len(results),
        "ranked_candidates": results
    }

@app.get("/api/v1/dashboard")
def get_dashboard():
    all_cands = DASHBOARD_DATA["candidates"]
    top_5 = sorted(all_cands, key=lambda x: x["hybrid_score"], reverse=True)[:5]
    
    return {
        "total_count": DASHBOARD_DATA["total_count"],
        "top_candidates": top_5,
        "all_candidates": all_cands
    }

@app.get("/api/v1/settings")
def get_settings():
    return GLOBAL_SETTINGS

@app.post("/api/v1/settings")
def save_settings(req: SettingsUpdate):
    GLOBAL_SETTINGS["threshold"] = req.threshold
    GLOBAL_SETTINGS["mode"] = req.mode
    return {"success": True, "settings": GLOBAL_SETTINGS}

@app.get("/api/v1/model/info")
def model_info():
    return {
        "active_model": "xgboost_mocked_v1",
        "version": 1,
        "validation_auc": 0.84,
        "trained_at": "Mocked",
        "n_samples": 450,
        "samples_since_retrain": random.randint(10, 30)
    }

@app.post("/api/v1/feedback")
def submit_feedback(req: FeedbackRequest):
    return {
        "success": True,
        "message": f"Feedback recorded. Decision: {req.decision}",
        "samples_since_retrain": random.randint(30, 50)
    }

@app.post("/api/v1/retrain")
def trigger_retrain():
    return {
        "success": True,
        "promoted": True,
        "new_auc": 0.86,
        "incumbent_auc": 0.84,
        "n_samples": 500,
        "reason": "Promoted"
    }

@app.post("/api/v1/extract-text")
async def extract_text(file: UploadFile = File(...)):
    filename = file.filename.lower()
    content = await file.read()
    
    text = ""
    try:
        if filename.endswith(".pdf"):
            if extract_pdf_text is None:
                raise HTTPException(status_code=500, detail="PDF extractor (pdfminer.six) is not installed in the server environment.")
            with io.BytesIO(content) as f:
                text = extract_pdf_text(f)
        elif filename.endswith(".docx"):
            if docx is None:
                raise HTTPException(status_code=500, detail="DOCX extractor (python-docx) is not installed in the server environment.")
            doc = docx.Document(io.BytesIO(content))
            text = "\n".join([p.text for p in doc.paragraphs])
        elif filename.endswith(".txt"):
            text = content.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload PDF, DOCX, or TXT.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {str(e)}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file.")

    return {"text": text.strip(), "filename": file.filename}

if FRONTEND_DIR.exists():
    @app.get("/", include_in_schema=False)
    def serve_frontend():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    # Mount the frontend directory at the root after all API routes
    # This allows relative paths like "style.css" and "app.js" to work correctly
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
