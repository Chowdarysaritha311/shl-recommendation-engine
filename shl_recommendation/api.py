"""
SHL Assessment Recommendation API
FastAPI backend with:
  GET  /health     — health check
  POST /recommend  — returns 5-10 assessments in JSON

Run locally:
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Deploy on Render:
  - Set GEMINI_API_KEY in environment variables
  - Start command: uvicorn api:app --host 0.0.0.0 --port $PORT
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import os
import time

from recommender import get_recommender

app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="RAG-based recommendation engine for SHL assessments.",
    version="1.0.0",
)

# Allow all origins (needed for Streamlit frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Schemas ─────────────────────────────────────────────────────────────────

class RecommendRequest(BaseModel):
    query: str

    class Config:
        json_schema_extra = {
            "example": {
                "query": "I am hiring for Java developers who can also collaborate effectively with my business teams."
            }
        }


class AssessmentResult(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int]
    remote_support: str
    test_type: List[str]


class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResult]


# ─── Startup ─────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-load the recommender so first request isn't slow."""
    print("Loading recommender on startup...")
    try:
        get_recommender()
        print("Recommender loaded successfully.")
    except Exception as e:
        print(f"Warning: Recommender failed to load on startup: {e}")


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend_assessments(request: RecommendRequest):
    """
    Given a natural language query or job description text,
    returns 5–10 most relevant SHL Individual Test assessments.
    """
    if not request.query or len(request.query.strip()) < 5:
        raise HTTPException(status_code=400, detail="Query is too short.")

    try:
        rec = get_recommender()
        results = rec.recommend(request.query, top_k=10)

        # Ensure min 1, max 10
        results = results[:10]
        if len(results) == 0:
            raise HTTPException(status_code=404, detail="No assessments found for this query.")

        return RecommendResponse(recommended_assessments=results)

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Assessment data not loaded: {str(e)}. Run scraper.py first."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/")
def root():
    return {
        "message": "SHL Assessment Recommendation API",
        "endpoints": {
            "health": "GET /health",
            "recommend": "POST /recommend — body: {\"query\": \"your query here\"}",
            "docs": "GET /docs"
        }
    }
