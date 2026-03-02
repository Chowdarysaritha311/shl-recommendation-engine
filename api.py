import os
import sys
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="SHL Assessment Recommendation API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class RecommendRequest(BaseModel):
    query: str

recommender = None

@app.on_event("startup")
async def startup_event():
    global recommender
    print("Loading recommender on startup...")
    try:
        if not os.path.exists("data/shl_assessments.csv"):
            print("CSV not found, running scraper...")
            os.makedirs("data", exist_ok=True)
            import subprocess
            subprocess.run([sys.executable, "scripts/scraper.py"], check=True)
        from recommender import get_recommender
        recommender = get_recommender()
        print("Recommender loaded successfully.")
    except Exception as e:
        print(f"Warning: Recommender failed to load on startup: {e}")

@app.get("/")
def root():
    return {"message": "SHL Assessment Recommendation API", "endpoints": {"health": "GET /health", "recommend": "POST /recommend", "docs": "GET /docs"}}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/recommend")
def recommend(request: RecommendRequest):
    global recommender
    if recommender is None:
        try:
            if not os.path.exists("data/shl_assessments.csv"):
                import subprocess
                subprocess.run([sys.executable, "scripts/scraper.py"], check=True)
            from recommender import get_recommender
            recommender = get_recommender()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Assessment data not loaded: {e}")
    try:
        results = recommender.recommend(request.query, top_k=10)
        return {"query": request.query, "recommended_assessments": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")
