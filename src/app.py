from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, AnyUrl
from typing import Optional, List
from src.recommender import Recommender

app = FastAPI(title="SHL Assessment Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rec = None

class RecommendRequest(BaseModel):
    query: Optional[str] = None
    url: Optional[AnyUrl] = None
    k: int = 10

class AssessmentItem(BaseModel):
    assessment_name: str
    url: AnyUrl
    test_type: Optional[str] = None
    assessment_length_min: Optional[int] = None
    languages: Optional[List[str]] = []

class RecommendResponse(BaseModel):
    results: List[AssessmentItem]

@app.on_event("startup")
def load_rec():
    global rec
    rec = Recommender()

@app.get("/health")
def health():
    return {"status":"healthy"}

@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest):
    if not body.query and not body.url:
        raise HTTPException(status_code=400, detail="Provide either 'query' or 'url'")
    k = max(1, min(10, body.k))
    try:
        items = rec.recommend(query=body.query, jd_url=str(body.url) if body.url else None, k=k)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"results": items}
