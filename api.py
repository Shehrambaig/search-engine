from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from main import HybridIRSystem

app = FastAPI(title="Hybrid Search Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

system = None

class SearchRequest(BaseModel):
    query: str
    use_reranking: Optional[bool] = True
    use_cache: Optional[bool] = True

class SearchResponse(BaseModel):
    query: str
    results: list
    num_results: int
    query_time: float
    from_cache: bool

@app.on_event("startup")
async def startup():
    global system
    print("Initializing search engine...")
    system = HybridIRSystem()

    data_path = os.getenv('DATA_PATH', 'data/raw/Articles.csv')
    index_dir = os.getenv('INDEX_DIR', 'data/processed')

    if os.path.exists(f'{index_dir}/bm25_index.pkl'):
        print(f"Loading index from {index_dir}")
        system.load_index(index_dir)
    else:
        print(f"Building new index from {data_path}")
        system.load_data(data_path)
        system.build_index()
        system.save_index(index_dir)

    print("Search engine ready!")

@app.get("/")
async def root():
    return {"message": "Hybrid Search Engine API", "status": "running"}

@app.get("/health")
async def health():
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")
    return {"status": "healthy", "system": "ready"}

@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        results = system.search(
            request.query,
            use_cache=request.use_cache,
            use_reranking=request.use_reranking
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    if system is None:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        stats = system.get_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
