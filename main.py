from fastapi import FastAPI
from pydantic import BaseModel
from engine import SemanticEngine
from cache_manager import CacheManager
import uvicorn

app = FastAPI()


engine = SemanticEngine(n_clusters=12)
engine.initialize()
cache_manager = CacheManager(threshold=0.60)

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
async def handle_query(request: QueryRequest):
    query_text = request.query
    query_emb = engine.encoder.encode([query_text])[0]
    
   
    cluster_id, _ = engine.get_cluster_data(query_emb)
    
   
    cache_result = cache_manager.check_cache(query_emb, cluster_id)
    if cache_result:
        return {"query": query_text, "cache_hit": True, **cache_result}

   
    doc_result, _ = engine.vector_store.search(query_emb)
    
   
    cache_manager.update(query_text, query_emb, doc_result[:500], cluster_id)
    
    return {
        "query": query_text,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": None,
        "result": doc_result[:500],
        "dominant_cluster": cluster_id
    }

@app.get("/cache/stats")
async def get_cache_stats():
    return cache_manager.get_stats()

@app.delete("/cache")
async def clear_cache():
    cache_manager.flush()
    return {"message": "Cache flushed"}

if __name__ == "__main__":

    uvicorn.run(app, host="0.0.0.0", port=8000)
