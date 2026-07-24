"""
HTTP API wrapper for the enhanced VASP search engine.

Usage:
  python search_server.py
  # → http://localhost:8766/search?q=magnetic&limit=10
  # → http://localhost:8766/health

KDG API still runs on :8765, this runs alongside on :8766.
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import sys, os

sys.path.insert(0, os.path.dirname(__file__))
from search_enhanced import EnhancedSearcher

app = FastAPI(title="VASP Graph Enhanced Search", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Init searcher once at startup ──
searcher: EnhancedSearcher | None = None


@app.on_event("startup")
def _startup():
    global searcher
    searcher = EnhancedSearcher()
    print("Enhanced searcher ready.", flush=True)


@app.get("/health")
def health():
    return {"status": "ok", "nodes": searcher.bm25._N if searcher else 0}


@app.get("/search")
def search(q: str = Query(..., description="Search query"),
           limit: int = Query(10, ge=1, le=100),
           verbose: bool = Query(False)):
    """Enhanced search: BM25 + graph boost + type boost."""
    if searcher is None:
        return {"error": "Searcher not initialized yet"}
    results = searcher.search(q, limit=limit, verbose=verbose)
    return {
        "query": q,
        "total": len(results),
        "results": results,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8766, log_level="info")
