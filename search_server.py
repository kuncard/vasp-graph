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
from fastapi.responses import HTMLResponse
import sys, os, argparse

sys.path.insert(0, os.path.dirname(__file__))
from search_enhanced import EnhancedSearcher, BM25Engine, load_nodes_from_kdg, load_edges_from_kdg
from search_enhanced import compute_pagerank, build_neighbor_graph

app = FastAPI(title="VASP Graph Enhanced Search", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Init searcher once at startup ──
searcher: EnhancedSearcher | None = None


@app.on_event("startup")
def _startup():
    global searcher
    db = getattr(app.state, "kdg_db", None)
    if db:
        import tempfile, os as _os, json as _json
        nodes = load_nodes_from_kdg(db)
        edges = load_edges_from_kdg(db)
        tmp_nodes = _os.path.join(tempfile.gettempdir(), "vasp_kdg_nodes.json")
        tmp_edges = _os.path.join(tempfile.gettempdir(), "vasp_kdg_edges.json")
        _json.dump(nodes, open(tmp_nodes, "w", encoding="utf-8"))
        _json.dump(edges, open(tmp_edges, "w", encoding="utf-8"))
        searcher = EnhancedSearcher(enriched_file=tmp_nodes, edges_file=tmp_edges)
        print(f"  Loaded {len(nodes)} nodes + {len(edges)} edges from KDG db: {db}", flush=True)
    else:
        searcher = EnhancedSearcher()
    print("Enhanced searcher ready.", flush=True)


@app.get("/", response_class=HTMLResponse)
def ui():
    """Serve the search UI."""
    ui_path = os.path.join(os.path.dirname(__file__), "search_ui.html")
    if os.path.exists(ui_path):
        return open(ui_path, encoding="utf-8").read()
    return "<h1>Search UI not found</h1>"


@app.get("/health")
def health():
    return {"status": "ok", "nodes": searcher.bm25._N if searcher else 0}


@app.get("/search")
def search(q: str = Query(..., description="Search query"),
           limit: int = Query(10, ge=1, le=100),
           subtype: str | None = Query(None, description="Filter: parameter|tutorial|domain|best_practice|pitfall|generic"),
           verbose: bool = Query(False)):
    """Enhanced search: BM25 + graph boost + type boost."""
    if searcher is None:
        return {"error": "Searcher not initialized yet"}
    from io import StringIO
    import sys as _sys
    buf = StringIO()
    old = _sys.stderr
    _sys.stderr = buf
    try:
        results = searcher.search(q, limit=limit, verbose=verbose, subtype=subtype)
    finally:
        _sys.stderr = old
    # Extract spell correction hints from stderr
    captured = buf.getvalue()
    spell_hints = [l.replace("Spell check: ", "").strip()
                   for l in captured.split("\n") if "Spell check:" in l]
    return {
        "query": q,
        "spell_corrections": spell_hints if spell_hints else None,
        "subtype_filter": subtype,
        "total": len(results),
        "results": results,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="VASP Graph Enhanced Search Server")
    p.add_argument("--db", default=None, help="KDG SQLite database path (optional)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()

    if args.db:
        app.state.kdg_db = args.db

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


