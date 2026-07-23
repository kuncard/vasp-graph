"""
Enhanced search wrapper for VASP knowledge graph.

Combines:
  A. Query expansion (domain synonyms + morphological variants)
  B. Graph-based reranking (PageRank × KDG score)

Usage:
  python search_enhanced.py "magnetic" --limit 10
  python search_enhanced.py "magnetic calculation setup" --limit 10 --verbose
"""

import json, argparse, re, sys, urllib.request, urllib.parse, urllib.error
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# A. Query Expansion — VASP domain synonyms + morphological variants
# ═══════════════════════════════════════════════════════════════════

# Morphological variants: word root → common surface forms
_MORPH_VARIANTS = {
    "magnetic": ["magnetism", "magnetization", "magnet", "magnetics"],
    "magnetism": ["magnetic", "magnetization", "magnet"],
    "magnetization": ["magnetic", "magnetism", "magnet"],
    "calculation": ["calculations", "compute", "computation", "computing"],
    "setup": ["set up", "setting up", "configuration", "configure"],
    "optimization": ["optimize", "optimisation", "relaxation", "minimization"],
    "relaxation": ["relax", "relaxations", "optimization", "minimization"],
    "convergence": ["converge", "converging", "converged", "convergency"],
    "functional": ["functionals", "xc", "exchange-correlation", "GGA", "LDA"],
    "bandstructure": ["band structure", "band-structure", "bands", "dispersion"],
    "dos": ["density of states", "density-of-states", "DOS"],
    "phonon": ["phonons", "vibrational", "vibration", "frequencies"],
    "dielectric": ["dielectrics", "permittivity", "optical"],
    "moleculardynamics": ["molecular dynamics", "MD", "ab initio md", "AIMD"],
}

# Domain concept → search terms (higher-level than morphology)
_CONCEPT_EXPAND = {
    "magnetic": ["spin", "noncollinear", "spin-orbit", "collinear", "ferromagnetic", "antiferromagnetic"],
    "magnetism": ["spin", "noncollinear", "spin-orbit", "collinear"],
    "spin": ["magnetic", "magnetism", "spin-orbit", "noncollinear", "spin spiral"],
    "encut": ["energy cutoff", "plane wave", "ENMAX", "ENMIN", "cutoff energy"],
    "isif": ["stress", "cell optimization", "cell relaxation", "volume relaxation", "IBRION", "PSTRESS"],
    "kpoints": ["k-points", "k point", "k-point mesh", "brillouin zone", "IBZKPT", "KSPACING"],
    "relaxation": ["ISIF", "IBRION", "NSW", "EDIFFG", "POTIM", "optimization"],
    "electronic": ["band", "eigenvalues", "charge density", "potential", "wavefunction"],
    "structure": ["lattice", "cell", "position", "POSCAR", "geometry"],
    "pseudopotential": ["POTCAR", "PAW", "PAW potential", "USPP", "ENMAX", "ENMIN"],
    "vdw": ["van der waals", "dispersion", "IVDW", "DFT-D", "vdW-DF"],
    "hybrid": ["HSE", "PBE0", "HFALPHA", "exact exchange", "screened exchange"],
}


def expand_query(query: str, max_tokens: int = 20) -> list[str]:
    """Expand a query string into a list of related search terms.

    Each term is a short phrase (2-4 words max) that KDG can search for.
    Original query tokens are included first.
    """
    tokens_raw = re.findall(r"[a-zA-Z0-9_-]+", query.lower())
    # Filter stopwords
    stopwords = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or",
                 "is", "are", "be", "how", "what", "why", "with", "can", "do", "does"}
    tokens = [t for t in tokens_raw if t not in stopwords and len(t) > 1]

    expanded = list(tokens)  # original tokens first (higher priority)

    for token in tokens:
        # Morphological variants
        for variant in _MORPH_VARIANTS.get(token, []):
            if variant not in expanded:
                expanded.append(variant)

        # Concept expansions
        for concept in _CONCEPT_EXPAND.get(token, []):
            if concept not in expanded:
                expanded.append(concept)

        # Also try stemmed: remove trailing 's', 'tion', 'ing', etc.
        stemmed = re.sub(r"(s|tion|ing|ed|al|ic|ity|ize)$", "", token)
        if len(stemmed) > 3 and stemmed != token and stemmed not in expanded:
            expanded.append(stemmed)

    # Deduplicate, keep order
    seen = set()
    result = []
    for t in expanded:
        if t not in seen:
            seen.add(t)
            result.append(t)

    return result[:max_tokens]


# ═══════════════════════════════════════════════════════════════════
# B. Graph-based reranking
# ═══════════════════════════════════════════════════════════════════

def compute_pagerank(edges_file: str = "data/test_edges.json") -> dict[str, float]:
    """Compute PageRank for all nodes from edge list.

    Returns {node_id: pagerank_score} normalized to 0-1.
    """
    try:
        import networkx as nx
    except ImportError:
        print("Warning: networkx not installed, skipping PageRank", file=sys.stderr)
        return {}

    with open(edges_file, encoding="utf-8") as f:
        edges_list = json.load(f)

    G = nx.DiGraph()
    for e in edges_list:
        G.add_edge(e["source"], e["target"])

    pr = nx.pagerank(G, alpha=0.85, max_iter=100)
    if not pr:
        return {}

    # Normalize to 0-1
    max_pr = max(pr.values())
    return {k: v / max_pr for k, v in pr.items()}


def compute_degree_centrality(edges_file: str = "data/test_edges.json") -> dict[str, float]:
    """Compute in-degree + out-degree centrality for all nodes.

    Returns {node_id: centrality} normalized to 0-1.
    """
    with open(edges_file, encoding="utf-8") as f:
        edges_list = json.load(f)

    degree = defaultdict(int)
    for e in edges_list:
        degree[e["source"]] += 1
        degree[e["target"]] += 1

    if not degree:
        return {}

    max_d = max(degree.values())
    return {k: v / max_d for k, v in degree.items()}


# ═══════════════════════════════════════════════════════════════════
# C. IDF — Inverse Document Frequency
# ═══════════════════════════════════════════════════════════════════

def compute_idf(enriched_file: str = "data/test_enriched.json") -> dict[str, float]:
    """Compute IDF weights for all words across the knowledge base.

    IDF = log(total_docs / doc_frequency)
    High IDF → rare word → carries more signal (e.g. "magnetic")
    Low IDF  → common word → stopword-like (e.g. "calculation", "setup")
    """
    import math

    with open(enriched_file, encoding="utf-8") as f:
        nodes = json.load(f)

    total = len(nodes)
    doc_freq: dict[str, int] = defaultdict(int)

    for n in nodes:
        title = (n.get("title") or "").lower()
        content = (n.get("content") or "").lower()
        text = title + " " + content
        # Tokenize: split on non-alphanumeric, min 3 chars
        tokens = set(re.findall(r"[a-z]{3,}", text))
        # Also split on underscores (VASP page IDs like "Spin-orbit_coupling")
        id_tokens = set(re.findall(r"[a-z]{3,}", n["id"].lower().replace("_", " ")))
        all_tokens = tokens | id_tokens
        for token in all_tokens:
            doc_freq[token] += 1

    # Compute IDF, clamping at reasonable bounds
    idf: dict[str, float] = {}
    for word, freq in doc_freq.items():
        if freq == 0:
            continue
        idf[word] = math.log(total / freq)
        # Cap: min 0.5 (for very common words), max 6 (for very rare words)
        idf[word] = max(0.5, min(6.0, idf[word]))

    return idf


# ═══════════════════════════════════════════════════════════════════
# D. Entropy-based functional word detection
# ═══════════════════════════════════════════════════════════════════

def compute_entropy_weight(enriched_file: str = "data/test_enriched.json") -> dict[str, float]:
    """Compute word weights based on distribution entropy across subtypes.

    Low entropy → word concentrated in few subtypes → content word → high weight
    High entropy → word spread evenly across subtypes → functional word → low weight

    Returns {word: weight} where 1.0 = neutral, >1.0 = content word, <1.0 = functional.
    """
    import math

    with open(enriched_file, encoding="utf-8") as f:
        nodes = json.load(f)

    # Build: word → {subtype: doc_count}
    word_subtype_docs: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    subtype_doc_total: dict[str, int] = defaultdict(int)

    for n in nodes:
        subtype = n.get("subtype", "generic")
        subtype_doc_total[subtype] += 1

        title = (n.get("title") or "").lower()
        content = (n.get("content") or "").lower()
        text = title + " " + content
        words = set(re.findall(r"[a-z]{3,}", text))

        for w in words:
            word_subtype_docs[w][subtype] += 1

    # Compute entropy for each word
    total_subtypes = len(subtype_doc_total)
    entropy_weight: dict[str, float] = {}
    subtypes_list = sorted(subtype_doc_total.keys())

    for word, subtype_counts in word_subtype_docs.items():
        # Total docs containing this word
        total_docs = sum(subtype_counts.values())
        if total_docs < 3:  # too rare to meaningfully analyze
            continue

        # Calculate entropy: -Σ p(s|w) * log(p(s|w))
        entropy = 0.0
        for st in subtypes_list:
            count = subtype_counts.get(st, 0)
            if count > 0:
                p = count / total_docs
                entropy -= p * math.log(p)

        # Max possible entropy = log(number of subtypes)
        max_entropy = math.log(len(subtypes_list))
        normalized = entropy / max_entropy if max_entropy > 0 else 0

        # Convert to weight: low entropy → high weight
        # entropy 0 → weight 3.0 (only in one subtype)
        # entropy 1 → weight 0.5 (perfectly uniform across subtypes)
        weight = 3.0 - 2.5 * normalized
        entropy_weight[word] = max(0.3, min(3.0, weight))

    return entropy_weight


# ═══════════════════════════════════════════════════════════════════
# KDG API client
# ═══════════════════════════════════════════════════════════════════

class KDGSearcher:
    """Thin wrapper around KDG HTTP API."""

    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url.rstrip("/")
        # Cache: entry_id -> {title, entry_type, id}
        self._entry_cache: dict[str, dict] = {}

    def search(self, query: str, limit: int = 50) -> list[dict]:
        """Search KDG and return list of entry dicts."""
        url = f"{self.base_url}/entries/search?q={urllib.parse.quote(query)}&limit={limit}"
        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                results = json.loads(resp.read())
                for r in results:
                    self._entry_cache[r["id"]] = r
                return results
        except Exception as e:
            print(f"  KDG search error: {e}", file=sys.stderr)
            return []

    def get_entry(self, entry_id: str) -> dict | None:
        if entry_id in self._entry_cache:
            return self._entry_cache[entry_id]
        url = f"{self.base_url}/entries/{entry_id}"
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                entry = json.loads(resp.read())
                self._entry_cache[entry_id] = entry
                return entry
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════
# Enhanced search engine
# ═══════════════════════════════════════════════════════════════════

class EnhancedSearcher:
    def __init__(self, base_url: str = "http://localhost:8765",
                 edges_file: str = "data/test_edges.json",
                 enriched_file: str = "data/test_enriched.json"):
        self.kdg = KDGSearcher(base_url)
        self.pagerank = compute_pagerank(edges_file)
        self.degree = compute_degree_centrality(edges_file)
        self.idf = compute_idf(enriched_file)
        print(f"  PageRank loaded: {len(self.pagerank)} nodes", file=sys.stderr)
        print(f"  Degree centrality loaded: {len(self.degree)} nodes", file=sys.stderr)
        print(f"  IDF vocabulary: {len(self.idf)} words", file=sys.stderr)

    def search(self, query: str, limit: int = 10, verbose: bool = False) -> list[dict]:
        """Enhanced search: expand query → multi-search → graph-rerank."""

        # 1. Expand query
        expanded = expand_query(query)
        if verbose:
            print(f"  Query: {query}", file=sys.stderr)
            print(f"  Expanded: {expanded}", file=sys.stderr)

        # 2. Multi-search: search each expanded term
        all_hits: dict[str, float] = {}  # entry_id -> accumulated score
        seen_details: dict[str, dict] = {}

        # Original query first (highest weight)
        primary_results = self.kdg.search(query, limit=min(limit * 5, 50))
        for rank, entry in enumerate(primary_results):
            score = 1.0 / (1 + rank)  # position-based score
            all_hits[entry["id"]] = score * 2.0  # primary query weight ×2
            seen_details[entry["id"]] = entry

        # Expanded terms (lower weight)
        for term in expanded[1:min(len(expanded), 10)]:
            results = self.kdg.search(term, limit=min(limit * 2, 30))
            for rank, entry in enumerate(results):
                eid = entry["id"]
                score = 0.5 / (1 + rank)  # half weight for expanded terms
                all_hits[eid] = all_hits.get(eid, 0) + score
                if eid not in seen_details:
                    seen_details[eid] = entry

        if verbose:
            print(f"  Raw hits before reranking: {len(all_hits)}", file=sys.stderr)

        # 3. Compute IDF weights for query tokens
        query_tokens = re.findall(r"[a-z]{3,}", query.lower())
        token_idfs = {}
        for tok in query_tokens:
            token_idfs[tok] = self.idf.get(tok, 1.0)  # default 1.0 for unknown words

        if verbose:
            idf_info = ", ".join(f"{t}={token_idfs[t]:.1f}" for t in query_tokens[:8])
            print(f"  Token IDF: {idf_info}", file=sys.stderr)

        # 4. Rerank: KDG score × IDF × graph × type
        scored: list[tuple[float, dict]] = []
        for eid, kdg_score in all_hits.items():
            entry = seen_details.get(eid, {})

            # IDF boost: weight each token by rareness
            title = (entry.get("title") or "").lower()
            idf_boost = 1.0
            matched_high_idf = False
            for tok, idf_val in token_idfs.items():
                if tok in title:
                    idf_boost = max(idf_boost, idf_val)
                    if idf_val > 3.0:
                        matched_high_idf = True

            # High-IDF match → full boost; low-IDF only → penalize
            idf_mult = idf_boost if matched_high_idf else max(0.3, idf_boost / 3.0)

            # Graph boost
            pr = self.pagerank.get(eid, 0.0)
            deg = self.degree.get(eid, 0.0)
            graph_boost = 1.0 + pr * 2.0 + deg * 1.0

            # Type boost
            type_boost = {"capability": 1.3, "procedure": 1.0, "constraint": 1.1,
                          "heuristic": 1.0, "generic": 0.7, "memory": 1.0}.get(
                entry.get("entry_type", "procedure"), 1.0)

            final_score = kdg_score * idf_mult * graph_boost * type_boost
            scored.append((final_score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        if verbose:
            print(f"  After reranking, top {limit}:", file=sys.stderr)

        return [entry for _, entry in scored[:limit]]


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Enhanced VASP knowledge graph search")
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", "-n", type=int, default=10)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--edges", default="data/test_edges.json")
    p.add_argument("--url", default="http://localhost:8765")
    args = p.parse_args()

    searcher = EnhancedSearcher(base_url=args.url, edges_file=args.edges)
    results = searcher.search(args.query, limit=args.limit, verbose=args.verbose)

    print(f"\nResults for: {args.query}")
    print(f"{'─' * 60}")
    for i, entry in enumerate(results):
        title = entry.get("title", entry.get("id", "?"))
        etype = entry.get("entry_type", "?")
        eid = entry["id"][:8]
        print(f"  {i+1:2d}. [{etype:12s}] {title[:55]}")

    print(f"{'─' * 60}")
    print(f"  {len(results)} results")


if __name__ == "__main__":
    main()
