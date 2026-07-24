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
    "kpoint": ["k-points", "k-point", "k point", "k-point mesh", "brillouin zone", "IBZKPT", "KSPACING"],
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
                 "is", "are", "be", "how", "what", "why", "with", "can", "do", "does",
                 "method", "methods", "using", "used", "use", "note", "notes",
                 "example", "examples", "result", "results", "value", "values",
                 "case", "cases", "follow", "following", "set", "setting",
                 "page", "pages", "see", "also", "may", "default",
                 "will", "must", "should", "need", "needs",
                 "one", "two", "first", "well", "much", "part", "type",
                 "per", "due", "via", "way", "many", "often", "without", "within",
                 "section", "describe", "described", "shown", "given",
                 "available", "possible", "important", "required"}
    tokens = [t for t in tokens_raw if t not in stopwords and len(t) > 1]

    # Handle VASP compound patterns in raw query (before hyphen-splitting loses them)
    _COMPOUND_MAP = {
        "k-point": "kpoints", "k-points": "kpoints", "k point": "kpoints",
        "k mesh": "kpoints", "k-mesh": "kpoints",
        "spin-orbit": "spin orbit coupling", "spin orbit": "spin orbit coupling",
        "gw": "gw calculations", "dft-d": "vdw", "dft d": "vdw",
    }
    raw_lower = query.lower().replace("_", " ").replace("-", " ")
    for pattern, expansion in _COMPOUND_MAP.items():
        if pattern in raw_lower and expansion not in tokens:
            tokens.append(expansion)

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


def build_neighbor_graph(edges_file: str = "data/test_edges.json") -> dict[str, set[str]]:
    """Build adjacency list: {node_id: {neighbor_ids}} from edge list."""
    with open(edges_file, encoding="utf-8") as f:
        edges_list = json.load(f)
    graph: dict[str, set[str]] = defaultdict(set)
    for e in edges_list:
        s, t = e["source"], e["target"]
        graph[s].add(t)
        graph[t].add(s)
    return dict(graph)


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


def compute_tf(enriched_file: str = "data/test_enriched.json") -> dict[str, dict[str, int]]:
    """Precompute term frequencies for all nodes.

    Returns {entry_id: {stemmed_word: raw_count}} for title+content text.
    Uses the same simple stemmer as import_to_kdg.py for consistency.
    """
    import math

    # Minimal inline stemmer (same logic as import_to_kdg.py)
    suffixes = [
        "izational", "isation", "izations", "tational", "ational",
        "ization", "fulness", "ousness", "iveness", "ability",
        "alities", "alisms", "ements", "ations", "istics",
        "ement", "ments", "ation", "ities", "fully",
        "ingly", "ously", "istic", "izing", "ising",
        "ical", "able", "ible", "ness", "ment",
        "ship", "tion", "sion", "ally", "ated",
        "ized", "ised", "ting", "ring", "ling",
        "ding", "sing", "ives", "isms",
        "ion", "est", "ity", "ism", "ize",
        "ers", "ies", "ing", "als", "ves",
        "ed", "es", "ly", "al", "ic",
        "er", "or", "s",
    ]
    def stem(w):
        w = w.lower()
        for sfx in suffixes:
            if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                return w[:-len(sfx)]
        return w

    with open(enriched_file, encoding="utf-8") as f:
        nodes = json.load(f)

    tf: dict[str, dict[str, int]] = {}
    for n in nodes:
        nid = n["id"]
        title = (n.get("title") or "").lower()
        content = (n.get("content") or "").lower()
        text = title + " " + content
        words = re.findall(r"[a-z]{3,}", text)

        counts: dict[str, int] = defaultdict(int)
        for w in words:
            s = stem(w)
            counts[s] += 1

        tf[nid] = dict(counts)

    return tf


# ═══════════════════════════════════════════════════════════════════
# BM25 engine — replaces KDG keyword search entirely
# ═══════════════════════════════════════════════════════════════════

class BM25Engine:
    """Local BM25 search over the enriched JSON, no KDG dependency for scoring."""

    def __init__(self, enriched_file: str = "data/test_enriched.json"):
        import math

        with open(enriched_file, encoding="utf-8") as f:
            self.nodes = json.load(f)

        self._node_map = {n["id"]: n for n in self.nodes}

        # Simple stemmer (same as import_to_kdg.py)
        suffixes = [
            "izational", "isation", "izations", "tational", "ational",
            "ization", "fulness", "ousness", "iveness", "ability",
            "alities", "alisms", "ements", "ations", "istics",
            "ement", "ments", "ation", "ities", "fully",
            "ingly", "ously", "istic", "izing", "ising",
            "ical", "able", "ible", "ness", "ment",
            "ship", "tion", "sion", "ally", "ated",
            "ized", "ised", "ting", "ring", "ling",
            "ding", "sing", "ives", "isms",
            "ion", "est", "ity", "ism", "ize",
            "ers", "ies", "ing", "als", "ves",
            "ed", "es", "ly", "al", "ic",
            "er", "or", "s",
        ]
        def stem(w):
            w = w.lower()
            for sfx in suffixes:
                if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                    return w[:-len(sfx)]
            return w

        # ── VASP-specific stop words ──
        # These appear everywhere but carry zero topic signal.
        # Filtered during both indexing and querying.
        self._stop_words = {
            # KDG originals
            "a", "an", "the", "of", "in", "on", "at", "to", "for",
            "and", "or", "is", "are", "be", "was", "were", "been",
            # High-frequency VASP functional words
            "method", "methods", "using", "used", "use", "note",
            "example", "examples", "result", "results", "value",
            "case", "cases", "follow", "following", "set", "setting",
            "page", "pages", "see", "also", "may", "default",
            "can", "will", "must", "should", "need", "needs",
            "two", "one", "first", "second", "well", "much",
            "part", "type", "form", "per", "due", "via",
            "way", "many", "often", "without", "within",
            "section", "describe", "described", "shown", "given",
            "available", "possible", "important", "required",
        }

        # ── Build inverted index with positions ──
        # {stemmed_word: {doc_idx: [pos1, pos2, ...]}}
        self._inverted: dict[str, dict[int, list[int]]] = defaultdict(
            lambda: defaultdict(list))
        self._doc_lengths: list[int] = []  # length in tokens per doc
        self._avgdl: float = 0.0
        self._N = len(self.nodes)

        for idx, n in enumerate(self.nodes):
            title = (n.get("title") or "").lower()
            content = (n.get("content") or "").lower()
            # Title words count 3× for BM25 scoring
            text = (title + " ") * 3 + content
            words = [w for w in re.findall(r"[a-z0-9]{2,}", text)
                     if w not in self._stop_words]

            self._doc_lengths.append(len(words))
            for pos, w in enumerate(words):
                self._inverted[stem(w)][idx].append(pos)

        self._avgdl = sum(self._doc_lengths) / max(1, self._N)

        # ── Precompute IDF (BM25 variant) ──
        self._idf: dict[str, float] = {}
        for word, doc_dict in self._inverted.items():
            df = len(doc_dict)
            self._idf[word] = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)

    # ── Proximity scoring ──
    def _proximity_boost(self, doc_idx: int, query_stems: list[str],
                         window: int = 30) -> float:
        """Boost documents where multiple query terms appear close together.

        Returns 1.0 + bonus where bonus increases with more co-located query terms.
        """
        if len(query_stems) < 2:
            return 1.0

        # Gather all positions for all query stems in this doc
        all_positions: list[int] = []
        for s in query_stems:
            positions = self._inverted.get(s, {}).get(doc_idx, [])
            all_positions.extend(positions)

        if len(all_positions) < 2:
            return 1.0

        all_positions.sort()

        # Count how many query stems have at least one occurrence within `window`
        # of another query stem's occurrence
        co_located_stems: set[str] = set()
        for i, s1 in enumerate(query_stems):
            for j, s2 in enumerate(query_stems):
                if i >= j:
                    continue
                pos1_list = self._inverted.get(s1, {}).get(doc_idx, [])
                pos2_list = self._inverted.get(s2, {}).get(doc_idx, [])

                # Check if any pair of positions is within window
                for p1 in pos1_list:
                    for p2 in pos2_list:
                        if abs(p1 - p2) <= window:
                            co_located_stems.add(s1)
                            co_located_stems.add(s2)
                            break
                    if s1 in co_located_stems:
                        break

        # Boost: 1.0 + 0.2 per extra co-located term pair beyond the first
        pairs = len(co_located_stems)
        if pairs >= 2:
            return 1.0 + 0.3 * (pairs - 1)
        return 1.0

    # ── BM25 scoring ──
    def search(self, query: str, limit: int = 50, k1: float = 1.5, b: float = 0.75
               ) -> list[tuple[int, float]]:
        """Return [(node_idx, bm25_score), ...] sorted descending."""
        import math

        # Tokenize: alphanumeric, min 2 chars, filter stop words
        tokens = [w for w in re.findall(r"[a-z0-9]{2,}", query.lower())
                  if w not in self._stop_words]
        # Allow known 1-char VASP tokens
        for ch in re.findall(r"[a-z]", query.lower()):
            if ch in {"k", "g", "f", "q", "x", "y", "z"}:
                tokens.append(ch)
        # Build stem lookup inline
        suffixes = [
            "izational", "isation", "izations", "tational", "ational",
            "ization", "fulness", "ousness", "iveness", "ability",
            "alities", "alisms", "ements", "ations", "istics",
            "ement", "ments", "ation", "ities", "fully",
            "ingly", "ously", "istic", "izing", "ising",
            "ical", "able", "ible", "ness", "ment",
            "ship", "tion", "sion", "ally", "ated",
            "ized", "ised", "ting", "ring", "ling",
            "ding", "sing", "ives", "isms",
            "ion", "est", "ity", "ism", "ize",
            "ers", "ies", "ing", "als", "ves",
            "ed", "es", "ly", "al", "ic",
            "er", "or", "s",
        ]
        def stem(w):
            w = w.lower()
            for sfx in suffixes:
                if w.endswith(sfx) and len(w) - len(sfx) >= 3:
                    return w[:-len(sfx)]
            return w

        stems = list({stem(t) for t in tokens})

        # Score each candidate document
        doc_scores: dict[int, float] = defaultdict(float)
        for s in stems:
            idf = self._idf.get(s, 0.0)
            if idf == 0.0:
                continue
            for doc_idx, positions in self._inverted.get(s, {}).items():
                tf = len(positions)  # term frequency = number of positions
                doc_len = self._doc_lengths[doc_idx]
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self._avgdl)
                doc_scores[doc_idx] += idf * numerator / denominator

        # Apply proximity boost for multi-term queries
        if len(stems) >= 2:
            for doc_idx in list(doc_scores.keys()):
                boost = self._proximity_boost(doc_idx, stems)
                doc_scores[doc_idx] *= boost

        # Sort
        scored = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def get_node(self, idx: int) -> dict:
        return self.nodes[idx]

    def get_node_by_id(self, nid: str) -> dict | None:
        return self._node_map.get(nid)


# ═══════════════════════════════════════════════════════════════════
# D. Entropy-based functional word detection (unused, kept for reference)
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
    def __init__(self,
                 edges_file: str = "data/test_edges.json",
                 enriched_file: str = "data/test_enriched.json"):
        self.bm25 = BM25Engine(enriched_file)
        self.pagerank = compute_pagerank(edges_file)
        self.degree = compute_degree_centrality(edges_file)
        self.neighbors = build_neighbor_graph(edges_file)
        print(f"  BM25 index: {len(self.bm25._inverted)} terms, {self.bm25._N} docs", file=sys.stderr)
        print(f"  PageRank: {len(self.pagerank)} nodes", file=sys.stderr)
        print(f"  Neighbor graph: {len(self.neighbors)} nodes", file=sys.stderr)

    def search(self, query: str, limit: int = 10, verbose: bool = False) -> list[dict]:
        """Enhanced search: BM25 scoring → graph boost → type boost.

        No KDG dependency for ranking. Query expansion is used as a second-pass
        boost: expanded terms contribute additional BM25 score at half weight.
        """

        q = query.strip()
        if not q:
            return self._fallback_top(limit, "Empty query — showing top pages by importance")

        # 1. Primary BM25 search (original query)
        primary: list[tuple[int, float]] = self.bm25.search(query, limit=200)
        if not primary:
            return self._fallback_top(limit, f"Query '{q}' had only stop words — showing top pages")
        scores: dict[int, float] = defaultdict(float)

        # ── Adaptive weights based on query characteristics ──
        # Specific query (high avg IDF, exact param name) → trust original more
        # Broad query (low avg IDF, single word) → lean on synonyms
        query_tokens = [w for w in re.findall(r"[a-z0-9]{2,}", q.lower())
                        if w not in self.bm25._stop_words]
        token_idfs = [self.bm25._idf.get(t, 1.0) for t in query_tokens]
        avg_idf = sum(token_idfs) / max(1, len(token_idfs))
        is_exact_param = any(
            t.upper() in self.bm25._node_map and
            self.bm25._node_map[t.upper()].get("subtype") == "parameter"
            for t in query_tokens
        )

        if is_exact_param or avg_idf > 4.0:
            # Highly specific: "MAGMOM", "ENCUTGWSOFT"
            primary_w, expanded_w = 4.0, 0.5
        elif len(query_tokens) == 1 and avg_idf < 2.5:
            # Broad single word: "relaxation", "method", "energy"
            primary_w, expanded_w = 1.5, 2.0
        else:
            # Balanced: multi-word or medium specificity
            primary_w, expanded_w = 2.0, 1.5

        if verbose:
            print(f"  Query: {query} (avg_idf={avg_idf:.1f})", file=sys.stderr)
            print(f"  Weights: primary={primary_w} expanded={expanded_w}", file=sys.stderr)

        # Normalize BM25 scores
        max_primary = max(s for _, s in primary) if primary else 1.0
        for idx, s in primary:
            scores[idx] += s / max_primary * primary_w

        # 2. Expanded query boost
        expanded = expand_query(query)
        if verbose:
            print(f"  Expanded: {expanded[:8]}...", file=sys.stderr)

        for term in expanded[1:min(len(expanded), 10)]:
            secondary = self.bm25.search(term, limit=100)
            max_sec = max(s for _, s in secondary) if secondary else 1.0
            for idx, s in secondary:
                scores[idx] += s / max_sec * expanded_w

        # 3. Neighbor vote: nodes near top BM25 hits get a boost
        #    Being in a "hot zone" of the graph signals relevance
        top_nids: set[str] = set()
        sorted_by_bm25 = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for idx, _ in sorted_by_bm25[:40]:  # top 40 form the "hot zone"
            top_nids.add(self.bm25.get_node(idx)["id"])

        neighbor_boosts: dict[str, float] = {}
        for idx in scores:
            nid = self.bm25.get_node(idx)["id"]
            nb = self.neighbors.get(nid, set())
            hot_neighbors = len(nb & top_nids)
            if hot_neighbors > 0:
                neighbor_boosts[nid] = 1.0 + 0.3 * hot_neighbors
        if verbose:
            boosted = sum(1 for v in neighbor_boosts.values() if v > 1.0)
            print(f"  Neighbor boost applied to {boosted} nodes", file=sys.stderr)

        # 4. Graph + type boost
        scored: list[tuple[float, dict]] = []
        for idx, bm25_score in scores.items():
            node = self.bm25.get_node(idx)
            nid = node["id"]
            title = node.get("title", "")

            pr = self.pagerank.get(nid, 0.0)
            deg = self.degree.get(nid, 0.0)
            # Graph boost: only significant when BM25 score is already meaningful
            # Uses sqrt to dampen extreme PageRank differences
            import math as _m
            graph_boost = 1.0 + _m.sqrt(pr) * 0.5 + _m.sqrt(deg) * 0.1

            st = node.get("subtype", "generic")
            type_boost = {"domain": 1.3, "parameter": 1.1, "tutorial": 1.0,
                          "best_practice": 1.0, "pitfall": 1.0, "generic": 0.6}.get(st, 1.0)

            nb_boost = neighbor_boosts.get(nid, 1.0)

            # Hub penalty: VASP core files (INCAR, POTCAR, KPOINTS, POSCAR, OUTCAR)
            # are linked to almost everything — penalize them when top result is a parameter
            _HUB_TITLES = {"INCAR", "POTCAR", "KPOINTS", "POSCAR", "OUTCAR"}
            top_subtype = self.bm25.get_node(sorted_by_bm25[0][0]).get("subtype", "")
            if top_subtype == "parameter" and title in _HUB_TITLES:
                nb_boost *= 0.5  # halve the neighbor boost for hub pages

            # Subtype preference: boost same-subtype results when top is dominant
            subtype_boost = 1.0
            if top_subtype == "parameter" and st == "parameter":
                subtype_boost = 1.2  # prefer parameters when top is a parameter
            elif top_subtype == "tutorial" and st == "tutorial":
                subtype_boost = 1.2  # prefer tutorials when top is a tutorial
            if subtype_boost != 1.0 and verbose:
                pass  # tracked below

            final_score = bm25_score * nb_boost * graph_boost * type_boost * subtype_boost
            scored.append((final_score, node))

        scored.sort(key=lambda x: x[0], reverse=True)

        if verbose:
            print(f"  Candidates scored: {len(scored)}", file=sys.stderr)
            print(f"  Top {limit}:", file=sys.stderr)

        # Return as KDG-compatible dicts (title, entry_type, id at minimum)
        return self._format_results(scored[:limit])

    def _fallback_top(self, limit: int, hint: str = "") -> list[dict]:
        """Fallback: return top pages by PageRank when query is empty/all-stopwords."""
        import sys
        if hint:
            print(f"  Note: {hint}", file=sys.stderr)
        scored = []
        for n in self.bm25.nodes:
            nid = n["id"]
            pr = self.pagerank.get(nid, 0.0)
            deg = self.degree.get(nid, 0.0)
            score = pr * 2.0 + deg * 1.0
            scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        return self._format_results(scored[:limit])

    def _format_results(self, scored: list[tuple[float, dict]]) -> list[dict]:
        """Convert internal nodes to KDG-compatible dict format."""
        out = []
        for _, node in scored:
            s = node.get("structured", {}) or {}
            qf = s.get("quick_facts", {}) or {}
            out.append({
                "id": node["id"],
                "title": node.get("title", node["id"]),
                "entry_type": node.get("entry_type", "capability"),
                "content": node.get("content", ""),
                "tags": node.get("tags", []),
                "structured": {
                    "definition": s.get("definition", ""),
                    "quick_facts": qf,
                    "options": s.get("options", []),
                    "warnings": s.get("warnings", []),
                },
                "subtype": node.get("subtype", "generic"),
            })
        return out


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(description="Enhanced VASP knowledge graph search")
    p.add_argument("query", help="Search query")
    p.add_argument("--limit", "-n", type=int, default=10)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--edges", default="data/test_edges.json")
    args = p.parse_args()

    searcher = EnhancedSearcher(edges_file=args.edges)
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
