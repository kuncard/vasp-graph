"""
Search quality evaluation harness.

Runs parameter sweeps over the EnhancedSearcher and reports Precision@K
against hand-annotated "golden" results.

Usage:
  python eval_search.py                # run all configs, show best
  python eval_search.py --verbose      # show per-query breakdown
"""

import sys, json, itertools, math
from search_enhanced import EnhancedSearcher, BM25Engine, compute_pagerank, build_neighbor_graph
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════
# Golden test set: {query: [expected_top_titles]}
# "Expected" = what a VASP expert would expect to see first.
# ═══════════════════════════════════════════════════════════════════

GOLDEN = {
    "magnetic": [
        "Magnetism",
        "Noncollinear magnetism",
        "MAGMOM",
        "LNONCOLLINEAR",
        "Spin-orbit coupling",
    ],
    "encut": [
        "ENCUT",
        "ENCUTFOCK",
        "ENCUTGW",
        "ENCUTGWSOFT",
        "PREC",
    ],
    "isif": [
        "ISIF",
        "IBRION",
        "NSW",
        "Forces",
        "Stress",
    ],
    "phonon": [
        "Phonons",
        "Electron-phonon interactions",
        "PHON_NWRITE",
        "ELPH_IGNORE_IMAG_PHONONS",
    ],
    "spin orbit coupling": [
        "Spin-orbit coupling",
        "LSORBIT",
        "LNONCOLLINEAR",
        "Magnetism",
    ],
    "magnetic calculation setup": [
        "Magnetism",
        "Calculation setup",
        "LNONCOLLINEAR",
        "MAGMOM",
    ],
    "energy cutoff": [
        "ENCUT",
        "ENCUTFOCK",
        "ENCUTGW",
        "PREC",
    ],
    "relaxation": [
        "Ionic minimization",
        "IBRION",
        "ISIF",
        "NSW",
        "EDIFFG",
    ],
    "kpoint": [
        "KPOINTS",
        "KSPACING",
        "Crystal momentum",
        "KPOINTS_OPT",
    ],
    "hybrid functional": [
        "Hybrid functionals",
        "HFALPHA",
        "HFRCUT",
        "HFSCREEN",
    ],
}


def precision_at_k(results: list[dict], expected: list[str], k: int = 5) -> float:
    """What fraction of top-K results are in the expected set?"""
    top = {r["title"] for r in results[:k]}
    hits = sum(1 for e in expected[:k] if e in top)
    return hits / k


class ConfigurableSearcher:
    """EnhancedSearcher with tunable parameters exposed."""

    def __init__(self, **kwargs):
        self.pg_weight = kwargs.get("pg_weight", 0.6)
        self.nb_boost = kwargs.get("nb_boost", 0.3)
        self.hotzone = kwargs.get("hotzone", 40)
        self.primary_w = kwargs.get("primary_w", 2.0)
        self.expanded_w = kwargs.get("expanded_w", 1.5)
        self.subtype_b = kwargs.get("subtype_b", 1.2)
        self._bm25 = BM25Engine()
        self._pr = compute_pagerank()
        self._nb = build_neighbor_graph()

    def search(self, query: str, limit: int = 10) -> list[dict]:
        from search_enhanced import expand_query, defaultdict
        import re
        bm = self._bm25

        primary = bm.search(query, limit=200)
        scores: dict[int, float] = defaultdict(float)
        max_p = max(s for _, s in primary) if primary else 1.0
        for idx, s in primary:
            scores[idx] += s / max_p * self.primary_w

        expanded = expand_query(query)
        for term in expanded[1:min(len(expanded), 10)]:
            sec = bm.search(term, limit=100)
            max_s = max(s for _, s in sec) if sec else 1.0
            for idx, s in sec:
                scores[idx] += s / max_s * self.expanded_w

        sorted_bm25 = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_ids = {bm.get_node(idx)["id"] for idx, _ in sorted_bm25[:self.hotzone]}

        nb_boosts: dict[str, float] = {}
        for idx in scores:
            nid = bm.get_node(idx)["id"]
            nb = self._nb.get(nid, set())
            hot = len(nb & top_ids)
            if hot > 0:
                nb_boosts[nid] = 1.0 + self.nb_boost * hot

        top_subtype = bm.get_node(sorted_bm25[0][0]).get("subtype", "")
        _HUBS = {"INCAR", "POTCAR", "KPOINTS", "POSCAR", "OUTCAR"}

        scored = []
        for idx, bm25_s in scores.items():
            node = bm.get_node(idx)
            nid = node["id"]
            title = node.get("title", "")
            st = node.get("subtype", "generic")

            nb_b = nb_boosts.get(nid, 1.0)
            if top_subtype == "parameter" and title in _HUBS:
                nb_b *= 0.5

            pr = self._pr.get(nid, 0.0)
            gb = 1.0 + math.sqrt(pr) * self.pg_weight

            tb = {"domain": 1.3, "parameter": 1.1, "tutorial": 1.0,
                  "best_practice": 1.0, "pitfall": 1.0, "generic": 0.6}.get(st, 1.0)

            sb = 1.0
            if top_subtype == "parameter" and st == "parameter":
                sb = self.subtype_b
            elif top_subtype == "tutorial" and st == "tutorial":
                sb = self.subtype_b

            scored.append((bm25_s * nb_b * gb * tb * sb, node))
        scored.sort(key=lambda x: x[0], reverse=True)

        from search_enhanced import EnhancedSearcher as ES
        es = ES.__new__(ES)  # hack to reuse format
        return es._format_results(scored[:limit])


# ═══════════════════════════════════════════════════════════════════
# Parameter sweep
# ═══════════════════════════════════════════════════════════════════

PARAM_GRID = {
    "pg_weight": [0.3, 0.6, 0.9],
    "nb_boost": [0.1, 0.3, 0.5],
    "hotzone": [20, 40, 60],
    "primary_w": [2.0, 3.0, 4.0],
    "expanded_w": [0.5, 1.0, 1.5],
    "subtype_b": [1.1, 1.2, 1.3],
}


def run_sweep(verbose: bool = False):
    print("=== Parameter Sweep ===")
    print(f"  {len(GOLDEN)} queries, testing combinations...")
    print()

    # Test current defaults first
    default_params = {"pg_weight": 0.6, "nb_boost": 0.3,
                      "hotzone": 40, "primary_w": 2.0, "expanded_w": 1.5,
                      "subtype_b": 1.2}
    baseline = evaluate_config(default_params, verbose)
    print(f"  Baseline (current): P@5 = {baseline:.1%}")
    print()

    # Simplified: test one param at a time, keep others at baseline
    best_params = dict(default_params)
    best_score = baseline

    for param, values in PARAM_GRID.items():
        for val in values:
            if val == default_params.get(param):
                continue
            cfg = dict(default_params)
            cfg[param] = val
            score = evaluate_config(cfg, verbose=False)
            delta = score - baseline
            marker = " ← BEST" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_params = dict(cfg)
            if verbose or score > baseline + 0.01:
                print(f"  {param}={val}: P@5={score:.1%} (Δ{delta:+.0%}){marker}")

    print()
    print(f"=== Best config: P@5 = {best_score:.1%} ===")
    for k, v in best_params.items():
        if v != default_params.get(k):
            print(f"  {k}: {default_params[k]} → {v}")
    print()

    if verbose:
        print("--- Per-query breakdown (best config) ---")
        evaluate_config(best_params, verbose=True)


def evaluate_config(params: dict, verbose: bool = False) -> float:
    searcher = ConfigurableSearcher(**params)
    scores = []
    for query, expected in GOLDEN.items():
        results = searcher.search(query, limit=10)
        p5 = precision_at_k(results, expected, k=5)
        scores.append(p5)
        if verbose:
            top3 = ", ".join(r["title"] for r in results[:5])
            hits = sum(1 for e in expected[:5] if e in {r["title"] for r in results[:5]})
            print(f"  {query:30s} P@5={p5:.0%}  top5: {top3[:80]}...")
    return sum(scores) / len(scores)


if __name__ == "__main__":
    verbose = "--verbose" in sys.argv
    run_sweep(verbose=verbose)
