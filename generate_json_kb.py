"""
Generate a single JSON knowledge base file for agent consumption.

Agent loads one file, searches in memory, follows edges directly —
no file-hopping, no wikilink parsing needed.

Usage:
    python generate_json_kb.py data/enriched.json data/edges.json -o kb.json
"""

from __future__ import annotations

import json, argparse, os
from collections import defaultdict


def generate(nodes_path: str, edges_path: str, output_path: str) -> None:
    with open(nodes_path, encoding="utf-8") as f:
        nodes_list = json.load(f)
    with open(edges_path, encoding="utf-8") as f:
        edges_list = json.load(f)

    # Link contexts
    ctx_path = nodes_path.replace(".json", "_link_contexts.json")
    link_contexts: dict[str, str] = {}
    if os.path.exists(ctx_path):
        with open(ctx_path, encoding="utf-8") as f:
            link_contexts = json.load(f)

    # Build node lookup
    nodes: dict[str, dict] = {}
    for n in nodes_list:
        nid = n["id"]
        nodes[nid] = {
            "id": nid,
            "title": n.get("title", nid),
            "type": n.get("entry_type", "unknown"),
            "subtype": n.get("subtype", "generic"),
            "category": n.get("category", ""),
            "tags": n.get("tags", [])[:8],
            "content": n.get("content", ""),
            "links": [],
            "referenced_by": [],
        }
        qf = n.get("structured", {}).get("quick_facts")
        if qf:
            nodes[nid]["quick_facts"] = {
                "type": qf.get("type", "?"),
                "default": qf.get("default", ""),
                "description": qf.get("raw_description", ""),
            }
        for field in ["definition", "options", "warnings"]:
            val = n.get("structured", {}).get(field)
            if val:
                nodes[nid][field] = val
        ts = n.get("structured", {}).get("tutorial_summary")
        if ts:
            nodes[nid]["tutorial_summary"] = ts

    # Build edges (both directions in one pass)
    for e in edges_list:
        src, tgt, rel = e["source"], e["target"], e["relation"]
        if src not in nodes or tgt not in nodes:
            continue
        ctx = link_contexts.get(tgt, "")
        if ctx and len(ctx) > 100:
            ctx = ctx[:100].rsplit(" ", 1)[0]

        nodes[src]["links"].append({
            "target": tgt,
            "title": nodes[tgt].get("title", tgt),
            "relation": rel,
            "context": ctx,
        })
        nodes[tgt]["referenced_by"].append({
            "source": src,
            "title": nodes[src].get("title", src),
            "relation": rel,
        })

    # Build index
    by_subtype: dict[str, list[str]] = defaultdict(list)
    by_category: dict[str, list[str]] = defaultdict(list)
    degree: dict[str, int] = {}

    for nid, n in nodes.items():
        by_subtype[n["subtype"]].append(nid)
        cat = n["category"] or "Uncategorized"
        by_category[cat].append(nid)
        degree[nid] = len(n["links"]) + len(n["referenced_by"])

    hubs = sorted(degree.items(), key=lambda x: -x[1])[:50]

    # Subtype cross-reference
    subtype_order = ["domain", "tutorial", "parameter", "best_practice", "pitfall", "generic"]
    cross_ref: dict[str, dict[str, int]] = {}
    for src_st in subtype_order:
        cross_ref[src_st] = {}
        for tgt_st in subtype_order:
            cnt = 0
            for nid in nodes:
                if nodes[nid]["subtype"] != src_st:
                    continue
                for link in nodes[nid]["links"]:
                    tgt_n = nodes.get(link["target"])
                    if tgt_n and tgt_n["subtype"] == tgt_st:
                        cnt += 1
            cross_ref[src_st][tgt_st] = cnt

    kb = {
        "meta": {
            "total_nodes": len(nodes),
            "total_edges": len(edges_list),
            "subtype_counts": {st: len(ids) for st, ids in by_subtype.items()},
        },
        "index": {
            "by_subtype": {st: sorted(ids) for st, ids in by_subtype.items()},
            "by_category": {cat: sorted(ids) for cat, ids in sorted(by_category.items())},
            "hubs": [{"id": nid, "degree": d} for nid, d in hubs],
            "cross_reference": cross_ref,
        },
        "pages": nodes,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(kb, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"Generated {output_path} ({size_kb:.0f} KB)")
    print(f"  {len(nodes)} nodes, {len(edges_list)} edges")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate single-file JSON knowledge base")
    p.add_argument("nodes", help="Path to enriched nodes.json")
    p.add_argument("edges", help="Path to edges.json")
    p.add_argument("--output", "-o", default="kb.json")
    args = p.parse_args()

    generate(args.nodes, args.edges, args.output)
