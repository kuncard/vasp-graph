"""
VASP Wiki HTML → Knowledge Graph Parser.

Parses HTTrack-mirrored VASP Wiki pages into structured nodes and edges
that can be imported into know-do-graph or any graph database.

Usage:
    python parse_wiki.py <wiki_dir> --output data/
"""

from __future__ import annotations

import json, re, os, argparse
from pathlib import Path
from collections import defaultdict

from wiki_parser import parse_html, clean_wiki_text, classify_page, deduplicate_nodes


# ── Main extraction ─────────────────────────────────────────────────


def extract_graph(wiki_dir: str) -> tuple[list[dict], list[dict]]:
    """Walk wiki_dir, parse all HTML files, return (nodes, edges) as dicts."""
    nodes: dict[str, dict] = {}
    edges: list[dict] = []

    html_files = list(Path(wiki_dir).rglob("*.html"))
    print(f"Found {len(html_files)} HTML files")

    for filepath in html_files:
        parsed = parse_html(str(filepath))
        if parsed is None:
            continue
        page_name = parsed["page_name"]

        # Skip non-content pages
        if page_name.startswith(("Special_", "Talk_", "User_", "Template_", "MediaWiki_", "File_", "Construction_")):
            continue

        # Skip truly empty pages only
        body = parsed["body_text"]
        if body.lower().startswith("there is currently no text in this page"):
            continue
        # Empty / stub category pages
        if "currently contains no pages or media" in body.lower() and len(body) < 80:
            continue

        # Classify
        entry_type, subtype = classify_page(page_name, parsed["categories"], parsed["body_text"])

        # ── Node ──
        tags = [t.strip() for t in parsed["categories"]]
        nodes[page_name] = {
            "id": page_name,
            "title": parsed["title"],
            "entry_type": entry_type,
            "subtype": subtype,
            "content": parsed["body_text"] if parsed["body_text"] else "",
            "raw_html": parsed.get("raw_html", ""),
            "tags": tags,
            "aliases": [],
            "category": tags[0] if tags else "",
        }

        # ── Edges ──
        for target in parsed["internal_links"]:
            if target and target != page_name:
                edges.append({"source": page_name, "target": target, "relation": "wikilink"})
        for cat_raw in parsed["categories"]:
            cat_page = f"Category_{cat_raw.replace(' ', '_')}"
            if cat_page != page_name:
                edges.append({"source": page_name, "target": cat_page, "relation": "belongs_to"})
        for sub in parsed["subcategories"]:
            if sub != page_name:
                edges.append({"source": page_name, "target": sub, "relation": "has_subcategory"})
        for member in parsed["member_pages"]:
            if member != page_name:
                edges.append({"source": page_name, "target": member, "relation": "contains"})

    # Deduplicate edges then nodes (shared function in wiki_parser)
    seen_edges = set()
    edges_deduped = []
    for e in edges:
        key = (e["source"], e["target"], e["relation"])
        if key not in seen_edges:
            seen_edges.add(key)
            edges_deduped.append(e)

    nodes_list, edges_final = deduplicate_nodes(list(nodes.values()), edges_deduped)
    print(f"Extracted {len(nodes_list)} nodes ({len(nodes) - len(nodes_list)} dupes removed) and {len(edges_final)} edges")
    return nodes_list, edges_final


# ── Export ──────────────────────────────────────────────────────────


def export_json(nodes: list[dict], edges: list[dict], output_dir: str) -> None:
    """Write nodes.json and edges.json to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    nodes_path = os.path.join(output_dir, "nodes.json")
    edges_path = os.path.join(output_dir, "edges.json")

    with open(nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(edges_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, ensure_ascii=False, indent=2)

    print(f"Exported {len(nodes)} nodes → {nodes_path}")
    print(f"Exported {len(edges)} edges → {edges_path}")


def print_stats(nodes: list[dict], edges: list[dict]) -> None:
    """Print summary statistics."""
    type_counts: dict[str, int] = {}
    subtype_counts: dict[str, int] = {}
    for n in nodes:
        type_counts[n["entry_type"]] = type_counts.get(n["entry_type"], 0) + 1
        subtype_counts[n["subtype"]] = subtype_counts.get(n["subtype"], 0) + 1

    relation_counts: dict[str, int] = {}
    for e in edges:
        relation_counts[e["relation"]] = relation_counts.get(e["relation"], 0) + 1

    print("\n=== Node types ===")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")
    print("\n=== Subtypes ===")
    for s, c in sorted(subtype_counts.items(), key=lambda x: -x[1]):
        print(f"  {s}: {c}")
    print("\n=== Edge relations ===")
    for r, c in sorted(relation_counts.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


# ── CLI ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse VASP Wiki HTML into graph data")
    parser.add_argument("wiki_dir", help="Path to VASP Wiki mirror (e.g. vasp/www.vasp.at/wiki/index.php/)")
    parser.add_argument("--output", "-o", default="data", help="Output directory (default: data/)")
    parser.add_argument("--stats", action="store_true", help="Print summary statistics")
    args = parser.parse_args()

    nodes, edges = extract_graph(args.wiki_dir)
    export_json(nodes, edges, args.output)

    if args.stats:
        print_stats(nodes, edges)
