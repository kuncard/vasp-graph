"""
Priority-based BFS crawl for VASP Wiki knowledge graph.

Starts from seed pages, follows edges in priority order, runs until the
graph is fully explored. All 1056 nodes reachable from Category_VASP.
"""

from __future__ import annotations
import json, argparse, os, logging
from collections import defaultdict
from wiki_parser import parse_html, classify_page, extract_param_facts, deduplicate_nodes, classify_by_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("crawl_errors.log"), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

EDGE_PRIORITY = {
    "contains":         1,
    "has_subcategory":  2,
    "belongs_to":       3,
    "wikilink_param":   4,
    "wikilink_tutorial":5,
    "wikilink_domain":  6,
    "wikilink_other":   7,
}

def _edge_priority(source_subtype: str, target_name: str, relation: str) -> int:
    if relation in EDGE_PRIORITY:
        return EDGE_PRIORITY[relation]
    name_type = classify_by_name(target_name)
    if name_type == "parameter":
        return EDGE_PRIORITY["wikilink_param"]
    if name_type == "domain":
        return EDGE_PRIORITY["wikilink_domain"]
    if name_type == "tutorial":
        return EDGE_PRIORITY["wikilink_tutorial"]
    return EDGE_PRIORITY["wikilink_other"]

def _add(edges, edge_set, src, tgt, rel):
    key = (src, tgt, rel)
    if key not in edge_set:
        edge_set.add(key)
        edges.append({"source": src, "target": tgt, "relation": rel})

def _auto_seeds(wiki_dir, probe_seeds, top_n=5):
    """Parse a few top pages to find structural root nodes."""
    probe_nodes, probe_edges = {}, []
    for ps in probe_seeds:
        p = parse_html(os.path.join(wiki_dir, ps + ".html"))
        if p is None: continue
        et, st = classify_page(ps, p["categories"], p["body_text"])
        probe_nodes[ps] = {"id": ps, "title": p["title"], "entry_type": et, "subtype": st}
        for m in p.get("member_pages", []): probe_edges.append({"source": ps, "target": m, "relation": "contains"})
        for s in p.get("subcategories", []): probe_edges.append({"source": ps, "target": s, "relation": "has_subcategory"})
    contains = defaultdict(int)
    belongs = defaultdict(int)
    ids = set(probe_nodes)
    for e in probe_edges:
        if e["source"] in ids and e["target"] in ids:
            if e["relation"] == "contains": contains[e["source"]] += 1
            else: belongs[e["source"]] += 1
    scored = [(contains.get(n,0) - belongs.get(n,0)*2, n) for n in ids if contains.get(n,0) > 0]
    scored.sort(reverse=True)
    return [s[1] for s in scored[:top_n]] or ["Category_VASP", "INCAR", "POSCAR", "KPOINTS", "POTCAR"]


def crawl(wiki_dir, seeds, max_depth=0, verbose=True):
    """Priority BFS. max_depth=0 means run until exhausted."""
    visited, nodes_map = set(), {}
    edges_list, edge_set = [], set()
    frontier, depth = set(seeds), 0

    while frontier and (max_depth == 0 or depth <= max_depth):
        next_all = []
        expanded = 0
        if verbose: print(f"\nDepth {depth}: {len(frontier)} pages", flush=True)

        for page_name in sorted(frontier):
            if page_name.startswith(("Special_", "Talk_", "User_", "Template_",
                                      "MediaWiki_", "File_", "Construction_")):
                visited.add(page_name); continue
            if page_name in visited: continue
            visited.add(page_name)

            filepath = os.path.join(wiki_dir, page_name + ".html")
            try:
                parsed = parse_html(filepath)
            except Exception as e:
                log.error("Parse failed: %s — %s", page_name, e)
                continue
            if parsed is None:
                log.warning("File not found: %s", filepath)
                continue
            body = parsed["body_text"]
            if body.lower().startswith("there is currently no text"):
                continue

            et, st = classify_page(page_name, parsed["categories"], body)
            nodes_map[page_name] = {
                "id": page_name, "title": parsed["title"],
                "entry_type": et, "subtype": st, "content": body,
                "raw_html": parsed.get("raw_html", ""),
                "tags": parsed["categories"][:8], "aliases": [],
                "category": parsed["categories"][0] if parsed["categories"] else ""}
            expanded += 1

            for member in parsed.get("member_pages", []):
                _add(edges_list, edge_set, page_name, member, "contains")
                if member not in visited: next_all.append((EDGE_PRIORITY["contains"], member))
            for sub in parsed.get("subcategories", []):
                _add(edges_list, edge_set, page_name, sub, "has_subcategory")
                if sub not in visited: next_all.append((EDGE_PRIORITY["has_subcategory"], sub))
            for cat in parsed.get("categories", []):
                cp = f"Category_{cat.replace(' ', '_')}"
                _add(edges_list, edge_set, page_name, cp, "belongs_to")
                if cp not in visited: next_all.append((EDGE_PRIORITY["belongs_to"], cp))
            for target in parsed.get("internal_links", []):
                if not target or target == page_name: continue
                _add(edges_list, edge_set, page_name, target, "wikilink")
                if target not in visited:
                    next_all.append((_edge_priority(st, target, "wikilink"), target))

        next_all.sort(key=lambda x: (x[0], x[1]))
        seen_n = set()
        next_frontier = set()
        for prio, tgt in next_all:
            if tgt not in seen_n and tgt not in visited:
                seen_n.add(tgt); next_frontier.add(tgt)

        if verbose: print(f"  → {len(next_frontier)} pages for next round (expanded {expanded})", flush=True)
        frontier = next_frontier
        depth += 1

    if verbose:
        st = defaultdict(int)
        for n in nodes_map.values(): st[n["subtype"]] += 1
        print(f"\nCrawled: {len(nodes_map)} nodes, {len(edges_list)} edges")
        for k, v in sorted(st.items()): print(f"  {k}: {v}")

    return list(nodes_map.values()), edges_list


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Priority BFS VASP Wiki crawler")
    p.add_argument("wiki_dir")
    p.add_argument("--seeds", "-s", default="")
    p.add_argument("--auto-seeds", type=int, default=5)
    p.add_argument("--depth", "-d", type=int, default=0, help="0=unlimited")
    p.add_argument("--output", "-o", default="data/crawled")
    p.add_argument("--no-dedup", action="store_true")
    p.add_argument("--no-enrich", action="store_true")
    args = p.parse_args()

    if args.seeds:
        seeds = [s.strip() for s in args.seeds.split(",")]
    else:
        print(f"Auto-detecting {args.auto_seeds} center nodes...")
        seeds = _auto_seeds(args.wiki_dir, ["Category_VASP", "Category_Examples", "Category_INCAR"], args.auto_seeds)
        print(f"  → {seeds}")

    print(f"Seeds: {seeds}\nDepth: {'unlimited' if args.depth==0 else args.depth}\n")

    nodes, edges = crawl(args.wiki_dir, seeds, args.depth)

    if not args.no_dedup:
        nodes, edges = deduplicate_nodes(nodes, edges)
        print(f"After dedup: {len(nodes)} nodes, {len(edges)} edges")
    if not args.no_enrich:
        for n in nodes:
            if n.get("subtype") != "parameter":
                continue
            facts = extract_param_facts(n.get("content", ""))
            if facts:
                n.setdefault("structured", {})
                n["structured"]["quick_facts"] = {
                    "tag": facts["tag"], "type": facts["type"],
                    "default": facts["default"], "raw_description": facts.get("raw_description", ""),
                }
                if facts.get("definition"):
                    n["structured"]["definition"] = facts["definition"]
                if facts.get("options"):
                    n["structured"]["options"] = facts["options"]
                if facts.get("warnings"):
                    n["structured"]["warnings"] = facts["warnings"]
        print(f"Parameters enriched: {sum(1 for n in nodes if n.get('structured',{}).get('quick_facts'))}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output + "_nodes.json", "w", encoding="utf-8") as f: json.dump(nodes, f, ensure_ascii=False, indent=2)
    with open(args.output + "_edges.json", "w", encoding="utf-8") as f: json.dump(edges, f, ensure_ascii=False, indent=2)
    print(f"\nExported → {args.output}_nodes.json + {args.output}_edges.json")
