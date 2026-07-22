"""
Unified VASP Knowledge Graph pipeline.

Combines: crawl/parse → reclassify → enrich → markdown → graph.

Usage:
    # Crawl-based (fast, focused)
    python pipeline.py vasp/vasp/www.vasp.at/wiki/index.php/ --mode crawl

    # Full dump
    python pipeline.py vasp/vasp/www.vasp.at/wiki/index.php/ --mode parse

    # Skip expensive LLM steps
    python pipeline.py ... --no-reclassify --no-tutorials

    # Generate specific outputs
    python pipeline.py ... --outputs markdown,graph
"""

import argparse, os, sys, time, json, subprocess
from collections import defaultdict


def run(cmd: str, step_name: str) -> bool:
    """Run a command, return True on success."""
    print(f"\n{'='*50}")
    print(f"  {step_name}")
    print(f"{'='*50}")
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if result.stderr:
            print(f"  stderr:\n{result.stderr.strip()}")
        if result.stdout:
            print(f"  stdout:\n{result.stdout.strip()}")
        return False
    if result.stdout:
        print(result.stdout.strip())
    return True


def main():
    p = argparse.ArgumentParser(description="VASP Knowledge Graph Pipeline")
    p.add_argument("wiki_dir")
    p.add_argument("--mode", choices=["crawl", "parse"], default="crawl",
                   help="crawl = priority BFS (fast, clean); parse = full dump (complete)")
    p.add_argument("--depth", type=int, default=0, help="0=unlimited")
    p.add_argument("--auto-seeds", type=int, default=5)
    p.add_argument("--seeds", default="")
    p.add_argument("--output-dir", "-o", default="output")
    p.add_argument("--no-reclassify", action="store_true")
    p.add_argument("--no-tutorials", action="store_true")
    p.add_argument("--no-graph", action="store_true")
    p.add_argument("--graph-max", type=int, default=0, help="Graph node limit (0=all)")
    p.add_argument("--api-key", default="", help="OpenAI API key")
    p.add_argument("--api-base", default="https://api.deepseek.com")
    p.add_argument("--model", default="deepseek-chat")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # API key setup
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
        os.environ["OPENAI_API_BASE"] = args.api_base
        os.environ["RECLASSIFY_MODEL"] = args.model

    # ── Step 1: Extract nodes+edges ──
    if args.mode == "crawl":
        seeds_flag = f'--seeds "{args.seeds}"' if args.seeds else f"--auto-seeds {args.auto_seeds}"
        cmd1 = f'python crawl_graph.py "{args.wiki_dir}" {seeds_flag} --depth {args.depth} --output {args.output_dir}/nodes_raw --no-enrich'
    else:
        cmd1 = f'python parse_wiki.py "{args.wiki_dir}" --output {args.output_dir}/'

    if not run(cmd1, "Step 1: Extract nodes + edges"):
        return

    # Determine node/edge files
    if args.mode == "crawl":
        nodes_in = f"{args.output_dir}/nodes_raw_nodes.json"
        edges_in = f"{args.output_dir}/nodes_raw_edges.json"
    else:
        nodes_in = f"{args.output_dir}/nodes.json"
        edges_in = f"{args.output_dir}/edges.json"

    # Rename crawled data to canonical names
    if args.mode == "crawl":
        import shutil
        canonical_nodes = f"{args.output_dir}/nodes.json"
        canonical_edges = f"{args.output_dir}/edges.json"
        if os.path.exists(nodes_in):
            if os.path.exists(canonical_nodes):
                os.remove(canonical_nodes)
            shutil.move(nodes_in, canonical_nodes)
            nodes_in = canonical_nodes
        if os.path.exists(edges_in):
            if os.path.exists(canonical_edges):
                os.remove(canonical_edges)
            shutil.move(edges_in, canonical_edges)
            edges_in = canonical_edges

    # ── Step 2: LLM reclassification ──
    nodes_reclassified = f"{args.output_dir}/nodes_reclassified.json"
    if args.no_reclassify:
        print("\n  Skipping reclassification (--no-reclassify)")
        nodes_reclassified = nodes_in
    else:
        cmd2 = (f'python reclassify_with_agent.py {nodes_in} '
                f'--only best_practice,generic,domain '
                f'--output {nodes_reclassified}')
        if not run(cmd2, "Step 2: LLM reclassification"):
            nodes_reclassified = nodes_in  # fall back to original

    # ── Step 3: Enrich (Quick Facts + link contexts + optional tutorials) ──
    nodes_enriched = f"{args.output_dir}/nodes_enriched.json"
    if args.no_tutorials:
        cmd3 = f'python enrich_nodes.py {nodes_reclassified} --output {nodes_enriched} --no-llm'
    else:
        cmd3 = f'python enrich_nodes.py {nodes_reclassified} --output {nodes_enriched}'

    if not run(cmd3, "Step 3: Enrich (Quick Facts + link contexts + tutorials)"):
        return

    # ── Step 4: Generate markdown ──
    cmd4 = f'python generate_markdown.py {nodes_enriched} {edges_in} --output {args.output_dir}/kb --single-file'
    if not run(cmd4, "Step 4: Generate markdown"):
        return

    # ── Step 5: Generate JSON KB (single-file, agent-friendly) ──
    cmd5 = f'python generate_json_kb.py {nodes_enriched} {edges_in} --output {args.output_dir}/kb.json'
    run(cmd5, "Step 5: Generate JSON knowledge base")

    # ── Step 6: Generate 2D graph ──
    if not args.no_graph:
        graph_max = "" if args.graph_max == 0 else f"--max {args.graph_max}"
        cmd6 = f'python graph2d.py {nodes_enriched} {edges_in} {graph_max} --output {args.output_dir}/graph.html'
        run(cmd6, "Step 6: Generate 2D graph")

    print(f"\n{'='*50}")
    print(f"  DONE")
    print(f"{'='*50}")
    print(f"  JSON KB:   {args.output_dir}/kb.json")
    print(f"  Markdown:  {args.output_dir}/kb/")
    print(f"  Graph:     {args.output_dir}/graph.html")
    print(f"  Node data: {nodes_enriched}")


if __name__ == "__main__":
    main()
