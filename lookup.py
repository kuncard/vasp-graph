"""
On-demand page lookup with automatic fallback to wiki mirror.

If the page is already in the knowledge base, return it directly.
If not, parse it from the wiki mirror, add it to the graph, and return it.

Usage:
    python lookup.py ENCUT                     # look up one page
    python lookup.py ENCUT ISPIN IBRION        # look up multiple pages
    python lookup.py --kb ../vasp_kb_crawl --wiki ../vasp/vasp/www.vasp.at/wiki/index.php/ ENCUT

This is what the Agent calls when it searches the KB and doesn't find something.
"""
import argparse, os, sys

# Add current dir to path so we can import wiki_parser
sys.path.insert(0, os.path.dirname(__file__))
from wiki_parser import parse_html, classify_page, clean_wiki_text


def lookup(page_name: str, kb_dir: str, wiki_dir: str) -> dict:
    """Find a page. Returns dict with 'source' ('kb' or 'wiki') and markdown content."""

    # 1. Check knowledge base
    kb_path = os.path.join(kb_dir, page_name + ".md")
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            return {"source": "kb", "page": page_name, "content": f.read()}

    # 2. Fallback to wiki mirror
    html_path = os.path.join(wiki_dir, page_name + ".html")
    if not os.path.exists(html_path):
        return {"source": "not_found", "page": page_name, "content": None,
                "suggestions": _suggest(page_name, kb_dir)}

    parsed = parse_html(html_path)
    if parsed is None:
        return {"source": "parse_error", "page": page_name, "content": None}

    body = parsed["body_text"]
    entry_type, subtype = classify_page(page_name, parsed["categories"], body)

    # Build markdown on the fly
    md = f"""---
id: {page_name}
title: "{parsed['title']}"
type: {entry_type}
subtype: {subtype}
category: "{parsed['categories'][0] if parsed['categories'] else ''}"
tags: [{', '.join(parsed['categories'][:8])}]
---

# {parsed['title']}

> **{subtype}** | Source: wiki mirror (not yet in KB)

## Content

{body}
"""
    return {"source": "wiki", "page": page_name, "content": md}


def _suggest(page_name: str, kb_dir: str) -> list[str]:
    """Suggest similar pages from the KB."""
    import re
    suggestions = []
    pattern = re.compile(re.escape(page_name), re.IGNORECASE)
    for fname in sorted(os.listdir(kb_dir)):
        if fname.endswith(".md") and pattern.search(fname):
            suggestions.append(fname[:-3])
        if len(suggestions) >= 5:
            break
    return suggestions


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="On-demand page lookup")
    p.add_argument("pages", nargs="+", help="Page name(s) to look up")
    p.add_argument("--kb", default="output/kb", help="Knowledge base directory")
    p.add_argument("--wiki", required=True, help="Wiki HTML mirror directory")
    p.add_argument("--json", action="store_true", help="Output as JSON (for agent consumption)")
    args = p.parse_args()

    for page_name in args.pages:
        result = lookup(page_name, args.kb, args.wiki)
        if args.json:
            import json
            print(json.dumps(result, ensure_ascii=False))
        else:
            if result["source"] == "not_found":
                print(f"NOT FOUND: {page_name}")
                if result.get("suggestions"):
                    print(f"  Suggestions: {', '.join(result['suggestions'])}")
            elif result["source"] == "parse_error":
                print(f"PARSE ERROR: {page_name}")
            else:
                print(f"[{result['source'].upper()}] {page_name}")
                if result["content"]:
                    print(result["content"][:500])
                    if len(result["content"]) > 500:
                        print("...")
