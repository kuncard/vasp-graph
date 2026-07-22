"""
LLM-powered classification for VASP Wiki pages.

Only classifies — does NOT rewrite content. Short prompt, short response, cheap API calls.

Usage:
    python reclassify_with_agent.py data/nodes.json --output data/nodes_reclassified.json

    # Dry-run: only reclassify non-parameter pages (parameters are already 95% accurate)
    python reclassify_with_agent.py data/nodes.json --output data/nodes_reclassified.json \\
        --skip parameter

    # Only do the worst offenders
    python reclassify_with_agent.py data/nodes.json --output data/nodes_reclassified.json \\
        --only best_practice,generic,domain
"""

from __future__ import annotations

import json, argparse, os, sys, time
from typing import Optional


def _get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: pip install openai")
        sys.exit(1)

    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    model = os.environ.get("RECLASSIFY_MODEL", "deepseek-chat")

    if not api_key:
        print("Error: set OPENAI_API_KEY or DASHSCOPE_API_KEY")
        sys.exit(1)
    return OpenAI(api_key=api_key, base_url=base_url), model


# ── Prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a VASP expert classifying wiki pages for a knowledge graph.

Output EXACTLY one line in this format:
<entry_type>:<subtype>

Valid combinations:
- capability:domain     — category page, topic overview, domain knowledge, foundational theory
- capability:parameter  — INCAR tag, input parameter, KPOINTS/POSCAR/POTCAR file format
- procedure:tutorial    — tutorial, how-to, example, step-by-step guide, calculation walkthrough
- heuristic:best_practice — operational advice, tips, recommended settings, discussion notes
- constraint:pitfall    — known issue, limitation, warning, common mistake, deprecated feature
- generic               — redirect page, stub, or truly unclassifiable

Rules:
- Category_talk_ pages (discussion about wiki organization) → heuristic:best_practice
- Pages that explain mathematical/physical formalism (not "how to use VASP" but "how VASP works internally") → capability:domain
- Pages that give practical advice on settings/parameters → heuristic:best_practice
- INCAR tags are ALWAYS capability:parameter, even if the page has theory content
- Tutorials/examples that show complete calculation workflows → procedure:tutorial
- Pages tagged "Common Pitfalls" → constraint:pitfall

Only output the classification line. No explanation."""


def classify_page(client, model: str, title: str, page_id: str, content: str, tags: list[str]) -> str:
    """Ask LLM to classify a single page. Returns 'entry_type:subtype' or 'generic'."""

    # Use first 600 chars as preview — enough for classification, cheap on tokens
    preview = content[:600] if content else "(empty)"

    user_msg = f"""Page ID: {page_id}
Title: {title}
Tags: {', '.join(tags) if tags else '(none)'}
Content preview: {preview}"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=30,
        )
        result = response.choices[0].message.content.strip().lower()
        return result
    except Exception as exc:
        print(f"  LLM error: {exc}")
        return "error"


def parse_classification(raw: str) -> tuple[str, str]:
    """Parse LLM output like 'capability:parameter' into (entry_type, subtype)."""
    raw = raw.strip().lower()
    if ":" in raw:
        parts = raw.split(":", 1)
        return parts[0].strip(), parts[1].strip()
    return raw, raw


# ── Main ──────────────────────────────────────────────────────────────


def reclassify(
    nodes_path: str,
    output_path: str,
    skip_types: Optional[list[str]] = None,
    only_types: Optional[list[str]] = None,
    delay: float = 0.2,
) -> None:
    client, model = _get_client()

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes_list = json.load(f)

    # ── Resume support ──
    classified_map: dict[str, str] = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for n in existing:
            if n.get("llm_classified"):
                classified_map[n["id"]] = f"{n['entry_type']}:{n['subtype']}"
        print(f"Resuming: {len(classified_map)} already classified")

    # ── Filter candidates ──
    candidates = []
    for n in nodes_list:
        nid = n["id"]
        st = n.get("subtype", "generic")
        if nid in classified_map:
            continue
        if only_types and st not in only_types:
            continue
        if skip_types and st in skip_types:
            continue
        candidates.append(n)

    print(f"Classifying {len(candidates)}/{len(nodes_list)} nodes with {model}")
    if skip_types:
        print(f"  Skipping: {skip_types}")
    if only_types:
        print(f"  Only: {only_types}")

    # ── Classification loop ──
    for i, n in enumerate(candidates):
        nid = n["id"]
        old_st = n.get("subtype", "generic")

        result = classify_page(
            client, model,
            title=n.get("title", nid),
            page_id=nid,
            content=n.get("content", ""),
            tags=n.get("tags", []),
        )

        if result == "error":
            continue

        entry_type, subtype = parse_classification(result)

        # Validate
        valid_types = {"capability", "procedure", "heuristic", "constraint", "generic"}
        if entry_type not in valid_types:
            print(f"  [{i+1}/{len(candidates)}] {nid}: invalid '{result}' — keeping as {old_st}")
            classified_map[nid] = f"{n['entry_type']}:{old_st}"
        else:
            if entry_type == "generic":
                subtype = "generic"
            classified_map[nid] = f"{entry_type}:{subtype}"
            if (i + 1) % 5 == 0 or old_st != subtype:
                arrow = "→" if old_st != subtype else "="
                print(f"  [{i+1}/{len(candidates)}] {nid}: {old_st} {arrow} {subtype}")

        if (i + 1) % 20 == 0:
            _save_output(nodes_list, classified_map, output_path)
            print(f"    checkpoint ({len(classified_map)} classified)")

        time.sleep(delay)

    # ── Final save ──
    _save_output(nodes_list, classified_map, output_path)

    # ── Stats ──
    changes = 0
    for n in nodes_list:
        c = classified_map.get(n["id"])
        if c:
            new_st = c.split(":", 1)[1] if ":" in c else c
            if n.get("subtype") != new_st:
                changes += 1

    print(f"\nDone. {len(classified_map)} classified, {changes} changed → {output_path}")
    print(f"Next: python generate_markdown.py {output_path} data/edges.json --output vasp_kb/")


def _save_output(nodes_list: list[dict], classified_map: dict[str, str], output_path: str):
    """Write nodes with updated classifications."""
    updated = []
    for n in nodes_list:
        n_copy = dict(n)
        c = classified_map.get(n["id"])
        if c:
            et, st = parse_classification(c)
            n_copy["entry_type"] = et
            n_copy["subtype"] = st
            n_copy["llm_classified"] = True
        else:
            n_copy["llm_classified"] = False
        updated.append(n_copy)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM reclassification for VASP wiki pages")
    parser.add_argument("nodes", help="Path to nodes.json")
    parser.add_argument("--output", "-o", default="data/nodes_reclassified.json")
    parser.add_argument("--skip", help="Comma-separated subtypes to skip (e.g. parameter)")
    parser.add_argument("--only", help="Comma-separated subtypes to ONLY process")
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()

    skip = [s.strip() for s in args.skip.split(",")] if args.skip else None
    only = [s.strip() for s in args.only.split(",")] if args.only else None

    reclassify(
        nodes_path=args.nodes,
        output_path=args.output,
        skip_types=skip,
        only_types=only,
        delay=args.delay,
    )
