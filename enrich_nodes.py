"""
Enrich nodes with structured data for agent consumption.

Step 1 — Parameter Quick Facts: Regex extraction of type, default, description.
Step 2 — Parameter Details: LLM extraction of definition, options, warnings.
Step 3 — Tutorials: LLM extraction of steps, prerequisites, inputs, outputs.
Step 4 — Link context: Cross-reference target page first-sentence for wiki link descriptions.

Usage:
    # Full pipeline
    python enrich_nodes.py data/nodes_reclassified.json --output data/nodes_enriched.json

    # Skip LLM (do only parameter extraction + link context)
    python enrich_nodes.py data/nodes_reclassified.json --output data/nodes_enriched.json --no-llm
"""
from __future__ import annotations

import json, argparse, os, re, sys, time
from typing import Optional
from collections import defaultdict

from wiki_parser import extract_param_facts


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Parameter Quick Facts (rule-based, imported from wiki_parser)
# ═══════════════════════════════════════════════════════════════════════
# extract_param_facts is now imported from wiki_parser (shared module)
# ═══════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Tutorial extraction (LLM)
# ═══════════════════════════════════════════════════════════════════════

TUTORIAL_PROMPT = """Extract a structured summary from this VASP tutorial page. Output EXACTLY:

## Summary
[One sentence summarizing the goal]

## Prerequisites
- [item 1]
- [item 2]

## Steps
1. [first step]
2. [second step]

## Input Files
- [file or param]

## Output Files
- [file or result]

## Next Steps
- [what to do after]

Title: {title}
Content: {content}

Only output the structured sections, no preamble."""


def _get_llm():
    try:
        from openai import OpenAI
    except ImportError:
        return None, None
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DASHSCOPE_API_KEY", "")
    base_url = os.environ.get("OPENAI_API_BASE", "https://api.deepseek.com")
    model = os.environ.get("RECLASSIFY_MODEL", "deepseek-chat")
    if not api_key:
        return None, None
    return OpenAI(api_key=api_key, base_url=base_url), model


def extract_tutorial(client, model, title: str, content: str) -> str:
    """Ask LLM to structure a tutorial page."""
    prompt = TUTORIAL_PROMPT.format(title=title, content=content[:3000])
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        return f"(LLM error: {exc})"


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Link context (cross-reference)
# ═══════════════════════════════════════════════════════════════════════

def build_link_context(nodes_map: dict[str, dict]) -> dict[str, str]:
    """For each node, extract a one-line description usable as link context."""
    contexts = {}
    for nid, n in nodes_map.items():
        content = n.get("content", "")
        # Get first meaningful sentence
        sentences = re.split(r'(?<=[.!])\s+', content[:300])
        for s in sentences:
            s = s.strip()
            if len(s) > 10:
                contexts[nid] = s[:120]
                break
    return contexts


# ═══════════════════════════════════════════════════════════════════════
# LLM parameter detail extraction
# ═══════════════════════════════════════════════════════════════════════

PARAM_DETAIL_PROMPT = """Extract structured details from this VASP INCAR parameter page. Output ONLY valid JSON, no explanation.

{{
  "type": "string",
  "definition": "one-sentence definition of what this tag does",
  "options": [
    {{"value": "Normal", "description": "selects IALGO=38 blocked-Davidson"}},
    {{"value": "VeryFast", "description": "selects IALGO=48 RMM-DIIS"}}
  ],
  "warnings": [
    "hybrid functionals are not supported for VeryFast",
    "LMAXMIX must be set appropriately for fast convergence"
  ]
}}

Rules:
- definition: first sentence of Description, or infer from content
- options: list all possible values with a short description of each
- warnings: IMPORTANT/Warning/Note/Caution sentences, or deprecated/not-recommended/not-supported items
- If a field has no data, use empty array/string
- Output ONLY the JSON, nothing else

Title: {title}
Content: {content}"""

DEFINITION_PROMPT = """Write a one-sentence definition of this VASP Wiki page for a knowledge graph. Output ONLY the sentence, no quotes, no explanation.

Title: {title}
Content: {content}"""


def _llm_extract_definition(client, model: str, node: dict) -> str | None:
    """Use LLM to extract a one-sentence definition from a domain page."""
    content = node.get("content", "")
    if len(content) > 1500:
        content = content[:1500]
    prompt = DEFINITION_PROMPT.format(
        title=node.get("title", node["id"]),
        content=content,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )
        return resp.choices[0].message.content.strip().strip('"')
    except Exception:
        return None


def _llm_extract_param(client, model: str, node: dict) -> dict | None:
    """Use LLM to extract definition, options, warnings from a parameter page."""
    content = node.get("content", "")
    if len(content) > 2500:
        content = content[:2500]

    prompt = PARAM_DETAIL_PROMPT.format(
        title=node.get("title", node["id"]),
        content=content,
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        raw = resp.choices[0].message.content.strip()
        # Extract JSON from response (may have markdown fences)
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
        return json.loads(raw)
    except Exception as exc:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def enrich(nodes_path: str, output_path: str, use_llm: bool = True) -> None:
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes_list = json.load(f)

    nodes_map = {n["id"]: n for n in nodes_list}

    # ── Step 1: Parameter Quick Facts ──
    print("Step 1: Extracting parameter Quick Facts (rule-based)...")
    param_count = 0
    for n in nodes_list:
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
            param_count += 1

    print(f"  {param_count}/{sum(1 for n in nodes_list if n['subtype']=='parameter')} parameters extracted")

    # ── Step 1.5: Domain page definition (LLM) ──
    client, model = _get_llm()
    if client:
        print(f"Step 1.5: Domain page definitions ({model})...")
        domains = [n for n in nodes_list if n.get("subtype") == "domain"
                   and not n.get("structured", {}).get("definition")]
        if domains:
            fixed = 0
            for i, n in enumerate(domains):
                defn = _llm_extract_definition(client, model, n)
                if defn:
                    n.setdefault("structured", {})["definition"] = defn
                    fixed += 1
                if (i + 1) % 20 == 0:
                    print(f"    [{i+1}/{len(domains)}] {fixed} fixed", flush=True)
                time.sleep(0.3)
            print(f"  Domain definitions: {fixed}/{len(domains)}")
        else:
            print("  All domains have definitions, skipping")
    else:
        print("Step 1.5: No LLM available, skipping")

    # ── Step 2: LLM parameter detail extraction ──
    client, model = _get_llm()
    if client:
        print(f"Step 2: LLM parameter detail extraction ({model})...")
        params = [n for n in nodes_list if n.get("subtype") == "parameter"]
        needs_llm = []
        for n in params:
            s = n.get("structured", {})
            qf = s.get("quick_facts", {})
            if qf.get("type") == "unknown":
                needs_llm.append(n)
            elif s.get("options") and len(s.get("options", [])) >= 3:
                pass
            elif any(w in (n.get("content", "")[:200]).lower()
                     for w in ["default:", "description:"]):
                pass

        if needs_llm:
            print(f"  {len(needs_llm)} parameters need LLM extraction")
            fixed = 0
            for i, n in enumerate(needs_llm):
                details = _llm_extract_param(client, model, n)
                if details:
                    s = n.setdefault("structured", {})
                    if details.get("definition"):
                        s["definition"] = details["definition"]
                    if details.get("options"):
                        s["options"] = details["options"]
                    if details.get("warnings"):
                        s["warnings"] = details["warnings"]
                    if details.get("type"):
                        s.setdefault("quick_facts", {})["type"] = details["type"]
                    fixed += 1
                if (i + 1) % 10 == 0:
                    print(f"    [{i+1}/{len(needs_llm)}] {fixed} fixed", flush=True)
                time.sleep(0.3)
            print(f"  LLM fixed: {fixed}/{len(needs_llm)}")
        else:
            print("  All parameters OK, skipping LLM")
    else:
        print("Step 2: No LLM available, skipping")

    # ── Step 3: Tutorial extraction (LLM) ──
    if use_llm:
        client, model = _get_llm()
        if client is None:
            print("Step 3: No LLM available, skipping tutorial extraction")
        else:
            print(f"Step 3: Extracting tutorial structure with {model}...")
            tutorials = [n for n in nodes_list if n.get("subtype") == "tutorial"]

            # Resume support
            done = {n["id"] for n in nodes_list if n.get("structured", {}).get("tutorial_summary")}

            for i, n in enumerate(tutorials):
                if n["id"] in done:
                    continue

                result = extract_tutorial(
                    client, model,
                    title=n.get("title", n["id"]),
                    content=n.get("content", ""),
                )

                if result and not result.startswith("(LLM error"):
                    n.setdefault("structured", {})["tutorial_summary"] = result

                if (i + 1) % 5 == 0:
                    print(f"  [{i+1}/{len(tutorials)}]", flush=True)

                time.sleep(0.3)

            print(f"  Done: {sum(1 for n in tutorials if n.get('structured',{}).get('tutorial_summary'))}/{len(tutorials)} tutorials")
    else:
        print("Step 3: Skipped (--no-llm)")

    # ── Step 3: Link context ──
    print("Step 4: Building link context...")
    link_contexts = build_link_context(nodes_map)
    # Attach context map to output (generate_markdown.py will use it)
    for n in nodes_list:
        n["_link_context"] = {}  # will be filled on write
    print(f"  {len(link_contexts)} contexts built")

    # ── Save ──
    # Store link_contexts globally in a sidecar file
    context_path = output_path.replace(".json", "_link_contexts.json")
    with open(context_path, "w", encoding="utf-8") as f:
        json.dump(link_contexts, f, ensure_ascii=False, indent=2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nodes_list, f, ensure_ascii=False, indent=2)

    # Stats
    quick_fact_count = sum(1 for n in nodes_list if n.get("structured", {}).get("quick_facts"))
    tutorial_count = sum(1 for n in nodes_list if n.get("structured", {}).get("tutorial_summary"))

    print(f"\nDone: {output_path}")
    print(f"  Parameters with Quick Facts: {quick_fact_count}")
    print(f"  Tutorials with structure: {tutorial_count}")
    print(f"  Link contexts: {context_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich nodes for agent consumption")
    parser.add_argument("nodes")
    parser.add_argument("--output", "-o", default="data/nodes_enriched.json")
    parser.add_argument("--no-llm", action="store_true", help="Skip tutorial LLM extraction")
    args = parser.parse_args()

    enrich(args.nodes, args.output, use_llm=not args.no_llm)
