"""
Import VASP knowledge graph into know-do-graph database.

Usage:
    python import_to_kdg.py data/enriched.json data/edges.json --db vasp_graph_kdg.db
"""

import json, argparse, os, uuid, re, unicodedata

# ── Lightweight English stemmer (no NLTK dependency) ──

# Suffix stripping rules ordered longest-first
_STEM_SUFFIXES = [
    "izational", "isation", "izations", "tational", "ational",
    "ization", "fulness", "ousness", "iveness", "ability",
    "alities", "alisms", "ements", "ations", "istics",
    "ement", "ments", "ation", "ities", "fully",
    "ingly", "ously", "istic", "izing", "ising",
    "ical", "able", "ible", "ness", "ment",
    "ship", "tion", "sion", "ally", "ated",
    "ized", "ised", "ting", "ring", "ling",
    "ding", "sing", "ives", "isms", "ives",
    "ion", "est", "ity", "ism", "ize",
    "ers", "ies", "ing", "als", "ves",
    "ed", "es", "ly", "al", "ic",
    "er", "or", "s",
]

def stem_word(w: str) -> str:
    """Return stem of a single word using simple suffix stripping."""
    w = w.lower()
    for suffix in _STEM_SUFFIXES:
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            return w[:-len(suffix)]
    return w


def index_terms(title: str, definition: str, content: str) -> list[str]:
    """Extract search index terms from a node: stem key content words.

    Returns list of unique stemmed words from title + definition, up to 30.
    These are appended to content so KDG ILIKE can cross-match variants.
    """
    # Extract meaningful words (4+ chars, alphabetic)
    text = f"{title} {definition} {(content or '')[:2000]}"
    words = set(re.findall(r"[a-zA-Z]{4,}", text.lower()))

    # Stem each word, keep unique stems that differ from the original
    stems: set[str] = set()
    for w in words:
        s = stem_word(w)
        if len(s) >= 4 and s != w:  # Only add if stemming actually changed it
            stems.add(s)

    # Sort for determinism
    return sorted(stems)[:30]

def slug_from_title(title: str) -> str:
    """Simple slug: lowercase, replace non-alphanum with hyphens."""
    slug = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", slug.lower()).strip("-")
    return slug or "entry"


def import_graph(nodes_path: str, edges_path: str, db_path: str):
    import os
    os.environ["KDG_DB_PATH"] = db_path

    from core.storage.database import create_database_engine
    from core.storage.models import Base
    from sqlalchemy.orm import Session
    from core.schemas.entry import Entry, EntryType, EntryMetadata
    from core.schemas.edge import Edge

    # ── Create DB ──
    engine = create_database_engine(db_path)
    Base.metadata.create_all(bind=engine)

    with open(nodes_path, encoding="utf-8") as f:
        nodes_list = json.load(f)
    with open(edges_path, encoding="utf-8") as f:
        edges_list = json.load(f)

    # ── Map our types to know-do-graph types ──
    type_map = {
        "capability": EntryType.capability,
        "procedure": EntryType.procedure,
        "heuristic": EntryType.heuristic,
        "constraint": EntryType.constraint,
        "generic": EntryType.generic,
        "memory": EntryType.memory,
    }
    rel_map = {
        "contains": "decomposes_to",
        "has_subcategory": "decomposes_to",
        "belongs_to": "wikilink",
        "wikilink": "wikilink",
    }

    id_map = {}  # our node_id -> kdg entry_id

    with Session(engine) as db:
        # ── Import nodes ──
        print(f"Importing {len(nodes_list)} nodes...")
        entries = []
        used_slugs = set()
        for i, n in enumerate(nodes_list):
            nid = n["id"]
            title = n.get("title", nid.replace("_", " "))
            entry_type = type_map.get(n.get("entry_type", "capability"), EntryType.capability)
            content = n.get("content", "") or ""
            tags = n.get("tags", []) or []
            s = n.get("structured", {}) or {}

            # Build Markdown content for know-do-graph
            md_lines = [f"# {title}", ""]
            qf = s.get("quick_facts", {}) or {}
            if qf:
                md_lines.append("## Quick Facts")
                md_lines.append(f"- Type: {qf.get('type', '?')}")
                md_lines.append(f"- Default: {qf.get('default', '?')}")
                md_lines.append("")
            defn = s.get("definition", "")
            if defn:
                md_lines.append(f"**Definition:** {defn}")
                md_lines.append("")
            opts = s.get("options", [])
            if opts:
                md_lines.append("## Options")
                for o in opts:
                    d = f" -- {o['description']}" if o.get("description") else ""
                    md_lines.append(f"- {o['value']}{d}")
                md_lines.append("")
            warns = s.get("warnings", [])
            if warns:
                md_lines.append("## Warnings")
                for w in warns:
                    md_lines.append(f"> {w}")
                md_lines.append("")
            ts = s.get("tutorial_summary", "")
            if ts:
                md_lines.append("## Summary")
                md_lines.append(ts)
                md_lines.append("")
            if content:
                # For parameter pages, skip the header (already in Quick Facts)
                body = content
                if qf and "---" in content:
                    body = content.split("---", 1)[1].strip()
                if body:
                    md_lines.append("## Content")
                    md_lines.append(body)

            kdg_id = str(uuid.uuid4())
            id_map[nid] = kdg_id

            slug = slug_from_title(title)
            # Ensure unique slug by appending counter if needed
            slug = f"{slug}-{i}" if slug in used_slugs else slug
            used_slugs.add(slug)

            md_content = "\n".join(md_lines)
            # Strip backslashes to avoid JSON serialization errors in know-do-graph API
            # (LaTeX commands like \partial, \nabla break JSON when unescaped)
            md_content = md_content.replace("\\", "")

            # Append stemmed index terms so KDG ILIKE can cross-match variants
            # e.g. "magnetic" search finds "Magnetism" page (stem → "magnet")
            idx = index_terms(title, defn, content)
            if idx:
                md_content += "\n\n<!-- idx: " + " ".join(idx) + " -->"
                # Also add top stemmed terms to tags for higher KDG match score (3 pts)
                # Merge with existing tags: orig first, then fill up with stems
                stem_tags = idx[:5]  # top 5 stems
                merged_tags = list(tags)[:8]
                for st in stem_tags:
                    if st not in merged_tags:
                        merged_tags.append(st)
                tags = merged_tags[:10]  # allow up to 10 tags

            entry = Entry(
                id=kdg_id,
                title=title,
                slug=slug,
                entry_type=entry_type,
                content=md_content,
                tags=tags[:10],
                metadata=EntryMetadata(custom={"subtype": n.get("subtype", "")}),
            )
            entries.append(entry)

            if (i + 1) % 200 == 0:
                print(f"  [{i+1}/{len(nodes_list)}]", flush=True)

        # Bulk insert
        from core.storage.models import EntryModel
        db_rows = []
        for e in entries:
            row = EntryModel(
                id=e.id, title=e.title, slug=e.slug,
                entry_type=e.entry_type.value, content=e.content,
                tags=json.dumps(e.tags, ensure_ascii=False),
                metadata_json=json.dumps(e.metadata.model_dump() if hasattr(e.metadata, 'model_dump') else {}, default=str, ensure_ascii=False),
                internal_refs=json.dumps(e.internal_refs, ensure_ascii=False),
                aliases=json.dumps([]),
                scripts_json=json.dumps([]),
                assets_json=json.dumps([]),
            )
            db_rows.append(row)
        db.add_all(db_rows)
        db.flush()
        print(f"  Done: {len(entries)} nodes")

        # ── Import edges ──
        print(f"Importing {len(edges_list)} edges...")
        edge_count = 0
        edge_batch = []
        for e in edges_list:
            src = id_map.get(e["source"])
            tgt = id_map.get(e["target"])
            if not src or not tgt:
                continue
            rel = rel_map.get(e.get("relation", "wikilink"), "wikilink")
            from core.storage.models import EdgeModel
            edge_batch.append(EdgeModel(
                id=str(uuid.uuid4()),
                source_id=src, target_id=tgt,
                relation=rel, weight=1.0,
                metadata_json="{}",
            ))
            edge_count += 1
            if len(edge_batch) >= 1000:
                db.add_all(edge_batch)
                db.flush()
                edge_batch = []
                print(f"  [{edge_count}/{len(edges_list)}]", flush=True)

        if edge_batch:
            db.add_all(edge_batch)
            db.flush()
        print(f"  Done: {edge_count} edges")

        db.commit()

    import os as _os
    size_mb = _os.path.getsize(db_path) / (1024 * 1024)
    print(f"\nDatabase: {db_path} ({size_mb:.1f} MB)")
    print(f"To serve: KDG_DB_PATH={db_path} know-do-graph serve")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Import VASP graph into know-do-graph")
    p.add_argument("nodes", help="nodes.json")
    p.add_argument("edges", help="edges.json")
    p.add_argument("--db", default="vasp_graph_kdg.db")
    args = p.parse_args()
    import_graph(args.nodes, args.edges, os.path.abspath(args.db))
