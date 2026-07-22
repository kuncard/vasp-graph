"""
Tests for deduplicate_nodes() — merging duplicate nodes by normalized title.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiki_parser import deduplicate_nodes


# ── No duplicates ──

def test_no_duplicates():
    nodes = [
        {"id": "A", "title": "Page A", "content": "hello", "tags": ["tag1"], "aliases": []},
        {"id": "B", "title": "Page B", "content": "world", "tags": ["tag2"], "aliases": []},
    ]
    edges = [{"source": "A", "target": "B", "relation": "wikilink"}]
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(nd) == 2
    assert len(ed) == 1


# ── Basic dedup: same normalized title ──

def test_duplicate_title_keeps_longest():
    nodes = [
        {"id": "ENCUT_v1", "title": "ENCUT", "content": "short", "tags": ["a"], "aliases": []},
        {"id": "ENCUT_v2", "title": "ENCUT", "content": "much longer content here", "tags": ["b"], "aliases": []},
    ]
    edges = [{"source": "ENCUT_v1", "target": "OTHER", "relation": "wikilink"}]
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(nd) == 1
    assert nd[0]["id"] == "ENCUT_v2"  # longer content wins
    assert nd[0]["content"] == "much longer content here"


# ── Tags and aliases merged from duplicates ──

def test_tags_merged():
    nodes = [
        {"id": "X1", "title": "X", "content": "longer", "tags": ["tag_a"], "aliases": ["alias1"]},
        {"id": "X2", "title": "X", "content": "shorter", "tags": ["tag_b"], "aliases": ["alias2"]},
    ]
    edges = []
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(nd) == 1
    assert set(nd[0]["tags"]) == {"tag_a", "tag_b"}
    assert set(nd[0]["aliases"]) == {"alias1", "alias2", "X"}  # X = dup title


# ── Edge remapping ──

def test_edges_remapped():
    nodes = [
        {"id": "dup_a", "title": "Same Page", "content": "longer content", "tags": [], "aliases": []},
        {"id": "dup_b", "title": "Same Page", "content": "short", "tags": [], "aliases": []},
        {"id": "Other", "title": "Other Page", "content": "...", "tags": [], "aliases": []},
    ]
    edges = [
        {"source": "dup_b", "target": "Other", "relation": "wikilink"},  # dup_b → canonical
        {"source": "Other", "target": "dup_b", "relation": "belongs_to"},
    ]
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(nd) == 2
    # dup_b edges should now point to dup_a (canonical)
    targets = {e["source"] for e in ed}
    sources = {e["target"] for e in ed}
    assert "dup_b" not in targets
    assert "dup_b" not in sources
    assert "dup_a" in targets or "dup_a" in sources


# ── Self-loop removed after remap ──

def test_self_loop_removed():
    nodes = [
        {"id": "canon", "title": "Canon", "content": "longer", "tags": [], "aliases": []},
        {"id": "dup", "title": "Canon", "content": "short", "tags": [], "aliases": []},
    ]
    edges = [
        {"source": "dup", "target": "canon", "relation": "wikilink"},  # becomes self-loop
    ]
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(ed) == 0  # self-loop removed


# ── Category_ and Category_talk_ NOT merged ──

def test_category_and_talk_not_merged():
    nodes = [
        {"id": "Category_VASP", "title": "VASP", "content": "category page content here", "tags": [], "aliases": []},
        {"id": "Category_talk_VASP", "title": "VASP", "content": "discussion about the category page", "tags": [], "aliases": []},
    ]
    nd, ed = deduplicate_nodes(nodes, [])
    assert len(nd) == 2  # both kept, different prefixes


# ── Category: prefix stripped in normalization ──

def test_category_colon_prefix_normalized():
    # Category_INCAR (id prefix "Category_") and INCAR (no prefix) get
    # different canonicalization prefixes ("cat:" vs ""), so they are NOT merged.
    # A category page and a regular page are different things.
    nodes = [
        {"id": "Category_INCAR", "title": "Category:INCAR", "content": "content A longer", "tags": ["a"], "aliases": []},
        {"id": "INCAR", "title": "INCAR", "content": "short", "tags": ["b"], "aliases": []},
    ]
    nd, ed = deduplicate_nodes(nodes, [])
    assert len(nd) == 2  # different id prefixes, not merged


# ── Underscore/space equivalence ──

def test_underscore_space_equivalence():
    nodes = [
        {"id": "A_B_C", "title": "A B C", "content": "longer content text here", "tags": [], "aliases": []},
        {"id": "A_B_C_v2", "title": "A_B_C", "content": "short", "tags": [], "aliases": []},
    ]
    nd, ed = deduplicate_nodes(nodes, [])
    assert len(nd) == 1
    assert nd[0]["id"] == "A_B_C"  # longer content


# ── Duplicate edge dedup within remapped ──

def test_duplicate_edges_after_remap():
    nodes = [
        {"id": "canon", "title": "Page", "content": "xxx", "tags": [], "aliases": []},
        {"id": "dup1", "title": "Page", "content": "x", "tags": [], "aliases": []},
        {"id": "dup2", "title": "Page", "content": "xx", "tags": [], "aliases": []},
        {"id": "Other", "title": "Other", "content": "...", "tags": [], "aliases": []},
    ]
    edges = [
        {"source": "dup1", "target": "Other", "relation": "wikilink"},
        {"source": "dup2", "target": "Other", "relation": "wikilink"},  # same edge after remap
    ]
    nd, ed = deduplicate_nodes(nodes, edges)
    # Only one wikilink from canon → Other (the remapped + deduped edge)
    remapped_count = sum(1 for e in ed if e["source"] == "canon" and e["target"] == "Other")
    assert remapped_count == 1


# ── Multiple relation types between same pair preserved ──

def test_different_relations_preserved():
    nodes = [
        {"id": "A", "title": "A", "content": "...", "tags": [], "aliases": []},
        {"id": "B", "title": "B", "content": "...", "tags": [], "aliases": []},
    ]
    edges = [
        {"source": "A", "target": "B", "relation": "wikilink"},
        {"source": "A", "target": "B", "relation": "belongs_to"},
    ]
    nd, ed = deduplicate_nodes(nodes, edges)
    assert len(ed) == 2  # different relations, both kept
