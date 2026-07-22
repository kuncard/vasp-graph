"""
Shared VASP Wiki HTML parsing, classification, and text cleaning.

Used by both parse_wiki.py (full dump) and crawl_graph.py (crawl-based).
"""

import os, re
from typing import Optional


def parse_html(filepath: str) -> Optional[dict]:
    """Parse a single VASP Wiki HTML page. Returns None if not found."""
    if not os.path.exists(filepath):
        return None

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        html = f.read()

    # ── Title ──
    title = ""
    title_m = re.search(r'<span class="mw-page-title-main">([^<]+)</span>', html)
    if not title_m:
        title_m = re.search(r"<title>(.*?)(?:\s*[-–]\s*VASP Wiki)?</title>", html)
    title = title_m.group(1).strip() if title_m else ""

    page_name = os.path.splitext(os.path.basename(filepath))[0]

    # ── Body content ──
    body_text = ""
    body_html = ""
    internal_links: list[str] = []

    # Three-level fallback for body extraction
    parser_m = re.search(
        r'class="mw-parser-output"[^>]*>(.*?)(?:<div[^>]*id="catlinks"|<div[^>]*class="printfooter")',
        html, re.DOTALL
    )
    if not parser_m:
        parser_m = re.search(
            r'class="mw-parser-output"[^>]*>(.*?)(?:</div>\s*</div>\s*(?:<div[^>]*id="catlinks"|<script))',
            html, re.DOTALL
        )
    if not parser_m:
        parser_m = re.search(
            r'class="mw-parser-output"[^>]*>(.*?)</div>\s*<script',
            html, re.DOTALL
        )

    if parser_m:
        body_html = parser_m.group(1)

        # Extract internal links
        for a in re.finditer(r'href="([^"]+)"', body_html):
            href = a.group(1)
            if href.startswith(("http", "#", "javascript", "mailto", "../images/", "../resources/")):
                continue
            if "Special:" in href or "Talk:" in href or "File:" in href:
                continue
            if href.endswith(".html"):
                target = href.replace(".html", "")
                internal_links.append(target)

        # Convert HTML body to Markdown
        body_text = _html_to_markdown(body_html)
        body_text = clean_wiki_text(body_text)

    # ── Categories ──
    categories: list[str] = []
    catlinks_m = re.search(r'id="mw-normal-catlinks"[^>]*>(.*?)</div>', html, re.DOTALL)
    if catlinks_m:
        cats = re.findall(r'title="Category:([^"]+)"', catlinks_m.group(1))
        categories = [c.strip() for c in cats if c.strip()]

    # ── Subcategories ──
    subcats: list[str] = []
    sc_m = re.search(r'id="mw-subcategories"[^>]*>(.*?)(?:id="mw-pages"|<noscript)', html, re.DOTALL)
    if sc_m:
        for a in re.finditer(r'href="([^"]+\.html)"', sc_m.group(1)):
            subcats.append(a.group(1).replace(".html", ""))

    # ── Member pages ──
    members: list[str] = []
    mp_m = re.search(r'id="mw-pages"[^>]*>(.*?)(?:<noscript|<div class="printfooter")', html, re.DOTALL)
    if mp_m:
        for a in re.finditer(r'href="([^"]+\.html)"', mp_m.group(1)):
            page = a.group(1).replace(".html", "")
            if not page.startswith(("Category_", "Special_", "Talk_")):
                members.append(page)

    return {
        "title": title or page_name.replace("_", " "),
        "page_name": page_name,
        "body_text": body_text[:16000] if len(body_text) > 16000 else body_text,
        "raw_html": body_html[:4000] if body_html and len(body_html) > 4000 else (body_html or ""),
        "internal_links": list(set(internal_links)),
        "categories": categories,
        "subcategories": subcats,
        "member_pages": members,
    }


# ═══════════════════════════════════════════════════════════════════════
# HTML → Markdown conversion
# ═══════════════════════════════════════════════════════════════════════

def _html_to_markdown(html: str) -> str:
    """Convert VASP Wiki HTML body to structured Markdown."""

    # ── MathML → LaTeX ──
    def _math_replace(m):
        latex = m.group(1)
        return '$' + latex + '$'

    # Replace math spans — extract LaTeX from annotation, discard everything else
    html = re.sub(
        r'<span class="mwe-math-element">.*?<annotation[^>]*>(.*?)</annotation>.*?</span>',
        _math_replace, html, flags=re.DOTALL
    )
    # Remove any remaining math images (already covered by span regex above)
    html = re.sub(r'<img[^>]*class="mwe-math[^"]*"[^>]*/?>', '', html)
    # Remove any remaining img tags
    html = re.sub(r'<img[^>]*/?>', '', html)

    # ── Structural HTML → Markdown ──
    html = re.sub(r'<h2[^>]*>(.*?)</h2>', r'\n## \1\n', html, flags=re.DOTALL)
    html = re.sub(r'<h3[^>]*>(.*?)</h3>', r'\n### \1\n', html, flags=re.DOTALL)
    html = re.sub(r'<h4[^>]*>(.*?)</h4>', r'\n#### \1\n', html, flags=re.DOTALL)
    html = re.sub(r'<b[^>]*>(.*?)</b>', r'**\1**', html, flags=re.DOTALL)
    html = re.sub(r'<strong[^>]*>(.*?)</strong>', r'**\1**', html, flags=re.DOTALL)
    html = re.sub(r'<i[^>]*>(.*?)</i>', r'*\1*', html, flags=re.DOTALL)
    html = re.sub(r'<em[^>]*>(.*?)</em>', r'*\1*', html, flags=re.DOTALL)
    html = re.sub(r'<code[^>]*>(.*?)</code>', r'`\1`', html, flags=re.DOTALL)
    html = re.sub(r'<pre[^>]*>(.*?)</pre>', r'\n```\n\1\n```\n', html, flags=re.DOTALL)
    html = re.sub(r'<br\s*/?>', '\n', html)
    html = re.sub(r'<hr\s*/?>', '\n---\n', html)

    # Lists
    html = re.sub(r'<li[^>]*>(.*?)</li>', r'- \1\n', html, flags=re.DOTALL)

    # Links
    html = re.sub(
        r'<a[^>]*class="mw-selflink"[^>]*>(.*?)</a>', r'\1', html, flags=re.DOTALL
    )
    html = re.sub(
        r'<a[^>]*href="([^"]+\.html)"[^>]*>(.*?)</a>',
        lambda m: '[[' + m.group(1).replace('.html', '') + '|' + re.sub(r'<[^>]+>', '', m.group(2)) + ']]',
        html, flags=re.DOTALL
    )
    html = re.sub(r'<a[^>]*href="([^"]+\.html)"[^>]*>(.*?)</a>',
                  lambda m: '[[' + m.group(1).replace('.html', '') + ']]',
                  html, flags=re.DOTALL)

    # Paragraphs
    html = re.sub(r'<p[^>]*>(.*?)</p>', r'\n\1\n', html, flags=re.DOTALL)

    # Divs with class hints
    html = re.sub(r'<div[^>]*class="[^"]*warning[^"]*"[^>]*>(.*?)</div>',
                  r'\n> **Warning:** \1\n', html, flags=re.DOTALL)
    html = re.sub(r'<div[^>]*class="[^"]*important[^"]*"[^>]*>(.*?)</div>',
                  r'\n> **Important:** \1\n', html, flags=re.DOTALL)

    # ── Strip remaining tags ──
    html = re.sub(r'<[^>]+>', ' ', html)

    # ── HTML entities ──
    html = html.replace('&#160;', ' ')
    html = re.sub(r'&#(\d+);', lambda m: chr(int(m.group(1))), html)
    html = re.sub(r'&#[xX]([0-9a-fA-F]+);', lambda m: chr(int(m.group(1), 16)), html)
    html = html.replace('&amp;', '&')
    html = html.replace('&lt;', '<')
    html = html.replace('&gt;', '>')
    html = html.replace('&quot;', '"')
    html = html.replace('&apos;', "'")
    html = re.sub(r'&[a-z]+;', ' ', html)

    # ── Whitespace ──
    html = re.sub(r'\n{3,}', '\n\n', html)
    html = re.sub(r'[ \t]{2,}', ' ', html)
    html = re.sub(r' *\n *', '\n', html)

    return html.strip()


# ═══════════════════════════════════════════════════════════════════════
# Text cleaning
# ═══════════════════════════════════════════════════════════════════════

def clean_wiki_text(text: str) -> str:
    """Remove wiki navigation residue and LaTeX artifacts."""

    # ── Navigation markers (only cut when in latter 60% of text) ──
    nav_markers = [
        r"\bRelated tags and (?:articles|sections)\b",
        r"\bExamples that use this tag\b",
        r"\bFurther things to try\b",
        r"\bList of tutorials\b",
        r"\bBack to the main page\b",
        r"\bRetrieved from\b",
        r"\bThe following \d+ pages? (?:is|are) in this category\b",
        r"\bPages in category\b",
        r"\bThis category (?:currently )?contains (?:only )?the following",
        r"\bDownload \w+\.(?:tgz|tar\.gz|zip)\b",
        r"\bSubcategories\b",
    ]
    for marker in nav_markers:
        m = re.search(marker, text, re.IGNORECASE)
        if m and m.start() > len(text) * 0.4:
            text = text[:m.start()].strip()

    # ── Protect $...$ math blocks ──
    math_blocks = []
    def _save_math(m):
        math_blocks.append(m.group(0))
        return f'<<<MATH{len(math_blocks)-1}>>>'
    text = re.sub(r'\$[^$]+\$', _save_math, text)

    # ── LaTeX cleanup (outside math blocks) ──
    text = re.sub(r'\{\\displaystyle\s+(.+)\}', r'\1', text)
    for cmd in ["mathrm", "textrm", "text", "mathit"]:
        text = re.sub(r'\{\\' + cmd + r'\{([^}]+)\}\}', r'\1', text)
    text = re.sub(r'\{\\mathbf\{([^}]+)\}\}', r'**\1**', text)
    for cmd in ["mathrm", "mathbf", "textrm", "text", "mathit", "displaystyle"]:
        text = re.sub(r'\\' + cmd + r'\{([^}]+)\}', r'\1', text)
        text = re.sub(r'\\' + cmd + r'\b', '', text)

    text = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', text)
    text = re.sub(r'\^\{([^{}]+)\}', r'^\1', text)
    text = re.sub(r'\_\{([^{}]+)\}', r'_\1', text)

    replacements = {
        "\\hbar": "ℏ", "\\pi": "π", "\\sigma": "σ", "\\Sigma": "Σ",
        "\\Gamma": "Γ", "\\Delta": "Δ", "\\Omega": "Ω", "\\infty": "∞",
        "\\times": "×", "\\cdot": "·", "\\ldots": "…", "\\approx": "≈",
        "\\equiv": "≡", "\\neq": "≠", "\\leq": "≤", "\\geq": "≥",
        "\\to": "→", "\\rightarrow": "→", "\\partial": "∂", "\\nabla": "∇",
        "\\int": "∫", "\\sum": "Σ", "\\prod": "∏", "\\sqrt": "√",
        "\\langle": "⟨", "\\rangle": "⟩", "\\epsilon": "ε", "\\alpha": "α",
        "\\beta": "β", "\\gamma": "γ", "\\delta": "δ", "\\lambda": "λ",
        "\\mu": "μ", "\\nu": "ν", "\\tau": "τ", "\\theta": "θ",
        "\\phi": "φ", "\\omega": "ω", "\\rho": "ρ", "\\eta": "η",
        "\\chi": "χ", "\\xi": "ξ", "\\zeta": "ζ",
    }
    for cmd, repl in replacements.items():
        text = text.replace(cmd, repl)

    text = re.sub(r'\\[a-zA-Z]+', '', text)
    for _ in range(3):
        new_text = re.sub(r'\{([^{}]*)\}', r'\1', text)
        if new_text == text:
            break
        text = new_text

    # ── Restore math blocks and clean LaTeX inside them ──
    def _clean_inline_math(latex: str) -> str:
        latex = re.sub(r'\{\\displaystyle\s+(.+)\}', r'\1', latex)
        latex = re.sub(r'\\mathrm\{([^{}]+)\}', r'\1', latex)
        latex = re.sub(r'\\mathbf\{([^{}]+)\}', r'\1', latex)
        latex = re.sub(r'\\textrm\{([^{}]+)\}', r'\1', latex)
        latex = re.sub(r'\\text\{([^{}]+)\}', r'\1', latex)
        latex = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', r'(\1)/(\2)', latex)
        for cmd, repl in replacements.items():
            latex = latex.replace(cmd, repl)
        latex = re.sub(r'\\[a-zA-Z]+', '', latex)
        for _ in range(3):
            new_latex = re.sub(r'\{([^{}]*)\}', r'\1', latex)
            if new_latex == latex:
                break
            latex = new_latex
        return latex

    for i, mb in enumerate(math_blocks):
        inner = mb[1:-1]  # strip $...$
        cleaned = _clean_inline_math(inner)
        text = text.replace(f'<<<MATH{i}>>>', f'${cleaned}$')

    # ── Whitespace ──
    text = re.sub(r'\$\s+', '$', text)
    text = re.sub(r'\s+\$', '$', text)
    # Normalize horizontal whitespace only; preserve newlines for Markdown structure
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text


# ═══════════════════════════════════════════════════════════════════════
# Classification
# ═══════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════
# Name-based classification (shared by classify_page + crawl BFS priority)
# ═══════════════════════════════════════════════════════════════════════

def classify_by_name(page_name: str) -> str | None:
    """Classify a page based solely on its name. Returns subtype or None."""
    if page_name.startswith("Category_"):
        return "domain"
    if re.match(r"^[A-Z][A-Z_0-9]+$", page_name) and len(page_name) < 50:
        return "parameter"
    name_lower = page_name.replace("_", " ").lower()
    if any(w in name_lower for w in ["tutorial", "how to", "howto", "calculation",
                                      "calculations", "structure optimization",
                                      "molecular dynamics"]):
        return "tutorial"
    if any(w in name_lower for w in ["best practice", "recommended", "convergence", "accuracy"]):
        return "best_practice"
    if any(w in name_lower for w in ["be careful", "beware", "do not", "avoid",
                                      "common pitfall", "known issue", "troubleshooting"]):
        return "pitfall"
    return None


# ═══════════════════════════════════════════════════════════════════════
# Full classification
# ═══════════════════════════════════════════════════════════════════════

def classify_page(page_name: str, categories: list[str], body_text: str = "") -> tuple[str, str]:
    """Return (entry_type, subtype). Full rule chain from parse_wiki.py."""
    cat_lower = [c.lower() for c in categories]
    cat_str = " ".join(cat_lower)

    # ── Category pages are always L1 domains ──
    if page_name.startswith("Category_"):
        return ("capability", "domain")

    # ── Category-tag-based classification (most reliable) ──
    if any("incar" in c for c in cat_lower):
        return ("capability", "parameter")
    if any(c in {"examples", "tutorials", "howto", "installation"} for c in cat_lower):
        return ("procedure", "tutorial")
    if any("theory" in c for c in cat_lower):
        return ("capability", "domain")
    if any(c in cat_str for c in {"common pitfalls", "known issues", "troubleshooting"}):
        return ("constraint", "pitfall")
    if any(c in cat_str for c in {"files", "input files", "output files", "pseudopotentials",
                                   "potcar tag", "incar"}):
        return ("procedure", "tutorial")

    # Domain topics → L1
    domain_cats = {
        "magnetism", "band structure", "density of states", "phonons",
        "exchange-correlation functionals", "hybrid functionals",
        "van der waals functionals", "dft+u", "dielectric properties",
        "molecular dynamics", "ionic minimization", "forces",
        "electrostatics", "symmetry", "crystal momentum",
        "electronic minimization", "electronic ground-state properties",
        "berry phases", "chemical shifts", "machine-learned force fields",
        "many-body perturbation theory", "bethe-salpeter equations",
        "time-dependent density-functional theory",
        "linear response", "transition states", "ensembles", "thermostats",
        "electron-phonon interactions", "wannier functions",
        "constrained-random-phase approximation", "gw", "acfdt", "mp2",
        "low-scaling gw and rpa", "xas", "nmr",
        "atoms and molecules", "defects", "spin spirals", "spin-orbit coupling",
        "noncollinear magnetism", "metadynamics", "biased molecular dynamics",
        "blue-moon ensemble", "constrained molecular dynamics",
        "interface pinning", "slow-growth approach", "thermodynamic integration",
        "advanced molecular-dynamics sampling", "ensemble properties",
        "density mixing", "charge density", "potential",
        "projector-augmented-wave method", "calculation setup", "electronic occupancy",
        "gpu", "hdf5 support", "development version", "version",
        "programming", "workshops", "vasp",
    }
    if any(c in cat_str for c in domain_cats):
        return ("capability", "domain")

    # ── Name-based heuristics ──
    name_type = classify_by_name(page_name)
    # ALL_CAPS name: only classify as parameter if tags confirm it or no tags at all
    if name_type == "parameter":
        if not cat_str or any(t in cat_str for t in ["incar", "potcar"]):
            return ("capability", "parameter")
    if name_type == "tutorial":
        return ("procedure", "tutorial")
    if name_type == "best_practice":
        return ("heuristic", "best_practice")
    if name_type == "pitfall":
        return ("constraint", "pitfall")

    # ── Content-based heuristics ──
    body_lower = body_text.lower() if body_text else ""
    if page_name.startswith(("Skins.", "skins.", "Construction_", "File_", "Template_",
                              "Category_talk_", "VASP_Wiki_")):
        return ("capability", "generic")
    if body_lower.startswith("redirect to:") or not body_text.strip():
        return ("capability", "generic")
    # Content keywords are too unreliable to force a specific type.
    # These pages will be caught by LLM reclassification instead.

    return ("capability", "generic")


# ═══════════════════════════════════════════════════════════════════════
# Parameter Quick Facts extraction (shared by crawl_graph + enrich_nodes)
# ═══════════════════════════════════════════════════════════════════════

def extract_param_facts(content: str) -> dict | None:
    """Parse VASP Wiki INCAR tag format. Handles multiple variants."""

    content = content.strip()
    # Strip Markdown bold markers that interfere with pattern matching
    content_clean = content.replace('**', '')

    # Skip non-parameter pages that got misclassified
    if not re.match(r'^[A-Z][A-Z_0-9]+\s*=', content_clean):
        if content_clean.startswith("The INCAR file is") or content_clean.startswith("You provide command"):
            return None
        if re.match(r'^\d+\.\d+\.\d+', content_clean):  # Changelog
            return None

    # Identify TAG name (first ALL_CAPS word before =)
    tag = ""
    tag_m = re.match(r'^([A-Z][A-Z_0-9]+)\s*=', content_clean)
    if tag_m:
        tag = tag_m.group(1)

    # ── Determine type from [brackets] or value list ──
    ptype = "unknown"
    # Match [type] in brackets, case-insensitive, allows modifiers like [3x3 real] or ≤[real]≤
    type_m = re.search(r'[=≤]\s*\[([^\]]*(?:real|integer|logical|string|complex|Real|Integer|Logical|String|Complex)[^\]]*)\]', content_clean)
    if type_m:
        raw_type = type_m.group(1).lower()
        if "real" in raw_type:
            ptype = "real"
        elif "integer" in raw_type:
            ptype = "integer"
        elif "logical" in raw_type:
            ptype = "logical"
        elif "string" in raw_type:
            ptype = "string"
        elif "complex" in raw_type:
            ptype = "complex"
        else:
            ptype = raw_type
    # Bare type keyword after = (e.g. "TAG = logical" or "TAG = real")
    elif re.search(r'=\s*real\b', content_clean, re.IGNORECASE):
        ptype = "real"
    elif re.search(r'=\s*logical\b', content_clean, re.IGNORECASE):
        ptype = "logical"
    elif re.search(r'=\s*integer\b', content_clean, re.IGNORECASE):
        ptype = "integer"
    elif re.search(r'=\s*string\b', content_clean, re.IGNORECASE):
        ptype = "string"
    elif re.search(r'=\s*complex\b', content_clean, re.IGNORECASE):
        ptype = "complex"
    else:
        # No type keyword — infer from value list
        val_m = re.search(r'=\s*(\.TRUE\.\s*\|\s*\.FALSE\.)', content_clean)
        if val_m:
            ptype = "logical"
        else:
            val_m = re.search(r'=\s*(-?\d+\s*\|\s*-?\d+)', content_clean)
            if val_m:
                ptype = "integer enum"
            else:
                val_m = re.search(r'=\s*(.+?)\s+Default:', content_clean)
                if val_m and '|' in val_m.group(1):
                    vals = val_m.group(1).strip()
                    ptype = "enum" if len(vals) < 80 else "string"

    # ── Extract default value ──
    default = ""
    default_m = re.search(r'Default:\s*(?:[A-Z_]+ = )?(.+?)(?:\s*Description:|\s*$)', content_clean)
    if default_m:
        default = default_m.group(1).strip()
        if len(default) > 120:
            default = default[:120].rsplit(" ", 1)[0] + "..."
        default = re.sub(r'\s*\{[^}]*\}\s*', '', default).strip()

    # ── Extract description ──
    desc = ""
    desc_m = re.search(r'Description:\s*(.+)', content, re.DOTALL)
    if desc_m:
        desc = desc_m.group(1).strip()
        if len(desc) > 1500:
            desc = desc[:1500].rsplit(" ", 1)[0]
    else:
        desc = content

    # ── Extract structured details ──
    details = _extract_param_details(content, content_clean, tag)

    return {
        "tag": tag or "?",
        "type": ptype,
        "default": default or "(see description)",
        "raw_description": desc,
        "definition": details.get("definition", ""),
        "options": details.get("options", []),
        "warnings": details.get("warnings", []),
    }


def _extract_param_details(content: str, content_clean: str, tag: str) -> dict:
    """Extract definition, options, and warnings from parameter page content."""
    result: dict = {"definition": "", "options": [], "warnings": []}

    # ── Definition: first sentence after Description: ──
    desc_m = re.search(r'Description:\s*(.+?)(?:\.\s|\.$|\s*---)', content_clean)
    if desc_m:
        result["definition"] = desc_m.group(1).strip()

    # ── Split at --- to get header and body ──
    parts = content_clean.split('---', 1)
    header = parts[0]
    body = parts[1] if len(parts) > 1 else ''

    # ── Options: extract from header value list and body descriptions ──
    options = []
    val_list_m = re.search(r'=\s*(.+?)(?:\s+Default:)', header)
    if val_list_m:
        vals_text = val_list_m.group(1).strip()
        # Skip if it's a type bracket like [real]
        if not re.match(r'^\[.*\]$', vals_text):
            # Split raw values by |
            raw_vals = [v.strip() for v in vals_text.split('|')]
            raw_vals = [v for v in raw_vals if v and v not in ('...',)]

            if len(raw_vals) > 1:
                # Try to find each value's description in the body
                for v in raw_vals:
                    desc = ''
                    # Look for "TAG =value description" or "- TAG =value description" in body
                    escaped_val = re.escape(v)
                    desc_m = re.search(
                        r'(?:^|\n)\s*(?:-\s*)?' + re.escape(tag) + r'\s*=\s*' + escaped_val + r'\s*(.*?)(?=\n|$)',
                        body, re.MULTILINE | re.IGNORECASE
                    )
                    if desc_m:
                        desc = desc_m.group(1).strip()
                    options.append({"value": v, "description": desc})
    if options:
        result["options"] = options

    # ── Warnings ──
    warn_keywords = [
        r'\bImportant\b', r'\bWarning\b', r'\bdeprecated\b',
        r'\bnot recommended\b', r'\bnot supported\b', r'\bstrongly recommend\b',
    ]
    # Split body into sentences, look for warning keywords
    sentences = re.split(r'(?<=[.!])\s+', body)
    for s in sentences:
        s = s.strip()
        if len(s) < 30 or len(s) > 400:
            continue
        for kw in warn_keywords:
            if re.search(kw, s, re.IGNORECASE):
                if s not in result["warnings"]:
                    result["warnings"].append(s)
                break

    if not result["options"]:
        del result["options"]
    if not result["warnings"]:
        del result["warnings"]

    return result


# ═══════════════════════════════════════════════════════════════════════
# Node deduplication (shared by crawl_graph + parse_wiki)
# ═══════════════════════════════════════════════════════════════════════

def deduplicate_nodes(nodes_list: list[dict], edges_list: list[dict]) -> tuple[list[dict], list[dict]]:
    """Merge duplicate nodes by normalized title. Keep longest content. Remap edges."""
    from collections import defaultdict

    def _norm(title: str, nid: str = "") -> str:
        t = title.strip().lower()
        if t.startswith("category:"):
            t = t[len("category:"):].strip()
        t = t.replace("_", " ")
        t = " ".join(t.split())
        if nid.startswith("Category_talk_"):
            return "talk:" + t
        if nid.startswith("Category_"):
            return "cat:" + t
        return t

    groups = defaultdict(list)
    for n in nodes_list:
        groups[_norm(n["title"], n["id"])].append(n)

    canonical: dict[str, str] = {}
    keep: dict[str, dict] = {}
    for norm_title, group in groups.items():
        if len(group) == 1:
            canonical[norm_title] = group[0]["id"]
            keep[group[0]["id"]] = group[0]
        else:
            best = max(group, key=lambda n: (len(n.get("content", "") or ""), len(n.get("tags", []))))
            canonical[norm_title] = best["id"]
            keep[best["id"]] = best
            all_tags = set(best.get("tags", []))
            all_aliases = set(best.get("aliases", []))
            for dup in group:
                all_tags.update(dup.get("tags", []))
                all_aliases.update(dup.get("aliases", []))
                all_aliases.add(dup.get("title", ""))
            best["tags"] = sorted(all_tags)
            best["aliases"] = sorted(a for a in all_aliases if a)

    dup_map = {n["id"]: canonical[_norm(n["title"], n["id"])] for n in nodes_list
               if n["id"] != canonical.get(_norm(n["title"], n["id"]))}

    remapped: list[dict] = []
    seen = set()
    for e in edges_list:
        src = dup_map.get(e["source"], e["source"])
        tgt = dup_map.get(e["target"], e["target"])
        if src == tgt:
            continue
        key = (src, tgt, e["relation"])
        if key not in seen:
            seen.add(key)
            remapped.append({"source": src, "target": tgt, "relation": e["relation"]})

    return list(keep.values()), remapped
