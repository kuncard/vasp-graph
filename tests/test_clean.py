"""
Tests for clean_wiki_text() — navigation removal and LaTeX cleanup.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiki_parser import clean_wiki_text


# ── LaTeX command replacements ──

def test_frac_to_division():
    assert "\\frac{a}{b}" not in clean_wiki_text("\\frac{a}{b}")
    assert "(a)/(b)" in clean_wiki_text("\\frac{a}{b}")

def test_superscript():
    assert "^x" in clean_wiki_text("^{x}")

def test_subscript():
    assert "_x" in clean_wiki_text("_{x}")

def test_greek_letters():
    result = clean_wiki_text("\\pi \\sigma \\Gamma \\Delta \\Omega")
    assert "π" in result
    assert "σ" in result
    assert "Γ" in result
    assert "Δ" in result
    assert "Ω" in result

def test_math_symbols():
    result = clean_wiki_text("\\times \\cdot \\approx \\equiv \\neq")
    assert "×" in result
    assert "·" in result
    assert "≈" in result
    assert "≡" in result
    assert "≠" in result

def test_comparison_operators():
    result = clean_wiki_text("\\leq \\geq")
    assert "≤" in result
    assert "≥" in result

def test_arrows():
    result = clean_wiki_text("\\to \\rightarrow")
    assert "→" in result

def test_calculus_symbols():
    result = clean_wiki_text("\\partial \\nabla \\int \\sum \\prod \\sqrt")
    assert "∂" in result
    assert "∇" in result
    assert "∫" in result
    assert "Σ" in result
    assert "∏" in result
    assert "√" in result

def test_brackets():
    result = clean_wiki_text("\\langle x \\rangle")
    assert "⟨" in result
    assert "⟩" in result

def test_other_greek():
    result = clean_wiki_text("\\hbar \\infty \\ldots")
    assert result.count("ħ") == 0  # hbar → 'h', not a special char
    assert "∞" in result
    assert "…" in result


# ── Display-style commands stripped ──

def test_displaystyle_removed():
    result = clean_wiki_text("\\displaystyle \\frac{a}{b}")
    assert "displaystyle" not in result

def test_math_font_commands_removed():
    result = clean_wiki_text("\\mathrm{some} \\mathbf{bold} \\textrm{text} \\mathit{italic}")
    assert "mathrm" not in result
    assert "mathbf" not in result
    assert "textrm" not in result
    assert "mathit" not in result

def test_unknown_latex_command_stripped():
    result = clean_wiki_text("\\someobscurecmd{arg}")
    assert "someobscurecmd" not in result
    assert "arg" in result  # content inside braces preserved


# ── Navigation suffix removal ──

def test_related_tags_removed():
    # Marker must appear after 40% of text for the function to cut it
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore. Related tags and articles: ENCUT, ISIF"
    result = clean_wiki_text(text)
    assert "Related tags and articles" not in result

def test_examples_that_use_tag_removed():
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Examples that use this tag: tutorial 1"
    result = clean_wiki_text(text)
    assert "Examples that use this tag" not in result

def test_retrieved_from_removed():
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore. Retrieved from VASP Wiki"
    result = clean_wiki_text(text)
    assert "Retrieved from" not in result

def test_pages_in_category_removed():
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore. The following 5 pages are in this category: A, B, C"
    result = clean_wiki_text(text)
    assert "following 5 pages" not in result

def test_download_link_removed():
    text = "Lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Download example.tgz for more"
    result = clean_wiki_text(text)
    assert "Download example.tgz" not in result


# ── Whitespace normalization ──

def test_extra_whitespace_normalized():
    result = clean_wiki_text("hello    world  \n\n  foo")
    assert "hello world\n\nfoo" in result


# ── Nothing to clean ──

def test_plain_text_unchanged():
    text = "Just a normal sentence with no markup."
    assert clean_wiki_text(text) == text
