"""
Tests for classify_page() — the 50+ rule priority chain.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wiki_parser import classify_page


# ── Category-tag-based (highest priority after Category_ prefix) ──

def test_category_tag_incar():
    assert classify_page("ENCUT", ["INCAR tag"], "") == ("capability", "parameter")

def test_category_tag_tutorials():
    assert classify_page("Some_Tutorial", ["tutorials"], "") == ("procedure", "tutorial")

def test_category_tag_examples():
    assert classify_page("Example_Calculation", ["examples"], "") == ("procedure", "tutorial")

def test_category_tag_howto():
    assert classify_page("How_to_DoS", ["howto"], "") == ("procedure", "tutorial")

def test_category_tag_installation():
    assert classify_page("Installation_Guide", ["installation"], "") == ("procedure", "tutorial")

def test_category_tag_theory():
    assert classify_page("DFT_Theory", ["theory"], "") == ("capability", "domain")

def test_category_tag_pitfalls():
    assert classify_page("Common_Mistakes", ["common pitfalls"], "") == ("constraint", "pitfall")

def test_category_tag_known_issues():
    assert classify_page("Known_Bugs", ["known issues"], "") == ("constraint", "pitfall")

def test_category_tag_troubleshooting():
    assert classify_page("Fix_It", ["troubleshooting"], "") == ("constraint", "pitfall")

def test_category_tag_input_files():
    assert classify_page("INCAR_Guide", ["input files"], "") == ("procedure", "tutorial")


# ── Category_ prefix ──

def test_category_prefix():
    assert classify_page("Category_VASP", [], "") == ("capability", "domain")

def test_category_prefix_with_tags():
    assert classify_page("Category_INCAR", ["INCAR tag"], "") == ("capability", "domain")


# ── Domain category matching (broad categories → L1 domain) ──

def test_domain_category_magnetism():
    assert classify_page("Spin_Page", ["magnetism"], "") == ("capability", "domain")

def test_domain_category_phonons():
    assert classify_page("Phonon_Calc", ["phonons"], "") == ("capability", "domain")

def test_domain_category_gw():
    assert classify_page("GW_Guide", ["gw"], "") == ("capability", "domain")

def test_domain_category_hybrid():
    assert classify_page("HSE06", ["hybrid functionals"], "") == ("capability", "domain")

def test_domain_category_dft_plus_u():
    assert classify_page("DFT_Plus_U", ["dft+u"], "") == ("capability", "domain")


# ── Name-based heuristics ──

def test_name_all_caps_param():
    """ALL_CAPS + underscore pattern → INCAR parameter"""
    assert classify_page("ISIF", [], "") == ("capability", "parameter")

def test_name_all_caps_with_number():
    assert classify_page("LREAL", [], "") == ("capability", "parameter")

def test_name_all_caps_too_long():
    """Not_An_INCAR_Tag_With_A_Very_Long_Name_Exceeding_50_Characters_Total"""
    long_name = "A" * 60  # > 50 chars
    assert classify_page(long_name, [], "") == ("capability", "generic")

def test_name_tutorial_keyword():
    assert classify_page("How_To_Calculate_Band_Structure", [], "") == ("procedure", "tutorial")

def test_name_best_practice_keyword():
    assert classify_page("Best_Practices_for_Accuracy", [], "") == ("heuristic", "best_practice")

def test_name_convergence_keyword():
    assert classify_page("Convergence_Tests_for_ENCUT", [], "") == ("heuristic", "best_practice")

def test_name_pitfall_keyword():
    assert classify_page("Common_Pitfalls_in_MD", [], "") == ("constraint", "pitfall")


# ── Content-based heuristics ──

def test_content_theory_keyword():
    assert classify_page("Some_Page", [], "The Hamiltonian and wave function formalism") == ("capability", "generic")

def test_content_equation_keyword():
    assert classify_page("Some_Page", [], "The Kohn-Sham equations are derived from") == ("capability", "generic")

def test_content_warning_keyword():
    assert classify_page("Some_Page", [], "Warning: this feature is deprecated") == ("capability", "generic")

def test_content_tutorial_keyword():
    assert classify_page("Some_Page", [], "Step by step tutorial for calculating") == ("capability", "generic")

def test_content_prerequisite_keyword():
    assert classify_page("Some_Page", [], "Prerequisites: you need the POTCAR file") == ("capability", "generic")


# ── Fallback / edge cases ──

def test_skins_prefix_generic():
    assert classify_page("Skins.CustomTheme", [], "") == ("capability", "generic")

def test_construction_prefix_generic():
    assert classify_page("Construction_Draft", [], "") == ("capability", "generic")

def test_redirect_content_generic():
    assert classify_page("Old_Page", [], "Redirect to: New_Page") == ("capability", "generic")

def test_empty_content_generic():
    assert classify_page("Empty_Page", [], "") == ("capability", "generic")

def test_no_hints_generic():
    assert classify_page("Random_Unmatched_Page", [], "some random content with no keywords") == ("capability", "generic")


# ── Priority ordering: category tags beat name heuristics beat content ──

def test_tag_beats_name():
    """Even if name looks like a parameter, category tag takes priority"""
    assert classify_page("ENCUT", ["examples"], "tutorial steps here") == ("procedure", "tutorial")

def test_name_beats_content():
    """Name pattern ALL_CAPS beats content keywords"""
    assert classify_page("ISIF", [], "this is about theory and equations") == ("capability", "parameter")
