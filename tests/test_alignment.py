"""
Tests for Module 3 (CRISPR Target Alignment Engine) & Module 4 (CPU Processing)
"""

import pytest
from src.crispr_alignment import (
    match_pam,
    compute_mismatches,
    calculate_alignment_score,
    CRISPRTargetMatch,
)
from src.cpu_engine import CPUEngine
from src.sequence_utils import reverse_complement


# ============================================================================
# PAM Motif Matching Tests
# ============================================================================

def test_match_pam_spcas9_ngg():
    """Verify SpCas9 'NGG' matches AGG, CGG, GGG, TGG, but rejects AGA, AGT, etc."""
    assert match_pam("AGG", "NGG") is True
    assert match_pam("CGG", "NGG") is True
    assert match_pam("GGG", "NGG") is True
    assert match_pam("TGG", "NGG") is True

    assert match_pam("AGT", "NGG") is False
    assert match_pam("AGA", "NGG") is False
    assert match_pam("AAG", "NGG") is False


def test_match_pam_degenerate_iupac():
    """Verify degenerate IUPAC PAM pattern matching (e.g., SaCas9 'NNGRRT')."""
    # NNGRRT: N=any, N=any, G=G, R=A/G, R=A/G, T=T
    assert match_pam("ATGAAT", "NNGRRT") is True
    assert match_pam("CCGGGT", "NNGRRT") is True
    assert match_pam("CCGGGA", "NNGRRT") is False  # Ends with A instead of T


# ============================================================================
# Mismatch Calculation & Scoring Tests
# ============================================================================

def test_compute_mismatches_exact():
    """Perfect match yields 0 mismatches and full '|' visual."""
    guide = "GAGTCCGAGCAGAAGAAGAA"
    target = "GAGTCCGAGCAGAAGAAGAA"
    count, pos, seed_mm, distal_mm, visual = compute_mismatches(guide, target)
    assert count == 0
    assert len(pos) == 0
    assert seed_mm == 0
    assert distal_mm == 0
    assert visual == "||||||||||||||||||||"


def test_compute_mismatches_seed_vs_distal():
    """
    Test mismatch positioning:
    In a 20bp guide, positions 1-10 are distal, positions 11-20 are seed (PAM-proximal).
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Mutation at position 2 (distal: A -> T) and position 19 (seed: A -> C)
    target = "GTGTCCGAGCAGAAGAAGCA"
    count, pos, seed_mm, distal_mm, visual = compute_mismatches(guide, target)
    assert count == 2
    assert pos == [2, 19]
    assert distal_mm == 1
    assert seed_mm == 1
    assert visual == "|.||||||||||||||||.|"


def test_calculate_alignment_score():
    """Exact match = 100. Seed mismatch penalizes more heavily than distal mismatch."""
    score_exact = calculate_alignment_score(20, 0, 0, 0)
    assert score_exact == 100.0

    # 1 distal mismatch vs 1 seed mismatch
    score_distal_1 = calculate_alignment_score(20, 1, seed_mismatches=0, distal_mismatches=1)
    score_seed_1 = calculate_alignment_score(20, 1, seed_mismatches=1, distal_mismatches=0)

    assert score_distal_1 < 100.0
    assert score_seed_1 < score_distal_1  # Seed mismatch has greater penalty


# ============================================================================
# CPU Engine Alignment Integration Tests
# ============================================================================

def test_cpu_engine_exact_match_forward():
    """CPU engine finds exact forward-strand match with NGG PAM."""
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Construct sequence with 50bp flank + guide + GGG (PAM) + 50bp flank
    ref = ("A" * 50) + guide + "GGG" + ("T" * 50)
    engine = CPUEngine()
    result = engine.align(ref, "test_scaffold", guide, pam_pattern="NGG", max_mismatches=0)

    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.strand == "+"
    assert match.start_pos == 51
    assert match.end_pos == 51 + 23 - 1
    assert match.target_dna_sequence == guide
    assert match.pam_sequence == "GGG"
    assert match.mismatch_count == 0
    assert match.alignment_score == 100.0


def test_cpu_engine_reverse_complement_match():
    """CPU engine finds target on the reverse-complement strand."""
    guide = "GAGTCCGAGCAGAAGAAGAA"
    pam = "CGG"
    full_target_forward = guide + pam  # Target in 5'->3' on antisense strand

    # Place the reverse complement of full_target_forward onto the reference sequence
    ref_rc = reverse_complement(full_target_forward)
    ref = ("C" * 30) + ref_rc + ("G" * 30)

    engine = CPUEngine()
    result = engine.align(ref, "test_reverse", guide, pam_pattern="NGG", max_mismatches=0)

    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.strand == "-"
    assert match.mismatch_count == 0
    assert match.pam_sequence == pam


def test_cpu_engine_mismatch_filtering():
    """Targets exceeding max_mismatches must be excluded."""
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Target with 2 mismatches
    target_2mm = "GTGTCCGAGCAGAAGAAGCA" + "TGG"
    ref = ("A" * 20) + target_2mm + ("T" * 20)

    engine = CPUEngine()

    # Allowed up to 1 mismatch -> Should find 0 matches
    res_cutoff_1 = engine.align(ref, "test_scaffold", guide, pam_pattern="NGG", max_mismatches=1)
    assert len(res_cutoff_1.matches) == 0

    # Allowed up to 2 mismatches -> Should find 1 match
    res_cutoff_2 = engine.align(ref, "test_scaffold", guide, pam_pattern="NGG", max_mismatches=2)
    assert len(res_cutoff_2.matches) == 1
    assert res_cutoff_2.matches[0].mismatch_count == 2
