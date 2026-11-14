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


# ============================================================================
# Dynamic Scoring and Coordinate Property Tests (Phase 4 - Step 2)
# ============================================================================

def test_dynamic_alignment_score_20bp_backward_compatibility():
    """Verify exact numerical backward compatibility for standard 20bp guides."""
    # 0 mismatches = 100.0
    assert calculate_alignment_score(20, 0, 0, 0) == 100.0

    # 1 distal mismatch: 100 * (1 - 1.0 / 35.0) = 97.14
    assert calculate_alignment_score(20, 1, seed_mismatches=0, distal_mismatches=1) == 97.14

    # 1 seed mismatch: 100 * (1 - 2.5 / 35.0) = 92.86
    assert calculate_alignment_score(20, 1, seed_mismatches=1, distal_mismatches=0) == 92.86

    # 1 distal + 1 seed mismatch: 100 * (1 - 3.5 / 35.0) = 90.00
    assert calculate_alignment_score(20, 2, seed_mismatches=1, distal_mismatches=1) == 90.00

    # All 20 bases mismatched (10 distal, 10 seed): penalty = 35.0 -> score = 0.0
    assert calculate_alignment_score(20, 20, seed_mismatches=10, distal_mismatches=10) == 0.0


def test_dynamic_alignment_score_variable_guide_lengths():
    """Verify dynamic normalization for non-20bp guide lengths (e.g. 17bp tru-gRNA and 24bp)."""
    # 17 bp guide (seed_len = 10, distal_len = 7, max_penalty = 7*1.0 + 10*2.5 = 32.0)
    assert calculate_alignment_score(17, 0, 0, 0) == 100.0
    # 2 distal mismatches on 17bp guide: 100 * (1 - 2.0 / 32.0) = 93.75
    assert calculate_alignment_score(17, 2, seed_mismatches=0, distal_mismatches=2) == 93.75
    # 1 seed mismatch on 17bp guide: 100 * (1 - 2.5 / 32.0) = 92.19
    assert calculate_alignment_score(17, 1, seed_mismatches=1, distal_mismatches=0) == 92.19

    # 24 bp guide (seed_len = 10, distal_len = 14, max_penalty = 14*1.0 + 10*2.5 = 39.0)
    assert calculate_alignment_score(24, 0, 0, 0) == 100.0
    # 1 distal mismatch on 24bp guide: 100 * (1 - 1.0 / 39.0) = 97.44
    assert calculate_alignment_score(24, 1, seed_mismatches=0, distal_mismatches=1) == 97.44

    # Extremely short 8 bp guide (seed_len = 8, distal_len = 0, max_penalty = 8*2.5 = 20.0)
    # 2 seed mismatches: 100 * (1 - 5.0 / 20.0) = 75.00
    assert calculate_alignment_score(8, 2, seed_mismatches=2, distal_mismatches=0) == 75.00


def test_crispr_target_match_forward_coordinate_properties():
    """Verify target_start, target_end, pam_start, pam_end on forward (+) strand."""
    target_seq = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp
    pam_seq = "TGG"                     # 3 bp
    match = CRISPRTargetMatch(
        reference_id="chr1",
        strand="+",
        start_pos=101,
        end_pos=123,
        target_dna_sequence=target_seq,
        pam_sequence=pam_seq,
        full_window=target_seq + pam_seq,
        mismatch_count=0,
        mismatch_positions=[],
        seed_mismatches=0,
        distal_mismatches=0,
        alignment_score=100.0,
        alignment_visual="|" * 20,
    )
    # Forward strand: target is 5', PAM is 3'
    assert match.target_start == 101
    assert match.target_end == 120
    assert match.pam_start == 121
    assert match.pam_end == 123

    # Check intervals
    assert match.target_end - match.target_start + 1 == 20
    assert match.pam_end - match.pam_start + 1 == 3
    assert match.end_pos - match.start_pos + 1 == 23

    # Verify to_dict contains new coordinate keys
    d = match.to_dict()
    assert d["Target Start"] == 101
    assert d["Target End"] == 120
    assert d["PAM Start"] == 121
    assert d["PAM End"] == 123


def test_crispr_target_match_reverse_coordinate_properties():
    """Verify target_start, target_end, pam_start, pam_end on reverse (-) strand."""
    target_seq = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp (5'->3' on antisense)
    pam_seq = "CGG"                     # 3 bp (5'->3' on antisense)
    match = CRISPRTargetMatch(
        reference_id="chr1",
        strand="-",
        start_pos=201,
        end_pos=223,
        target_dna_sequence=target_seq,
        pam_sequence=pam_seq,
        full_window=target_seq + pam_seq,
        mismatch_count=0,
        mismatch_positions=[],
        seed_mismatches=0,
        distal_mismatches=0,
        alignment_score=100.0,
        alignment_visual="|" * 20,
    )
    # Reverse strand: on forward reference, PAM complement is left (lower coordinates),
    # target complement is right (higher coordinates)
    assert match.pam_start == 201
    assert match.pam_end == 203
    assert match.target_start == 204
    assert match.target_end == 223

    # Check intervals
    assert match.pam_end - match.pam_start + 1 == 3
    assert match.target_end - match.target_start + 1 == 20
    assert match.end_pos - match.start_pos + 1 == 23

    # Verify to_dict contains new coordinate keys
    d = match.to_dict()
    assert d["Target Start"] == 204
    assert d["Target End"] == 223
    assert d["PAM Start"] == 201
    assert d["PAM End"] == 203


def test_cpu_engine_ground_truth_subsequence_coordinate_slices():
    """
    Ground-truth end-to-end integration:
    Verify that Python string slicing on the forward reference using
    target_start, target_end, pam_start, pam_end returns the exact
    expected sequences for both forward and reverse matches.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp
    pam_fwd = "GGG"
    pam_rev = "CGG"

    # Forward target: 50 bp A + guide + GGG + 20 bp T
    # Start of fwd target: 51. Window: 51 to 73.
    # Reverse target: place reverse_complement(guide + CGG) flanked by C and G
    rev_target_seq = guide + pam_rev
    rev_target_rc = reverse_complement(rev_target_seq)

    ref = ("A" * 50) + (guide + pam_fwd) + ("C" * 40) + rev_target_rc + ("T" * 30)
    engine = CPUEngine()
    result = engine.align(ref, "scaffold_gt", guide, pam_pattern="NGG", max_mismatches=0)

    assert len(result.matches) == 2
    fwd_match = next(m for m in result.matches if m.strand == "+")
    rev_match = next(m for m in result.matches if m.strand == "-")

    # 1. Forward match coordinate slice verification:
    # 1-based start 51, end 73
    assert fwd_match.start_pos == 51
    assert fwd_match.end_pos == 73
    assert fwd_match.target_start == 51
    assert fwd_match.target_end == 70
    assert fwd_match.pam_start == 71
    assert fwd_match.pam_end == 73
    # Slices on forward reference
    assert ref[fwd_match.start_pos - 1 : fwd_match.end_pos] == fwd_match.full_window
    assert ref[fwd_match.target_start - 1 : fwd_match.target_end] == fwd_match.target_dna_sequence
    assert ref[fwd_match.pam_start - 1 : fwd_match.pam_end] == fwd_match.pam_sequence

    # 2. Reverse match coordinate slice verification:
    # Offset of rev_target_rc: 50 + 23 + 40 = 113 (0-based) -> 1-based start 114
    # Length of rev_target_rc is 23 -> 1-based end is 114 + 23 - 1 = 136
    assert rev_match.start_pos == 114
    assert rev_match.end_pos == 136
    assert rev_match.pam_start == 114
    assert rev_match.pam_end == 116
    assert rev_match.target_start == 117
    assert rev_match.target_end == 136
    # Slices on forward reference must equal reverse complement of antisense match
    assert ref[rev_match.start_pos - 1 : rev_match.end_pos] == reverse_complement(rev_match.full_window)
    assert ref[rev_match.pam_start - 1 : rev_match.pam_end] == reverse_complement(rev_match.pam_sequence)
    assert ref[rev_match.target_start - 1 : rev_match.target_end] == reverse_complement(rev_match.target_dna_sequence)


# ============================================================================
# Edge-Case and Boundary Tests (Phase 4 - Step 4)
# ============================================================================

def test_alignment_short_reference_returns_empty_safely():
    """
    Requirement 1: Reference shorter than sliding window returns empty result safely.
    Tests both empty reference (0 bp) and short reference (8 bp < 23 bp window).
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp
    engine = CPUEngine()

    # 1. Zero-length empty reference
    res_empty = engine.align("", "empty_ref", guide, pam_pattern="NGG")
    assert len(res_empty.matches) == 0
    assert res_empty.total_windows_evaluated == 0
    assert res_empty.execution_time_seconds >= 0.0

    # 2. Reference shorter than window (window = 23 bp)
    res_short = engine.align("ATGCATGC", "short_ref", guide, pam_pattern="NGG")
    assert len(res_short.matches) == 0
    assert res_short.total_windows_evaluated == 0


def test_alignment_no_pam_in_reference():
    """
    Requirement 2: Reference with matching guide sequence but no valid PAM returns 0 matches.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Flanked by AAA and TTT (no G bases for NGG PAM)
    ref = ("A" * 30) + guide + "AAA" + ("T" * 30)
    engine = CPUEngine()
    res = engine.align(ref, "no_pam_ref", guide, pam_pattern="NGG", max_mismatches=0)

    assert len(res.matches) == 0
    assert res.total_windows_evaluated > 0


def test_alignment_variable_guide_length_17bp_and_24bp():
    """
    Requirement 3: Variable guide lengths within supported range (17 bp tru-gRNA and 24 bp extended).
    Verifies exact forward and reverse coordinates, window lengths, and sub-intervals.
    """
    engine = CPUEngine()

    # 1. 17 bp tru-gRNA (Fu et al. 2014) + NGG (3 bp) -> window = 20 bp
    guide_17 = "GAGTCCGAGCAGAAGAA"  # 17 bp
    pam_17 = "TGG"
    fwd_17_full = guide_17 + pam_17  # 20 bp
    ref_17 = ("C" * 20) + fwd_17_full + ("G" * 20)

    res_17 = engine.align(ref_17, "trugRNA_17", guide_17, pam_pattern="NGG", max_mismatches=0)
    assert len(res_17.matches) == 1
    m17 = res_17.matches[0]
    assert m17.strand == "+"
    assert m17.start_pos == 21
    assert m17.end_pos == 40
    assert m17.target_start == 21
    assert m17.target_end == 37  # 21 + 17 - 1 = 37
    assert m17.pam_start == 38   # 21 + 17 = 38
    assert m17.pam_end == 40
    assert m17.target_dna_sequence == guide_17
    assert m17.pam_sequence == pam_17
    assert m17.mismatch_count == 0
    assert m17.alignment_score == 100.0

    # 2. 24 bp extended guide + NGG (3 bp) -> window = 27 bp
    guide_24 = "GAGTCCGAGCAGAAGAAGAAGATC"  # 24 bp
    pam_24 = "CGG"
    fwd_24_full = guide_24 + pam_24  # 27 bp
    # Place on reverse strand
    ref_24 = ("A" * 30) + reverse_complement(fwd_24_full) + ("T" * 30)

    res_24 = engine.align(ref_24, "extended_24", guide_24, pam_pattern="NGG", max_mismatches=0)
    assert len(res_24.matches) == 1
    m24 = res_24.matches[0]
    assert m24.strand == "-"
    # 0-based offset of reverse window: 30 -> 1-based start: 31
    assert m24.start_pos == 31
    assert m24.end_pos == 31 + 27 - 1  # 57
    assert m24.pam_start == 31
    assert m24.pam_end == 33           # 3 bp PAM
    assert m24.target_start == 34      # 31 + 3 = 34
    assert m24.target_end == 57        # 24 bp target
    assert m24.target_dna_sequence == guide_24
    assert m24.pam_sequence == pam_24
    assert m24.alignment_score == 100.0


def test_alignment_exact_mismatch_positions_and_scoring_combinations():
    """
    Requirement 7: Exact mismatch positions, seed vs distal partitioning, and scoring correctness.
    Tests specific known mismatch positions:
    - Position 1: Distal 5' boundary
    - Position 10: Distal-seed boundary
    - Position 11: Seed start boundary
    - Position 20: Seed PAM-adjacent 3' boundary
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp

    # Mutate positions 1 (G->A), 10 (C->T), 11 (A->C), 20 (A->C)
    guide_list = list(guide)
    guide_list[0] = "A"   # pos 1
    guide_list[9] = "T"   # pos 10
    guide_list[10] = "C"  # pos 11
    guide_list[19] = "C"  # pos 20
    mutated_target = "".join(guide_list)
    pam = "AGG"
    ref = ("C" * 20) + (mutated_target + pam) + ("G" * 20)

    engine = CPUEngine()
    res = engine.align(ref, "scaffold_mm", guide, pam_pattern="NGG", max_mismatches=4)

    assert len(res.matches) == 1
    m = res.matches[0]
    assert m.mismatch_count == 4
    assert m.mismatch_positions == [1, 10, 11, 20]
    assert m.distal_mismatches == 2  # positions 1 and 10
    assert m.seed_mismatches == 2    # positions 11 and 20
    assert m.alignment_visual == ".||||||||..||||||||."

    # Score calculation:
    # distal: 2 * 1.0 = 2.0
    # seed: 2 * 2.5 = 5.0
    # total penalty = 7.0
    # max penalty = 10 * 1.0 + 10 * 2.5 = 35.0
    # score = 100 * (1 - 7.0 / 35.0) = 100 * (1 - 0.20) = 80.00
    assert m.alignment_score == 80.00


def test_alignment_iupac_extended_motifs():
    """
    Requirement 8: IUPAC PAM matching for diverse ambiguous patterns:
    - SpCas9 alternative NAG: matches AAG, CAG, GAG, TAG; rejects AGG, ACG.
    - Extended SpCas9 NRG (R = A or G): matches AAG (R=A), AGG (R=G); rejects ACG (C is not R).
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"

    # Reference with targets followed by AAG, AGG, ACG
    ref = (
        ("T" * 20) + (guide + "AAG") +  # Match 1 (NAG: yes, NRG: yes)
        ("T" * 20) + (guide + "AGG") +  # Match 2 (NAG: no,  NRG: yes)
        ("T" * 20) + (guide + "ACG") +  # Match 3 (NAG: no,  NRG: no)
        ("T" * 20)
    )

    engine = CPUEngine()

    # Test NAG PAM (only AAG matches, AGG and ACG do not end with AG)
    res_nag = engine.align(ref, "nag_test", guide, pam_pattern="NAG", max_mismatches=0)
    assert len(res_nag.matches) == 1
    assert res_nag.matches[0].pam_sequence == "AAG"
    assert res_nag.matches[0].start_pos == 21

    # Test NRG PAM (both AAG and AGG match because R = A or G; ACG is rejected)
    res_nrg = engine.align(ref, "nrg_test", guide, pam_pattern="NRG", max_mismatches=0)
    assert len(res_nrg.matches) == 2
    assert res_nrg.matches[0].pam_sequence == "AAG"
    assert res_nrg.matches[0].start_pos == 21
    assert res_nrg.matches[1].pam_sequence == "AGG"
    assert res_nrg.matches[1].start_pos == 64


def test_alignment_duplicate_result_prevention_on_repeats():
    """
    Requirement 12: Duplicate result prevention.
    Verifies that identical target sequences occurring at multiple loci in a
    repetitive genome are reported at their distinct respective coordinates
    without redundant duplication of the same locus.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"
    target_window = guide + "GGG"  # 23 bp

    # 3 identical tandem copies separated by 10 bp spacers
    spacer = "A" * 10
    ref = spacer + target_window + spacer + target_window + spacer + target_window + spacer

    engine = CPUEngine()
    res = engine.align(ref, "repeat_array", guide, pam_pattern="NGG", max_mismatches=0)

    # Exactly 3 distinct matches
    assert len(res.matches) == 3

    # Offsets:
    # Copy 1: 10 spacer -> 1-based start 11, end 33
    # Copy 2: 10 + 23 + 10 = 43 -> 1-based start 44, end 66
    # Copy 3: 43 + 23 + 10 = 76 -> 1-based start 77, end 99
    coords = [(m.start_pos, m.end_pos) for m in res.matches]
    assert coords == [(11, 33), (44, 66), (77, 99)]

    # All start coordinates are strictly unique (no duplicate entries)
    start_positions = [m.start_pos for m in res.matches]
    assert len(start_positions) == len(set(start_positions))

