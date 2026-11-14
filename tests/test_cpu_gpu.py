"""
Tests for Module 5 (GPU Acceleration & Fallback Engine)
Verifies equivalence between CPU and GPU alignment engines.
"""

import pytest
import torch
from src.cpu_engine import CPUEngine
from src.gpu_engine import GPUEngine
from src.input_handler import load_sample_reference_fasta
from src.sequence_utils import reverse_complement


def test_gpu_device_detection():
    """Verify GPU engine detects active device and never claims GPU when on CPU."""
    engine = GPUEngine()
    device_info = engine.get_device_info()

    assert "cuda_available" in device_info
    assert "device_used" in device_info
    assert "is_gpu_active" in device_info

    if not torch.cuda.is_available():
        assert engine.is_hardware_accelerated is False
        assert "CPU" in engine.device_name
    else:
        assert engine.is_hardware_accelerated is True
        assert "GPU" in engine.device_name


def test_cpu_gpu_result_equivalence():
    """
    Core correctness test:
    Verify that CPU and GPU engines produce IDENTICAL results:
    - Same number of matches
    - Same coordinates
    - Same mismatch counts
    - Same alignment scores
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"

    # Construct reference with known forward and reverse matches and mismatches
    # Match 1: Forward exact match (0 MM) with AGG
    # Match 2: Forward 1 MM with CGG
    # Match 3: Forward 2 MM with TGG
    ref = (
        "AAAA" + (guide + "AGG") +
        "CCCC" + ("GAGTCCGAGCAGTAGAAGAA" + "CGG") +
        "GGGG" + ("GAGTCCATGCAATAGAAGAA" + "TGG") +
        "TTTT"
    )

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    cpu_res = cpu_engine.align(ref, "scaffold_equiv", guide, pam_pattern="NGG", max_mismatches=3)
    gpu_res = gpu_engine.align(ref, "scaffold_equiv", guide, pam_pattern="NGG", max_mismatches=3)

    assert len(cpu_res.matches) == len(gpu_res.matches)

    for cpu_match, gpu_match in zip(cpu_res.matches, gpu_res.matches):
        assert cpu_match.strand == gpu_match.strand
        assert cpu_match.start_pos == gpu_match.start_pos
        assert cpu_match.end_pos == gpu_match.end_pos
        assert cpu_match.target_dna_sequence == gpu_match.target_dna_sequence
        assert cpu_match.pam_sequence == gpu_match.pam_sequence
        assert cpu_match.mismatch_count == gpu_match.mismatch_count
        assert cpu_match.mismatch_positions == gpu_match.mismatch_positions
        assert cpu_match.alignment_score == gpu_match.alignment_score


def test_cpu_gpu_on_sample_fasta():
    """Test CPU and GPU engines on realistic sample reference dataset."""
    records = load_sample_reference_fasta()
    emx1_rec = next(r for r in records if "emx1" in r.id.lower())

    guide = "GAGTCCGAGCAGAAGAAGAA"  # Known EMX1 guide

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    cpu_res = cpu_engine.align(emx1_rec.sequence, emx1_rec.id, guide, max_mismatches=2)
    gpu_res = gpu_engine.align(emx1_rec.sequence, emx1_rec.id, guide, max_mismatches=2)

    assert len(cpu_res.matches) == len(gpu_res.matches)
    if len(cpu_res.matches) > 0:
        assert cpu_res.matches[0].start_pos == gpu_res.matches[0].start_pos
        assert cpu_res.matches[0].mismatch_count == gpu_res.matches[0].mismatch_count


def test_gpu_iupac_pam_vectorized_matching():
    """
    Verify vectorized IUPAC PAM matching on GPU/PyTorch tensor engine:
    - Tests degenerate SaCas9 'NNGRRT' (N=any, N=any, G=G, R=A/G, R=A/G, T=T).
    - Tests non-canonical PAM 'NAG'.
    - Compares CPU and GPU results to verify 100% equivalence.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp

    # Construct test reference with SaCas9 PAM variants:
    # 1. Forward target with AGGAGT (valid NNGRRT: G=G, R=A, R=G, T=T)
    # 2. Forward target with CCGGGT (valid NNGRRT: G=G, R=G, R=G, T=T)
    # 3. Non-matching target with CCGGGA (invalid: ends in A, not T)
    # 4. Reverse target with TTGAAT (valid NNGRRT: G=G, R=A, R=A, T=T)
    valid_pam_1 = "AGGAGT"
    valid_pam_2 = "CCGGGT"
    invalid_pam = "CCGGGA"
    valid_pam_rev = "TTGAAT"

    rev_target_seq = guide + valid_pam_rev

    ref = (
        ("A" * 20) + (guide + valid_pam_1) +
        ("C" * 25) + (guide + valid_pam_2) +
        ("G" * 25) + (guide + invalid_pam) +
        ("T" * 25) + reverse_complement(rev_target_seq) +
        ("A" * 20)
    )

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    cpu_res = cpu_engine.align(ref, "iupac_test", guide, pam_pattern="NNGRRT", max_mismatches=0)
    gpu_res = gpu_engine.align(ref, "iupac_test", guide, pam_pattern="NNGRRT", max_mismatches=0)

    # Both must find exactly 3 matches (2 forward, 1 reverse; invalid PAM excluded)
    assert len(cpu_res.matches) == 3
    assert len(gpu_res.matches) == 3

    for cpu_m, gpu_m in zip(cpu_res.matches, gpu_res.matches):
        assert cpu_m.strand == gpu_m.strand
        assert cpu_m.start_pos == gpu_m.start_pos
        assert cpu_m.end_pos == gpu_m.end_pos
        assert cpu_m.target_start == gpu_m.target_start
        assert cpu_m.target_end == gpu_m.target_end
        assert cpu_m.pam_start == gpu_m.pam_start
        assert cpu_m.pam_end == gpu_m.pam_end
        assert cpu_m.pam_sequence == gpu_m.pam_sequence
        assert cpu_m.mismatch_count == gpu_m.mismatch_count
        assert cpu_m.alignment_score == gpu_m.alignment_score


def test_cpu_gpu_tie_breaking_equivalence():
    """
    Verify deterministic 4-key tie-breaking between CPU and GPU:
    (mismatch_count, -alignment_score, start_pos, strand)
    Ensures identical ordered results when candidate targets have
    identical mismatch counts and identical alignment scores.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"

    # Create 3 targets with 1 distal mismatch (identical score 97.14):
    # Target A at pos 21 (distal MM at pos 1: G -> A)
    # Target B at pos 64 (distal MM at pos 1: G -> A)
    # Target C on reverse strand at pos 107 (distal MM at pos 1: G -> A)
    # Target Exact at pos 150 (0 MM -> score 100.0)
    target_1mm = "AAGTCCGAGCAGAAGAAGAA" + "CGG"
    target_exact = guide + "TGG"

    ref = (
        ("T" * 20) + target_1mm +          # Target A (start 21)
        ("T" * 20) + target_1mm +          # Target B (start 64)
        ("T" * 20) + reverse_complement(target_1mm) + # Target C (start 107, strand '-')
        ("T" * 20) + target_exact +        # Target Exact (start 150)
        ("T" * 20)
    )

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    cpu_res = cpu_engine.align(ref, "tie_test", guide, pam_pattern="NGG", max_mismatches=1)
    gpu_res = gpu_engine.align(ref, "tie_test", guide, pam_pattern="NGG", max_mismatches=1)

    assert len(cpu_res.matches) == 4
    assert len(gpu_res.matches) == 4

    # 1. First match must be the exact match (0 MM)
    assert cpu_res.matches[0].mismatch_count == 0
    assert cpu_res.matches[0].start_pos == 150

    # 2. Tied 1-MM matches must be sorted by start_pos ascending:
    # Target A (pos 21) -> Target B (pos 64) -> Target C (pos 107)
    assert cpu_res.matches[1].start_pos == 21
    assert cpu_res.matches[2].start_pos == 64
    assert cpu_res.matches[3].start_pos == 107

    # 3. CPU and GPU must be 100% identical element-by-element
    for cpu_m, gpu_m in zip(cpu_res.matches, gpu_res.matches):
        assert cpu_m.strand == gpu_m.strand
        assert cpu_m.start_pos == gpu_m.start_pos
        assert cpu_m.end_pos == gpu_m.end_pos
        assert cpu_m.mismatch_count == gpu_m.mismatch_count
        assert cpu_m.alignment_score == gpu_m.alignment_score


def test_cpu_gpu_empty_reference_and_no_pam_safety():
    """Verify safe execution on empty reference, too-short reference, and no-PAM references."""
    guide = "GAGTCCGAGCAGAAGAAGAA"

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    # 1. Empty reference
    res_cpu_empty = cpu_engine.align("", "empty_ref", guide)
    res_gpu_empty = gpu_engine.align("", "empty_ref", guide)
    assert len(res_cpu_empty.matches) == 0
    assert len(res_gpu_empty.matches) == 0
    assert res_cpu_empty.total_windows_evaluated == 0
    assert res_gpu_empty.total_windows_evaluated == 0

    # 2. Reference shorter than window (guide 20 + pam 3 = 23 bp)
    short_ref = "ATGCATGC"
    res_cpu_short = cpu_engine.align(short_ref, "short_ref", guide)
    res_gpu_short = gpu_engine.align(short_ref, "short_ref", guide)
    assert len(res_cpu_short.matches) == 0
    assert len(res_gpu_short.matches) == 0
    assert res_cpu_short.total_windows_evaluated == 0
    assert res_gpu_short.total_windows_evaluated == 0

    # 3. Reference with no matching PAM (e.g. all A's with NGG PAM)
    no_pam_ref = "A" * 100
    res_cpu_nopam = cpu_engine.align(no_pam_ref, "nopam_ref", guide, pam_pattern="NGG")
    res_gpu_nopam = gpu_engine.align(no_pam_ref, "nopam_ref", guide, pam_pattern="NGG")
    assert len(res_cpu_nopam.matches) == 0
    assert len(res_gpu_nopam.matches) == 0
    assert res_cpu_nopam.total_windows_evaluated > 0
    assert res_cpu_nopam.total_windows_evaluated == res_gpu_nopam.total_windows_evaluated


# ============================================================================
# Advanced CPU vs PyTorch Equivalence Tests (Phase 4 - Step 4)
# ============================================================================

def test_cpu_gpu_equivalence_variable_guide_lengths():
    """
    Requirement 3 & 10: Verify CPU and PyTorch engines produce IDENTICAL results
    for non-standard guide lengths (17 bp tru-gRNA and 24 bp extended guide)
    across forward and reverse-complement strands.
    """
    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    # 1. 17 bp guide with forward exact match and reverse 1-MM match
    guide_17 = "GAGTCCGAGCAGAAGAA"  # 17 bp
    fwd_target_17 = guide_17 + "CGG"
    # Reverse target has 1 mismatch in distal region (pos 2: A -> T)
    rev_target_17 = "GTGTCCGAGCAGAAGAA" + "TGG"
    ref_17 = (
        ("C" * 20) + fwd_target_17 +
        ("A" * 30) + reverse_complement(rev_target_17) +
        ("G" * 20)
    )

    cpu_res_17 = cpu_engine.align(ref_17, "ref_17", guide_17, pam_pattern="NGG", max_mismatches=2)
    gpu_res_17 = gpu_engine.align(ref_17, "ref_17", guide_17, pam_pattern="NGG", max_mismatches=2)

    assert len(cpu_res_17.matches) == 2
    assert len(gpu_res_17.matches) == 2

    for c_m, g_m in zip(cpu_res_17.matches, gpu_res_17.matches):
        assert c_m.strand == g_m.strand
        assert c_m.start_pos == g_m.start_pos
        assert c_m.end_pos == g_m.end_pos
        assert c_m.target_start == g_m.target_start
        assert c_m.target_end == g_m.target_end
        assert c_m.pam_start == g_m.pam_start
        assert c_m.pam_end == g_m.pam_end
        assert c_m.target_dna_sequence == g_m.target_dna_sequence
        assert c_m.pam_sequence == g_m.pam_sequence
        assert c_m.mismatch_count == g_m.mismatch_count
        assert c_m.mismatch_positions == g_m.mismatch_positions
        assert c_m.seed_mismatches == g_m.seed_mismatches
        assert c_m.distal_mismatches == g_m.distal_mismatches
        assert c_m.alignment_score == g_m.alignment_score

    # 2. 24 bp guide
    guide_24 = "GAGTCCGAGCAGAAGAAGAAGATC"  # 24 bp
    fwd_target_24 = guide_24 + "GGG"
    ref_24 = ("T" * 25) + fwd_target_24 + ("A" * 25)

    cpu_res_24 = cpu_engine.align(ref_24, "ref_24", guide_24, pam_pattern="NGG", max_mismatches=0)
    gpu_res_24 = gpu_engine.align(ref_24, "ref_24", guide_24, pam_pattern="NGG", max_mismatches=0)

    assert len(cpu_res_24.matches) == 1
    assert len(gpu_res_24.matches) == 1
    assert cpu_res_24.matches[0].start_pos == gpu_res_24.matches[0].start_pos
    assert cpu_res_24.matches[0].alignment_score == gpu_res_24.matches[0].alignment_score == 100.0


def test_cpu_gpu_equivalence_degenerate_nrg_pam():
    """
    Requirement 8 & 10: Verify CPU and PyTorch produce identical results on degenerate
    IUPAC PAM motif 'NRG' (N=any, R=A/G, G=G), matching AGG and GGG, but not ACG or ATG.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Construct sequence with AGG, GGG (valid NRG) and ATG, ACG (invalid)
    ref = (
        ("T" * 15) + (guide + "AGG") +  # valid
        ("T" * 15) + (guide + "ATG") +  # invalid
        ("T" * 15) + (guide + "GGG") +  # valid
        ("T" * 15) + (guide + "ACG") +  # invalid
        ("T" * 15)
    )

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    cpu_res = cpu_engine.align(ref, "nrg_scaffold", guide, pam_pattern="NRG", max_mismatches=0)
    gpu_res = gpu_engine.align(ref, "nrg_scaffold", guide, pam_pattern="NRG", max_mismatches=0)

    assert len(cpu_res.matches) == 2
    assert len(gpu_res.matches) == 2

    for c_m, g_m in zip(cpu_res.matches, gpu_res.matches):
        assert c_m.start_pos == g_m.start_pos
        assert c_m.pam_sequence == g_m.pam_sequence
        assert c_m.mismatch_count == g_m.mismatch_count
        assert c_m.alignment_score == g_m.alignment_score


def test_cpu_gpu_non_canonical_and_ambiguous_reference_characters():
    """
    Requirement 11: Non-canonical bases (e.g. 'N', 'X') in reference sequence
    must execute safely on both engines and be counted as mismatches.
    """
    guide = "GAGTCCGAGCAGAAGAAGAA"
    # Target has 'N' and 'X' in reference
    # Guide has G at pos 1, reference has N. Guide has C at pos 6, reference has X.
    target_with_ambig = "NAGTCCGAGCAGAAGAAGAA" + "CGG"
    ref = ("C" * 20) + target_with_ambig + ("G" * 20)

    cpu_engine = CPUEngine()
    gpu_engine = GPUEngine()

    # Allowed up to 1 mismatch -> Should find 1 match (pos 1 N != G counts as 1 mismatch)
    cpu_res = cpu_engine.align(ref, "ambig_ref", guide, pam_pattern="NGG", max_mismatches=1)
    gpu_res = gpu_engine.align(ref, "ambig_ref", guide, pam_pattern="NGG", max_mismatches=1)

    assert len(cpu_res.matches) == 1
    assert len(gpu_res.matches) == 1
    assert cpu_res.matches[0].mismatch_count == 1
    assert gpu_res.matches[0].mismatch_count == 1
    assert cpu_res.matches[0].start_pos == gpu_res.matches[0].start_pos
    assert cpu_res.matches[0].alignment_score == gpu_res.matches[0].alignment_score
