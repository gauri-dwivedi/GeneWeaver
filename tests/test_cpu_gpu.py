"""
Tests for Module 5 (GPU Acceleration & Fallback Engine)
Verifies equivalence between CPU and GPU alignment engines.
"""

import pytest
import torch
from src.cpu_engine import CPUEngine
from src.gpu_engine import GPUEngine
from src.input_handler import load_sample_reference_fasta


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

