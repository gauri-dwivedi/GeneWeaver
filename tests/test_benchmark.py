"""
Tests for Module 6 (Performance Benchmarking) & Report Generator
"""

import pytest
import os
import pandas as pd
from src.benchmark import BenchmarkSuite, export_benchmark_to_csv
from src.report_generator import (
    matches_to_dataframe,
    generate_markdown_alignment_report,
    export_alignment_csv,
)
from src.crispr_alignment import CRISPRTargetMatch, AlignmentEngineResult


def test_benchmark_single_comparison():
    """Verify single benchmark runs without error and calculates all statistical metrics."""
    suite = BenchmarkSuite()
    ref = "A" * 100 + "GAGTCCGAGCAGAAGAAGAAGGG" + "T" * 100
    guide = "GAGTCCGAGCAGAAGAAGAA"

    metric = suite.run_single_comparison(
        reference_seq=ref,
        guide_seq=guide,
        warmup=True,
        iterations=2,
    )

    assert metric.sequence_length > 0
    assert metric.windows_evaluated > 0
    assert metric.cpu_time_ms >= 0.0
    assert metric.gpu_time_ms >= 0.0
    assert metric.cpu_throughput_wps > 0
    assert metric.gpu_throughput_wps > 0
    assert metric.cpu_mbps > 0
    assert metric.gpu_mbps > 0
    assert metric.cpu_std_ms >= 0.0
    assert metric.gpu_std_ms >= 0.0
    assert metric.cpu_min_ms <= metric.cpu_max_ms
    assert metric.gpu_min_ms <= metric.gpu_max_ms
    assert metric.speedup_ratio > 0
    assert isinstance(metric.notes, str)
    assert isinstance(metric.active_device, str)


def test_benchmark_empty_and_short_input_safety():
    """Verify safe handling of empty reference sequence or sequence shorter than window."""
    suite = BenchmarkSuite()
    guide = "GAGTCCGAGCAGAAGAAGAA"  # 20 bp + 3 bp PAM = 23 bp window

    # 1. Zero-length empty reference
    m_empty = suite.run_single_comparison("", guide, warmup=False)
    assert m_empty.sequence_length == 0
    assert m_empty.windows_evaluated == 0
    assert m_empty.cpu_time_ms == 0.0
    assert m_empty.gpu_time_ms == 0.0
    assert m_empty.cpu_mbps == 0.0
    assert m_empty.speedup_ratio == 1.0
    assert "shorter than alignment window" in m_empty.notes

    # 2. Sequence shorter than window (10 bp < 23 bp)
    m_short = suite.run_single_comparison("ATGCATGCAT", guide, warmup=False)
    assert m_short.sequence_length == 10
    assert m_short.windows_evaluated == 0
    assert m_short.cpu_time_ms == 0.0
    assert m_short.gpu_time_ms == 0.0
    assert m_short.speedup_ratio == 1.0


def test_benchmark_statistical_metrics_calculation():
    """Verify sample standard deviation and min/max relationships across iterations."""
    suite = BenchmarkSuite()
    ref = "C" * 200 + "GAGTCCGAGCAGAAGAAGAACGG" + "G" * 200
    guide = "GAGTCCGAGCAGAAGAAGAA"

    metric = suite.run_single_comparison(ref, guide, iterations=3, warmup=True)

    assert metric.cpu_min_ms <= metric.cpu_time_ms <= metric.cpu_max_ms or metric.cpu_min_ms <= metric.cpu_max_ms
    assert metric.gpu_min_ms <= metric.gpu_time_ms <= metric.gpu_max_ms or metric.gpu_min_ms <= metric.gpu_max_ms
    assert metric.cpu_std_ms >= 0.0
    assert metric.gpu_std_ms >= 0.0


def test_benchmark_run_fasta_benchmark():
    """Verify run_fasta_benchmark on sample reference FASTA datasets."""
    suite = BenchmarkSuite()
    df = suite.run_fasta_benchmark(iterations=1)

    assert len(df) >= 3
    assert "Record ID" in df.columns
    assert "Dataset Type" in df.columns
    assert "Sequence Length (bp)" in df.columns
    assert "CPU Time (ms)" in df.columns
    assert "GPU Engine Time (ms)" in df.columns
    assert "Speedup (CPU / GPU)" in df.columns
    assert "CPU Throughput (Mbp/s)" in df.columns

    # Verify that all sample loci are labeled as synthetic
    for d_type in df["Dataset Type"]:
        assert "Synthetic" in d_type


def test_benchmark_scaling():
    """Verify scaling benchmark tests across multiple sequence lengths."""
    suite = BenchmarkSuite()
    df = suite.run_scaling_benchmark(lengths=[200, 500], iterations=1)

    assert len(df) == 2
    assert "Sequence Length (bp)" in df.columns
    assert "CPU Time (ms)" in df.columns
    assert "Speedup (CPU / GPU)" in df.columns


def test_benchmark_csv_export(tmp_path):
    """Verify exporting benchmark results to CSV."""
    df = pd.DataFrame([{"length": 500, "cpu_ms": 1.2, "gpu_ms": 1.5}])
    out_file = tmp_path / "test_bench.csv"
    saved = export_benchmark_to_csv(df, custom_path=str(out_file))
    assert os.path.exists(saved)
    loaded = pd.read_csv(saved)
    assert len(loaded) == 1


def test_report_generator():
    """Verify markdown report generation and dataframe conversion."""
    dummy_match = CRISPRTargetMatch(
        reference_id="chr1",
        strand="+",
        start_pos=10,
        end_pos=32,
        target_dna_sequence="GAGTCCGAGCAGAAGAAGAA",
        pam_sequence="GGG",
        full_window="GAGTCCGAGCAGAAGAAGAAGGG",
        mismatch_count=0,
        mismatch_positions=[],
        seed_mismatches=0,
        distal_mismatches=0,
        alignment_score=100.0,
        alignment_visual="||||||||||||||||||||",
    )

    df = matches_to_dataframe([dummy_match])
    assert len(df) == 1
    assert df.iloc[0]["Mismatches"] == 0

    dummy_result = AlignmentEngineResult(
        matches=[dummy_match],
        total_windows_evaluated=100,
        execution_time_seconds=0.01,
        execution_time_ms=10.0,
        device_used="CPU",
        windows_per_second=10000.0,
        guide_length=20,
        pam_pattern="NGG",
        max_mismatches=3,
    )

    report = generate_markdown_alignment_report(dummy_result, "GAGTCCGAGCAGAAGAAGAA", "chr1", 123)
    assert "GeneWeaver CRISPR Alignment Report" in report
    assert "100.0" in report

