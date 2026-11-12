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
    """Verify single benchmark runs without error and calculates metrics."""
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
    assert metric.speedup_ratio > 0
    assert isinstance(metric.notes, str)


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

