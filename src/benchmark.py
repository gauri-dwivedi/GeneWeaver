"""
GeneWeaver: Performance Benchmarking Suite
==========================================
Measures, compares, and profiles sequence alignment throughput between
CPU (Python sliding window) and GPU (PyTorch vectorized tensor operations).

Scientific & Benchmarking Principles:
-------------------------------------
1. Warmup Execution:
   The first PyTorch kernel or CUDA call includes one-time overhead (driver initialization,
   kernel compilation, and memory allocation). GeneWeaver runs a dedicated warmup pass
   to eliminate cold-start bias.
2. Multiple Iterations & Median Measurement:
   To prevent operating system jitter and background task spikes, each measurement
   is repeated across multiple iterations and summarized.
3. GPU Overhead Analysis:
   - For short DNA sequences (<1,000 bp), the time taken to allocate tensors and launch kernels
     often exceeds the computational runtime. Thus, the CPU engine can appear faster.
   - For larger sequences (>10,000 bp), the massive parallelism of tensor operations dominates,
     leading to significant throughput advantages.
4. Strict Honesty:
   Benchmark values are measured in real time. GeneWeaver NEVER generates fake speedup numbers.
"""

import time
import os
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

from src.cpu_engine import CPUEngine
from src.gpu_engine import GPUEngine
from src.sequence_utils import clean_sequence


@dataclass
class BenchmarkMetric:
    """
    Performance measurement data for a specific dataset size.
    """
    sequence_length: int
    windows_evaluated: int
    cpu_time_ms: float
    gpu_time_ms: float
    cpu_throughput_wps: float
    gpu_throughput_wps: float
    speedup_ratio: float
    active_device: str
    is_cuda_accelerated: bool
    notes: str


class BenchmarkSuite:
    """
    Automated benchmarking suite comparing CPU and GPU alignment performance.
    """

    def __init__(self):
        self.cpu_engine = CPUEngine()
        self.gpu_engine = GPUEngine()

    def run_single_comparison(
        self,
        reference_seq: str,
        guide_seq: str,
        pam_pattern: str = "NGG",
        max_mismatches: int = 3,
        scan_both_strands: bool = True,
        warmup: bool = True,
        iterations: int = 3,
    ) -> BenchmarkMetric:
        """
        Run a rigorously timed comparison on a specific sequence and guide.

        Parameters:
            reference_seq (str): Reference DNA sequence.
            guide_seq (str): Guide sequence.
            pam_pattern (str): PAM pattern.
            max_mismatches (int): Max mismatch threshold.
            scan_both_strands (bool): Whether to evaluate both DNA strands.
            warmup (bool): Run warmup iteration before timing.
            iterations (int): Number of timing repetitions (median taken).

        Returns:
            BenchmarkMetric: Performance metrics.
        """
        clean_ref = clean_sequence(reference_seq)
        clean_guide = clean_sequence(guide_seq)

        # Warmup pass
        if warmup:
            self.cpu_engine.align(clean_ref[:100], "warmup", clean_guide, pam_pattern, max_mismatches, False)
            self.gpu_engine.align(clean_ref[:100], "warmup", clean_guide, pam_pattern, max_mismatches, False)

        cpu_times_ms: List[float] = []
        gpu_times_ms: List[float] = []

        total_windows = 0

        for _ in range(iterations):
            cpu_res = self.cpu_engine.align(
                clean_ref, "bench_ref", clean_guide, pam_pattern, max_mismatches, scan_both_strands
            )
            cpu_times_ms.append(cpu_res.execution_time_ms)
            total_windows = cpu_res.total_windows_evaluated

            gpu_res = self.gpu_engine.align(
                clean_ref, "bench_ref", clean_guide, pam_pattern, max_mismatches, scan_both_strands
            )
            gpu_times_ms.append(gpu_res.execution_time_ms)

        median_cpu_ms = float(np.median(cpu_times_ms))
        median_gpu_ms = float(np.median(gpu_times_ms))

        # Avoid division by zero
        safe_cpu_s = max(median_cpu_ms / 1000.0, 1e-7)
        safe_gpu_s = max(median_gpu_ms / 1000.0, 1e-7)

        cpu_wps = round(total_windows / safe_cpu_s, 2)
        gpu_wps = round(total_windows / safe_gpu_s, 2)

        speedup = round(safe_cpu_s / safe_gpu_s, 2)

        # Student diagnostic explanation
        if median_gpu_ms > median_cpu_ms:
            if not self.gpu_engine.cuda_available:
                notes = (
                    "Running in CPU PyTorch Fallback mode. Vector tensor setup and memory allocation "
                    "overhead on small inputs can be higher than pure Python loop execution."
                )
            else:
                notes = (
                    "GPU memory transfer and kernel launch overhead exceeds compute time for small input size. "
                    "Expected behavior for small sequence lengths."
                )
        else:
            if self.gpu_engine.cuda_available:
                notes = f"CUDA GPU acceleration achieved {speedup}x speedup via parallel matrix unfolding."
            else:
                notes = f"Vectorized tensor batching on CPU achieved {speedup}x speedup over iterative loop."

        return BenchmarkMetric(
            sequence_length=len(clean_ref),
            windows_evaluated=total_windows,
            cpu_time_ms=round(median_cpu_ms, 3),
            gpu_time_ms=round(median_gpu_ms, 3),
            cpu_throughput_wps=cpu_wps,
            gpu_throughput_wps=gpu_wps,
            speedup_ratio=speedup,
            active_device=self.gpu_engine.device_name,
            is_cuda_accelerated=self.gpu_engine.is_hardware_accelerated,
            notes=notes,
        )

    def run_scaling_benchmark(
        self,
        lengths: Optional[List[int]] = None,
        guide_seq: str = "GAGTCCGAGCAGAAGAAGAA",
        pam_pattern: str = "NGG",
        max_mismatches: int = 3,
        iterations: int = 2,
    ) -> pd.DataFrame:
        """
        Execute an automated scaling benchmark across multiple sequence lengths.

        Parameters:
            lengths (List[int]): List of sequence lengths to test.
            guide_seq (str): Guide sequence.
            pam_pattern (str): PAM pattern.
            max_mismatches (int): Max mismatch threshold.
            iterations (int): Timing iterations per length.

        Returns:
            pd.DataFrame: Table of benchmark results.
        """
        if lengths is None:
            lengths = [500, 2000, 5000, 15000]

        # Seed reproducible synthetic sequence
        rng = np.random.default_rng(42)
        bases = np.array(["A", "C", "G", "T"])

        rows = []
        for seq_len in lengths:
            # Generate random DNA of exact length
            random_bases = rng.choice(bases, size=seq_len)
            test_seq = "".join(random_bases)

            # Insert known target to guarantee non-empty candidate matches
            insert_pos = seq_len // 2
            test_seq = test_seq[:insert_pos] + guide_seq + "GGG" + test_seq[insert_pos + 23:]

            metric = self.run_single_comparison(
                reference_seq=test_seq,
                guide_seq=guide_seq,
                pam_pattern=pam_pattern,
                max_mismatches=max_mismatches,
                scan_both_strands=True,
                warmup=True,
                iterations=iterations,
            )

            rows.append({
                "Sequence Length (bp)": metric.sequence_length,
                "Windows Evaluated": metric.windows_evaluated,
                "CPU Time (ms)": metric.cpu_time_ms,
                "GPU Engine Time (ms)": metric.gpu_time_ms,
                "CPU Throughput (win/s)": metric.cpu_throughput_wps,
                "GPU Engine Throughput (win/s)": metric.gpu_throughput_wps,
                "Speedup (CPU / GPU)": metric.speedup_ratio,
                "Active Device": metric.active_device,
                "CUDA Enabled": metric.is_cuda_accelerated,
                "Notes": metric.notes,
            })

        return pd.DataFrame(rows)


def export_benchmark_to_csv(df: pd.DataFrame, custom_path: Optional[str] = None) -> Path:
    """
    Save benchmark DataFrame to CSV in the results/ directory.

    Parameters:
        df (pd.DataFrame): Benchmark dataframe.
        custom_path (Optional[str]): Custom destination path.

    Returns:
        Path: Path to the generated CSV file.
    """
    if custom_path:
        target_path = Path(custom_path)
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = results_dir / f"benchmark_run_{timestamp}.csv"

    df.to_csv(target_path, index=False)
    return target_path
