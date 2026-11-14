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

    Attributes:
        sequence_length (int): Total base pairs in evaluated reference sequence.
        windows_evaluated (int): Total candidate sliding windows scanned across both strands.
        cpu_time_ms (float): Median execution latency for CPU reference engine in ms.
        gpu_time_ms (float): Median execution latency for PyTorch tensor engine in ms.
        cpu_throughput_wps (float): Scanning throughput for CPU in windows/second.
        gpu_throughput_wps (float): Scanning throughput for PyTorch in windows/second.
        cpu_mbps (float): Processing speed in Megabases/second (Mbp/s) for CPU.
        gpu_mbps (float): Processing speed in Megabases/second (Mbp/s) for PyTorch.
        cpu_std_ms (float): Sample standard deviation (ddof=1) of CPU latency across iterations.
        gpu_std_ms (float): Sample standard deviation (ddof=1) of PyTorch latency across iterations.
        cpu_min_ms (float): Minimum measured CPU latency across iterations in ms.
        cpu_max_ms (float): Maximum measured CPU latency across iterations in ms.
        gpu_min_ms (float): Minimum measured PyTorch latency across iterations in ms.
        gpu_max_ms (float): Maximum measured PyTorch latency across iterations in ms.
        speedup_ratio (float): Relative speedup (median_cpu_ms / median_gpu_ms).
        active_device (str): Name of active compute device (CPU fallback or CUDA GPU).
        is_cuda_accelerated (bool): True if executed on physical NVIDIA CUDA GPU hardware.
        notes (str): Detailed engineering and diagnostic explanation.
    """
    sequence_length: int
    windows_evaluated: int
    cpu_time_ms: float
    gpu_time_ms: float
    cpu_throughput_wps: float
    gpu_throughput_wps: float
    cpu_mbps: float
    gpu_mbps: float
    cpu_std_ms: float
    gpu_std_ms: float
    cpu_min_ms: float
    cpu_max_ms: float
    gpu_min_ms: float
    gpu_max_ms: float
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

        Statistical & Measurement Methodology:
        --------------------------------------
        1. Clean and normalize inputs prior to starting timers (excludes data prep overhead).
        2. If warmup is enabled, an untimed pass of the target sequence is executed across
           both engines to initialize JIT bytecode, allocate tensor pools, and prime drivers.
        3. Multiple repeated runs (default 3) are timed via high-resolution monotonic timer
           time.perf_counter() (with torch.cuda.synchronize() if running on CUDA).
        4. Central tendency is represented by the median to resist operating system context-switching noise.
        5. Variability is measured via sample standard deviation (ddof=1).
        6. Distinguishes CPU reference, PyTorch CPU fallback, and true CUDA GPU acceleration.

        Parameters:
            reference_seq (str): Reference DNA sequence.
            guide_seq (str): Guide sequence.
            pam_pattern (str): PAM pattern.
            max_mismatches (int): Max mismatch threshold.
            scan_both_strands (bool): Whether to evaluate both DNA strands.
            warmup (bool): Run warmup iteration before timing.
            iterations (int): Number of timing repetitions (median and sample std dev taken).

        Returns:
            BenchmarkMetric: Detailed performance metrics and statistical variability.
        """
        clean_ref = clean_sequence(reference_seq)
        clean_guide = clean_sequence(guide_seq)
        clean_pam = clean_sequence(pam_pattern)

        ref_len = len(clean_ref)
        guide_len = len(clean_guide)
        window_len = guide_len + len(clean_pam)

        # Defensive guard for empty or too-short inputs
        if ref_len < window_len or ref_len == 0:
            return BenchmarkMetric(
                sequence_length=ref_len,
                windows_evaluated=0,
                cpu_time_ms=0.0,
                gpu_time_ms=0.0,
                cpu_throughput_wps=0.0,
                gpu_throughput_wps=0.0,
                cpu_mbps=0.0,
                gpu_mbps=0.0,
                cpu_std_ms=0.0,
                gpu_std_ms=0.0,
                cpu_min_ms=0.0,
                cpu_max_ms=0.0,
                gpu_min_ms=0.0,
                gpu_max_ms=0.0,
                speedup_ratio=1.0,
                active_device=self.gpu_engine.device_name,
                is_cuda_accelerated=self.gpu_engine.is_hardware_accelerated,
                notes="Sequence length is shorter than alignment window (L < W). Zero windows evaluated.",
            )

        # Untimed warmup pass on full sequence length to initialize
        # PyTorch memory pools and CUDA driver initialization
        if warmup:
            self.cpu_engine.align(clean_ref, "warmup_ref", clean_guide, clean_pam, max_mismatches, scan_both_strands)
            self.gpu_engine.align(clean_ref, "warmup_ref", clean_guide, clean_pam, max_mismatches, scan_both_strands)

        cpu_times_ms: List[float] = []
        gpu_times_ms: List[float] = []
        total_windows = 0

        actual_iterations = max(1, iterations)
        for _ in range(actual_iterations):
            cpu_res = self.cpu_engine.align(
                clean_ref, "bench_ref", clean_guide, clean_pam, max_mismatches, scan_both_strands
            )
            cpu_times_ms.append(cpu_res.execution_time_ms)
            total_windows = cpu_res.total_windows_evaluated

            gpu_res = self.gpu_engine.align(
                clean_ref, "bench_ref", clean_guide, clean_pam, max_mismatches, scan_both_strands
            )
            gpu_times_ms.append(gpu_res.execution_time_ms)

        # Statistical metrics
        median_cpu_ms = float(np.median(cpu_times_ms))
        median_gpu_ms = float(np.median(gpu_times_ms))

        cpu_min_ms = float(np.min(cpu_times_ms))
        cpu_max_ms = float(np.max(cpu_times_ms))
        gpu_min_ms = float(np.min(gpu_times_ms))
        gpu_max_ms = float(np.max(gpu_times_ms))

        # Sample standard deviation (Bessel's correction: ddof=1) across repetitions
        cpu_std_ms = float(np.std(cpu_times_ms, ddof=1)) if len(cpu_times_ms) > 1 else 0.0
        gpu_std_ms = float(np.std(gpu_times_ms, ddof=1)) if len(gpu_times_ms) > 1 else 0.0

        # Avoid division by zero
        safe_cpu_s = max(median_cpu_ms / 1000.0, 1e-7)
        safe_gpu_s = max(median_gpu_ms / 1000.0, 1e-7)

        # Throughput: windows/second
        cpu_wps = round(total_windows / safe_cpu_s, 2)
        gpu_wps = round(total_windows / safe_gpu_s, 2)

        # Throughput: Megabases / second (Mbp/s = (ref_len / 1e6) / seconds)
        cpu_mbps = round((ref_len / 1_000_000.0) / safe_cpu_s, 3)
        gpu_mbps = round((ref_len / 1_000_000.0) / safe_gpu_s, 3)

        speedup = round(safe_cpu_s / safe_gpu_s, 2)

        # Diagnostic explanation clearly distinguishing CPU reference, PyTorch CPU fallback, and true CUDA
        if self.gpu_engine.is_hardware_accelerated:
            # Physical CUDA GPU active
            if median_gpu_ms > median_cpu_ms:
                ratio = round(1.0 / max(speedup, 1e-5), 2)
                notes = (
                f"CUDA GPU kernel launch and memory transfer overhead exceeded compute time for small input ({ref_len:,} bp). "
                    f"CPU was faster by {ratio}x due to zero transfer overhead."
                )
            else:
                notes = (
                    f"CUDA GPU hardware acceleration achieved {speedup}x speedup via parallel matrix unfolding "
                    f"across GPU streaming multiprocessors."
                )
        else:
            # PyTorch CPU Vectorized Fallback
            if median_gpu_ms > median_cpu_ms:
                notes = (
                    f"PyTorch CPU Vectorized Fallback: Tensor memory allocation overhead on {ref_len:,} bp exceeded scalar loop runtime. "
                    f"Measured on host CPU (CUDA unavailable); pure Python reference was faster."
                )
            else:
                notes = (
                    f"PyTorch CPU Vectorized Fallback: Vector C++ tensor batching on CPU achieved {speedup}x speedup over scalar Python loop. "
                    f"Measured on host CPU (CUDA unavailable; this is CPU tensor optimization, NOT physical GPU hardware)."
                )

        return BenchmarkMetric(
            sequence_length=ref_len,
            windows_evaluated=total_windows,
            cpu_time_ms=round(median_cpu_ms, 3),
            gpu_time_ms=round(median_gpu_ms, 3),
            cpu_throughput_wps=cpu_wps,
            gpu_throughput_wps=gpu_wps,
            cpu_mbps=cpu_mbps,
            gpu_mbps=gpu_mbps,
            cpu_std_ms=round(cpu_std_ms, 3),
            gpu_std_ms=round(gpu_std_ms, 3),
            cpu_min_ms=round(cpu_min_ms, 3),
            cpu_max_ms=round(cpu_max_ms, 3),
            gpu_min_ms=round(gpu_min_ms, 3),
            gpu_max_ms=round(gpu_max_ms, 3),
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
                "CPU Std Dev (ms)": metric.cpu_std_ms,
                "GPU Engine Time (ms)": metric.gpu_time_ms,
                "GPU Std Dev (ms)": metric.gpu_std_ms,
                "CPU Throughput (win/s)": metric.cpu_throughput_wps,
                "GPU Engine Throughput (win/s)": metric.gpu_throughput_wps,
                "CPU Throughput (Mbp/s)": metric.cpu_mbps,
                "GPU Engine Throughput (Mbp/s)": metric.gpu_mbps,
                "Speedup (CPU / GPU)": metric.speedup_ratio,
                "Active Device": metric.active_device,
                "CUDA Enabled": metric.is_cuda_accelerated,
                "Notes": metric.notes,
            })

        return pd.DataFrame(rows)

    def run_fasta_benchmark(
        self,
        fasta_records: Optional[List[Any]] = None,
        guide_seq: str = "GAGTCCGAGCAGAAGAAGAA",
        pam_pattern: str = "NGG",
        max_mismatches: int = 3,
        iterations: int = 2,
    ) -> pd.DataFrame:
        """
        Execute benchmark profiling on reference FASTA records.
        Clearly labels synthetic demonstration datasets as synthetic.

        Parameters:
            fasta_records (Optional[List[SequenceRecord]]): Records to benchmark.
                If None, loads built-in sample reference dataset.
            guide_seq (str): CRISPR guide sequence.
            pam_pattern (str): PAM motif.
            max_mismatches (int): Max mismatch threshold.
            iterations (int): Timing iterations per record.

        Returns:
            pd.DataFrame: Comparative performance metrics across reference records.
        """
        if fasta_records is None:
            from src.input_handler import load_sample_reference_fasta
            fasta_records = load_sample_reference_fasta(strict=True)

        rows = []
        for rec in fasta_records:
            metric = self.run_single_comparison(
                reference_seq=rec.sequence,
                guide_seq=guide_seq,
                pam_pattern=pam_pattern,
                max_mismatches=max_mismatches,
                scan_both_strands=True,
                warmup=True,
                iterations=iterations,
            )

            # Clearly label synthetic vs experimental provenance
            desc_lower = (rec.description or "").lower()
            id_lower = (rec.id or "").lower()
            dataset_type = "Synthetic Demonstration Locus" if ("synthetic" in desc_lower or "synthetic" in id_lower) else "Reference Locus"

            rows.append({
                "Record ID": rec.id,
                "Dataset Type": dataset_type,
                "Sequence Length (bp)": metric.sequence_length,
                "Windows Evaluated": metric.windows_evaluated,
                "CPU Time (ms)": metric.cpu_time_ms,
                "CPU Std Dev (ms)": metric.cpu_std_ms,
                "GPU Engine Time (ms)": metric.gpu_time_ms,
                "GPU Std Dev (ms)": metric.gpu_std_ms,
                "CPU Throughput (Mbp/s)": metric.cpu_mbps,
                "GPU Engine Throughput (Mbp/s)": metric.gpu_mbps,
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

