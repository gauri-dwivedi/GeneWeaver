"""
GeneWeaver: GPU-Accelerated Sequence Alignment Engine
=====================================================
Utilizes PyTorch tensor computations to parallelize sliding-window CRISPR target
matching and mismatch calculations.

Educational Principles for Students:
------------------------------------
How GPU Acceleration Works in Bioinformatics:
1. Conventional CPUs evaluate DNA windows iteratively (one window at a time in a loop).
2. Modern GPUs possess thousands of compute cores optimized for parallel matrix arithmetic.
3. In GeneWeaver:
   a. Nucleotides (A, C, G, T) are vectorized into numeric tensors:
      A -> 0, C -> 1, G -> 2, T -> 3.
   b. PyTorch's `unfold()` creates a 2D tensor of all candidate sliding windows simultaneously
      without redundant memory duplication.
   c. Vectorized matrix comparisons `(windows != guide_tensor)` evaluate millions of base pairs
      in parallel batches.
   d. The reduction `sum(dim=-1)` calculates mismatch counts across all windows in a single kernel launch.

Hardware Detection & Integrity:
- If a compatible NVIDIA CUDA GPU is present, operations execute on `cuda`.
- If CUDA is not available, the engine automatically falls back to CPU tensor processing.
- The active processing device is strictly reported. GeneWeaver NEVER claims GPU processing
  occurred when running on CPU.
"""

import time
from typing import List, Tuple, Dict, Any, Optional
import torch
from src.crispr_alignment import (
    CRISPRTargetMatch,
    AlignmentEngineResult,
    match_pam,
    compute_mismatches,
    calculate_alignment_score,
)
from src.sequence_utils import clean_sequence, reverse_complement
from src.validator import IUPAC_NUCLEOTIDE_MAP

# Numerical mapping dictionary for DNA bases
BASE_TO_INT: Dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
}
INT_TO_BASE: Dict[int, str] = {v: k for k, v in BASE_TO_INT.items()}


def encode_sequence_to_tensor(seq: str, device: torch.device) -> torch.Tensor:
    """
    Encode an uppercase nucleotide sequence string into a 1D PyTorch int64 tensor.
    Any non-ACGT character is encoded as 9 (wildcard / non-match).

    Parameters:
        seq (str): Cleaned DNA string.
        device (torch.device): Target torch device (cuda or cpu).

    Returns:
        torch.Tensor: 1D tensor of integer-encoded nucleotides.
    """
    encoded = [BASE_TO_INT.get(ch, 9) for ch in seq]
    return torch.tensor(encoded, dtype=torch.int64, device=device)


class GPUEngine:
    """
    Tensor-accelerated CRISPR target alignment engine using PyTorch.
    """

    def __init__(self, force_cpu: bool = False):
        """
        Initialize the GPU engine, probing for CUDA hardware acceleration.

        Parameters:
            force_cpu (bool): If True, forces execution on CPU even if CUDA is present.
        """
        self.cuda_available = torch.cuda.is_available() and not force_cpu

        if self.cuda_available:
            self.device = torch.device("cuda:0")
            gpu_name = torch.cuda.get_device_name(0)
            self.device_name = f"GPU: {gpu_name} (CUDA)"
            self.is_hardware_accelerated = True
        else:
            self.device = torch.device("cpu")
            self.device_name = "CPU (PyTorch Vectorized Fallback - CUDA Unavailable)"
            self.is_hardware_accelerated = False

    def get_device_info(self) -> Dict[str, Any]:
        """
        Return comprehensive metadata about the current compute device.
        """
        info = {
            "cuda_available": torch.cuda.is_available(),
            "device_used": self.device_name,
            "is_gpu_active": self.is_hardware_accelerated,
            "torch_version": torch.__version__,
        }
        if self.cuda_available:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["device_count"] = torch.cuda.device_count()
            info["current_device"] = torch.cuda.current_device()
        else:
            info["fallback_reason"] = "No CUDA-capable GPU detected; using CPU tensor engine."
        return info

    def _scan_strand_tensor(
        self,
        dna_seq: str,
        guide_seq: str,
        pam_pattern: str,
        max_mismatches: int,
        strand_symbol: str,
        reference_id: str,
        original_ref_len: int,
        is_reverse: bool = False,
    ) -> Tuple[List[CRISPRTargetMatch], int]:
        """
        Perform vectorized sliding-window search over a single strand using PyTorch tensors.

        Parameters:
            dna_seq (str): Sequence of the strand to evaluate.
            guide_seq (str): Guide sequence.
            pam_pattern (str): PAM pattern (e.g. 'NGG').
            max_mismatches (int): Maximum mismatch threshold.
            strand_symbol (str): '+' or '-'.
            reference_id (str): Reference scaffold ID.
            original_ref_len (int): Length of the forward reference.
            is_reverse (bool): True if evaluating the reverse complement strand.

        Returns:
            Tuple[List[CRISPRTargetMatch], int]: Identified matches and evaluated window count.
        """
        guide_len = len(guide_seq)
        pam_len = len(pam_pattern)
        window_len = guide_len + pam_len
        strand_len = len(dna_seq)

        if strand_len < window_len:
            return [], 0

        # Encode sequence and guide into tensors
        strand_tensor = encode_sequence_to_tensor(dna_seq, self.device)
        guide_tensor = encode_sequence_to_tensor(guide_seq, self.device)

        # PyTorch unfold: creates sliding windows of shape [num_windows, window_len]
        # Memory-efficient view without data copying
        windows = strand_tensor.unfold(dimension=0, size=window_len, step=1)
        num_windows = windows.size(0)

        matches: List[CRISPRTargetMatch] = []

        # Vectorized PAM evaluation:
        # Extract candidate PAM tensors (last pam_len positions of each window)
        pam_slices = windows[:, guide_len:window_len]  # Shape: [num_windows, pam_len]
        target_slices = windows[:, 0:guide_len]        # Shape: [num_windows, guide_len]

        # For canonical SpCas9 'NGG', vectorized check:
        # pam_pattern is 3 chars: [0]='N', [1]='G', [2]='G'
        if pam_pattern.upper() == "NGG" and pam_len == 3:
            g_val = BASE_TO_INT["G"]
            # Base 1 and Base 2 must be 'G' (index 1 and 2 in pam slice)
            pam_mask = (pam_slices[:, 1] == g_val) & (pam_slices[:, 2] == g_val)
        else:
            # General IUPAC PAM check using candidate string lookup
            pam_mask = torch.zeros(num_windows, dtype=torch.bool, device=self.device)
            pam_slices_cpu = pam_slices.cpu()
            for w_idx in range(num_windows):
                pam_chars = "".join(INT_TO_BASE.get(int(v), "N") for v in pam_slices_cpu[w_idx])
                if match_pam(pam_chars, pam_pattern):
                    pam_mask[w_idx] = True

        valid_window_indices = torch.nonzero(pam_mask, as_tuple=False).squeeze(-1)

        if valid_window_indices.numel() == 0:
            return [], num_windows

        # Filter candidate targets that have a matching PAM
        candidate_targets = target_slices[valid_window_indices]  # Shape: [num_valid, guide_len]

        # Batch mismatch tensor comparison:
        # Broadcast guide_tensor [guide_len] across all candidate targets [num_valid, guide_len]
        mismatch_matrix = (candidate_targets != guide_tensor.unsqueeze(0))
        mismatch_counts = mismatch_matrix.sum(dim=-1)  # Shape: [num_valid]

        # Filter targets that satisfy max_mismatches threshold
        qualifying_mask = (mismatch_counts <= max_mismatches)
        qualifying_indices = valid_window_indices[qualifying_mask]
        qualifying_mm_counts = mismatch_counts[qualifying_mask]

        if qualifying_indices.numel() == 0:
            return [], num_windows

        # Transfer only qualifying indices back to CPU host to build detailed match records
        qual_idx_list = qualifying_indices.cpu().tolist()
        if isinstance(qual_idx_list, int):
            qual_idx_list = [qual_idx_list]

        for w_idx in qual_idx_list:
            cand_target = dna_seq[w_idx : w_idx + guide_len]
            cand_pam = dna_seq[w_idx + guide_len : w_idx + window_len]

            mm_count, mm_pos, seed_mm, distal_mm, visual = compute_mismatches(guide_seq, cand_target)
            score = calculate_alignment_score(guide_len, mm_count, seed_mm, distal_mm)

            if is_reverse:
                fwd_start = original_ref_len - (w_idx + window_len) + 1
                fwd_end = original_ref_len - w_idx
                full_win = dna_seq[w_idx : w_idx + window_len]
            else:
                fwd_start = w_idx + 1
                fwd_end = w_idx + window_len
                full_win = dna_seq[w_idx : w_idx + window_len]

            match = CRISPRTargetMatch(
                reference_id=reference_id,
                strand=strand_symbol,
                start_pos=fwd_start,
                end_pos=fwd_end,
                target_dna_sequence=cand_target,
                pam_sequence=cand_pam,
                full_window=full_win,
                mismatch_count=mm_count,
                mismatch_positions=mm_pos,
                seed_mismatches=seed_mm,
                distal_mismatches=distal_mm,
                alignment_score=score,
                alignment_visual=visual,
            )
            matches.append(match)

        return matches, num_windows

    def align(
        self,
        reference_seq: str,
        reference_id: str,
        guide_seq: str,
        pam_pattern: str = "NGG",
        max_mismatches: int = 3,
        scan_both_strands: bool = True,
    ) -> AlignmentEngineResult:
        """
        Scan a reference DNA sequence against a CRISPR guide sequence using PyTorch tensors.

        Parameters:
            reference_seq (str): Reference DNA sequence string.
            reference_id (str): Identifier of the reference sequence.
            guide_seq (str): 5' to 3' CRISPR guide sequence.
            pam_pattern (str): PAM motif (e.g. 'NGG').
            max_mismatches (int): Maximum allowable mismatches.
            scan_both_strands (bool): If True, scans both forward and reverse strands.

        Returns:
            AlignmentEngineResult: Complete alignment results including device and timing metrics.
        """
        clean_ref = clean_sequence(reference_seq)
        clean_guide = clean_sequence(guide_seq)
        clean_pam = clean_sequence(pam_pattern)

        ref_len = len(clean_ref)
        guide_len = len(clean_guide)

        matches: List[CRISPRTargetMatch] = []
        total_windows = 0

        # Synchronize GPU queue if running on CUDA to ensure precise kernel benchmarking
        if self.cuda_available:
            torch.cuda.synchronize()
        start_time = time.perf_counter()

        # 1. Forward strand scanning
        fwd_matches, fwd_windows = self._scan_strand_tensor(
            dna_seq=clean_ref,
            guide_seq=clean_guide,
            pam_pattern=clean_pam,
            max_mismatches=max_mismatches,
            strand_symbol="+",
            reference_id=reference_id,
            original_ref_len=ref_len,
            is_reverse=False,
        )
        matches.extend(fwd_matches)
        total_windows += fwd_windows

        # 2. Reverse strand scanning
        if scan_both_strands:
            ref_rc = reverse_complement(clean_ref)
            rev_matches, rev_windows = self._scan_strand_tensor(
                dna_seq=ref_rc,
                guide_seq=clean_guide,
                pam_pattern=clean_pam,
                max_mismatches=max_mismatches,
                strand_symbol="-",
                reference_id=reference_id,
                original_ref_len=ref_len,
                is_reverse=True,
            )
            matches.extend(rev_matches)
            total_windows += rev_windows

        if self.cuda_available:
            torch.cuda.synchronize()
        end_time = time.perf_counter()

        elapsed_seconds = end_time - start_time
        elapsed_ms = elapsed_seconds * 1000.0

        # Sort matches identically to CPU engine
        matches.sort(key=lambda m: (m.mismatch_count, -m.alignment_score))

        throughput = (total_windows / elapsed_seconds) if elapsed_seconds > 0 else float("inf")

        return AlignmentEngineResult(
            matches=matches,
            total_windows_evaluated=total_windows,
            execution_time_seconds=elapsed_seconds,
            execution_time_ms=elapsed_ms,
            device_used=self.device_name,
            windows_per_second=round(throughput, 2),
            guide_length=guide_len,
            pam_pattern=clean_pam,
            max_mismatches=max_mismatches,
        )
