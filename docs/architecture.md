# GeneWeaver System Architecture & Design Specification

## 1. Architectural Overview

**GeneWeaver** is structured around a decoupled, three-tier architecture:
1. **Bioinformatics & Data Layer (`src/input_handler.py`, `src/validator.py`, `src/sequence_utils.py`)**:
   Responsible for input sanitization, FASTA file stream parsing via Biopython, IUPAC validation, and sequence normalisation.
2. **Dual Computational Core (`src/crispr_alignment.py`, `src/cpu_engine.py`, `src/gpu_engine.py`)**:
   Implements dual-strand sliding-window search algorithms across both CPU (pure Python reference) and GPU (PyTorch vectorized tensor unfolding).
3. **Analytics, Presentation & Visualization Layer (`app.py`, `pages/*.py`, `src/ui_components.py`, `src/benchmark.py`, `src/report_generator.py`)**:
   Interactive multi-page Streamlit web dashboard, Plotly statistical charting, mismatch visualization, and automated CSV/Markdown report generation.

---

## 2. Component Diagram & Data Flow

```
                      +---------------------------------------+
                      |         Data Input Sources            |
                      |  - Pasted DNA / Multi-FASTA           |
                      |  - FASTA Upload (.fasta/.fa)          |
                      |  - Packaged Synthetic Reference Loci  |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |       Input Handler & Validator       |
                      |  - SequenceRecord construction        |
                      |  - Canonical base validation (ATGC)   |
                      |  - IUPAC PAM pattern compilation      |
                      |  - Nucleotide metrics & GC content    |
                      +-------------------+-------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
     +-------------------------------+         +-------------------------------+
     |       CPU Engine Core         |         |      GPU / Tensor Core        |
     | - Sliding window string slice |         | - Integer Tensor Encoding     |
     | - Iterative character checks  |         | - PyTorch unfold() Striding   |
     | - time.perf_counter() timer   |         | - CUDA Batch Kernel Launch    |
     | - Deterministic Ground Truth  |         | - CPU Tensor Fallback Engine  |
     +---------------+---------------+         +---------------+---------------+
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |    CRISPR Scoring & Visualizer        |
                      |  - Mismatch counting & 1-based indices|
                      |  - Seed (11-20) vs Distal (1-10) mm   |
                      |  - Weighted specificity score (0-100) |
                      |  - ASCII / HTML connector breakdown   |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Streamlit Interactive UI          |
                      |  - Sequence Analyzer                  |
                      |  - CRISPR Alignment Inspector         |
                      |  - Benchmark Performance Dashboard    |
                      |  - Dataset Explorer & CSV Exporter    |
                      +---------------------------------------+
```

---

## 3. Class and Module Reference

### `src/sequence_utils.py`
- `clean_sequence(seq: str) -> str`: Normalizes whitespace, strips numbers, and transforms bases to uppercase.
- `reverse_complement(seq: str) -> str`: Calculates the $5' \rightarrow 3'$ antisense sequence using standard Watson-Crick base pairing ($A \leftrightarrow T$, $C \leftrightarrow G$).
- `calculate_nucleotide_stats(seq: str) -> Dict[str, Any]`: Computes counts of A, C, G, T, GC/AT percentages, and purine/pyrimidine ratios.
- `transcribe_to_rna(seq: str) -> str`: Replaces Thymine (T) with Uracil (U).

### `src/validator.py`
- `ValidationResult`: Dataclass containing `is_valid`, `cleaned_sequence`, `errors`, `warnings`, `invalid_characters`, and `stats`.
- `validate_dna_sequence(...) -> ValidationResult`: Enforces canonical base validity, length bounds, and reports exact 1-indexed coordinates of illegal characters.
- `validate_guide_sequence(...) -> ValidationResult`: Enforces guide RNA constraints and flags poly-T transcription terminators.
- `validate_pam_pattern(...) -> ValidationResult`: Validates PAM against the 15 IUPAC nucleotide ambiguity codes.

### `src/crispr_alignment.py`
- `CRISPRTargetMatch`: Dataclass representing a candidate genomic target site with coordinates, strand, PAM, mismatch list, seed/distal breakdown, and specificity score.
- `AlignmentEngineResult`: Comprehensive execution result with wall-clock latency, throughput, device metadata, and candidate match list.
- `match_pam(candidate: str, pattern: str) -> bool`: Degenerate IUPAC pattern matching for Cas nuclease recognition.
- `compute_mismatches(guide: str, target: str) -> Tuple`: Extracts mismatch count, positions, seed count, distal count, and visual connector string.
- `calculate_alignment_score(...) -> float`: Seed-weighted specificity penalty algorithm.

### `src/cpu_engine.py` & `src/gpu_engine.py`
- `CPUEngine.align(...)`: Deterministic CPU sliding window reference implementation.
- `GPUEngine.align(...)`: Tensor-accelerated sliding window batch comparison using PyTorch's `unfold()` and integer nucleotide representations. Gracefully switches between `cuda:0` and `cpu` with strict hardware reporting.

---

## 4. Hardware Acceleration and Boundary Handling

| State | Detection Condition | Execution Strategy | UI Label |
| :--- | :--- | :--- | :--- |
| **CUDA Available** | `torch.cuda.is_available() == True` | PyTorch tensors transferred to GPU memory (`cuda:0`). Windows synchronized before and after kernels. | `🚀 GPU: <device_name> (CUDA)` |
| **CUDA Unavailable** | `torch.cuda.is_available() == False` | PyTorch tensors evaluated using CPU vectorized SIMD kernels. | `⚡ CPU (PyTorch Vectorized Fallback)` |
| **Forced CPU** | `force_cpu=True` flag | Pure Python sliding window loop. Baseline reference. | `CPU (Python Native Reference)` |
