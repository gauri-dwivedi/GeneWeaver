# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine
## Individual Internship Project Report
**Organization:** Infotact Solutions  
**Author:** BTech Computer Science Student  
**Academic Year:** 2026–2027  

---

## 1. Abstract
CRISPR-Cas9 is a transformative genome editing technology that relies on an RNA guide sequence to direct the Cas9 endonuclease to specific genomic coordinates. A major bioinformatic challenge in CRISPR experiments is identifying potential off-target binding sites across large reference genomes. Conventional alignment methods running in interpreted languages can suffer from computational bottlenecks when scanning millions of candidate base pair windows. This project introduces **GeneWeaver**, a functional, interactive software application developed in Python. GeneWeaver pairs a deterministic CPU reference alignment engine with a parallelized PyTorch tensor execution engine to scan double-stranded DNA for candidate CRISPR target sites, recognize Protospacer Adjacent Motifs (PAM), calculate mismatch counts, and evaluate seed-weighted specificity scores. The application features an interactive Streamlit web interface with real-time performance benchmarking between CPU and GPU/tensor modes. The system was verified across 33 automated unit tests, and empirical benchmarking demonstrated substantial throughput advantages for vectorized processing on larger sequence datasets.

---

## 2. Introduction
Genome engineering using the CRISPR-Cas9 system has revolutionized biotechnology, functional genomics, and gene therapy. In this system, a single-guide RNA (sgRNA) containing a ~20 nucleotide spacer sequence pairs with complementary genomic target sites (protospacers) flanked by a short Protospacer Adjacent Motif (PAM), such as $5'\text{-NGG-}3'$ for *Streptococcus pyogenes* Cas9 (SpCas9).

While CRISPR is remarkably specific, off-target binding and cleavage can occur at genomic sites that have sequence homology with the guide RNA despite having one or more base mismatches. Identifying these candidate off-target sites computationally is an essential safety and design step before executing laboratory experiments.

As a computer science student exploring bioinformatics, this project offered an opportunity to combine sequence analysis algorithms, high-performance computing concepts, and modern web application development. GeneWeaver was developed from scratch to demonstrate how low-level vectorization and GPU-compatible tensor striding can accelerate repetitive sliding-window string comparisons.

---

## 3. Problem Statement
Identifying CRISPR targets across genomic sequences presents two distinct challenges:
1. **Algorithmic Accuracy**: The search must account for biological constraints, such as PAM motifs (including degenerate IUPAC codes), scanning both the forward (sense) and reverse-complement (antisense) DNA strands, and assigning biologically realistic penalties where mutations adjacent to the PAM (the seed region) disrupt Cas9 activity much more than distal mutations.
2. **Computational Scalability**: Scanning a DNA sequence requires evaluating an overlapping window at every consecutive base pair. In traditional Python implementations, evaluating millions of windows using standard character-by-character loops can be slow. Furthermore, beginners often struggle to understand why and when GPU computing provides a speedup over CPU computing.

There is a need for an accessible, transparent, and functional tool that demonstrates both the biological logic of CRISPR alignment and the computer science principles of hardware acceleration without black-box or fabricated metrics.

---

## 4. Objectives
The primary objectives of the GeneWeaver project are:
1. **Sequence Input and Validation**: Build a robust module to accept pasted DNA text or uploaded FASTA files, validate canonical nucleotide characters (A, T, G, C), strip whitespace and numbers, and calculate nucleotide composition metrics.
2. **CRISPR Alignment Engine**: Implement a dual-strand sliding window algorithm capable of matching guide sequences against DNA targets, recognizing canonical (`NGG`) and custom PAM patterns, and recording 1-based biological coordinates.
3. **Seed-Weighted Scoring**: Implement a scoring algorithm that reflects CRISPR biophysics by penalizing mismatches in the PAM-proximal seed region (bases 11–20) more severely than PAM-distal mismatches.
4. **Dual-Engine Execution (CPU & GPU)**: Develop a deterministic CPU reference engine alongside a PyTorch tensor batch engine. Ensure the GPU engine automatically detects CUDA hardware, falls back to CPU tensor operations when CUDA is absent, and never fabricates fake GPU metrics.
5. **Interactive Benchmarking**: Empirically profile execution latency, throughput (windows per second), and speedup across varying dataset sizes.
6. **Student-Friendly Interactive UI**: Create a clean, scientific Streamlit web dashboard with visualization tools, mismatch highlighters, and report download features.
7. **Automated Verification**: Validate all algorithms, edge cases, and hardware fallbacks using an automated Pytest test suite.

---

## 5. Existing Approach
Existing CRISPR guide design tools fall into two main categories:
1. **Online Web Portals (e.g., CRISPOR, Benchling, CHOPCHOP)**: These web platforms rely on remote cloud clusters and backend command-line aligners such as BWA or Bowtie. While powerful, they act as closed systems for students and do not allow users to observe the internal mechanics of hardware acceleration, tensor batching, or CPU-GPU execution trade-offs on local hardware.
2. **Static Scripts & Demonstration Demos**: Educational sample projects often rely on static or hardcoded outputs, do not perform genuine multi-strand coordinate mapping, or lack automated error handling for non-canonical nucleotide inputs.

---

## 6. Proposed System
GeneWeaver provides a completely functional, local, open-source bioinformatics application:
- **Biopython Integration**: Employs industry-standard sequence parsers for FASTA file ingestion.
- **Dual-Engine Design**:
  - The **CPU Reference Engine** executes an explicit, transparent sliding-window comparison, acting as the verifiable ground truth.
  - The **GPU/Tensor Engine** utilizes PyTorch integer-encoded tensors ($A=0, C=1, G=2, T=3$) and the zero-copy `unfold()` operation to evaluate thousands of candidate windows in parallel.
- **Hardware-Aware Fallback**: If an NVIDIA GPU is available, the engine runs on CUDA; if not, it automatically runs PyTorch CPU vectorization, explicitly reporting the active device to the user.
- **Interactive Multi-Page Streamlit App**: Users can explore sequences, perform alignments, inspect visual mismatch connectors, run real-time scaling benchmarks, and export CSV and Markdown reports.

---

## 7. Technologies Used
- **Python (v3.13)**: Primary programming language.
- **Biopython (v1.88)**: Biological sequence handling and standard FASTA stream parsing.
- **NumPy (v2.5)**: Numerical vector operations and random sequence generation for benchmarks.
- **PyTorch (v2.15)**: Multi-dimensional tensor striding (`unfold`), parallel PAM masking, and CUDA acceleration.
- **Pandas (v3.1)**: Structured tabular manipulation, filtering, and CSV export.
- **Plotly (v7.1)**: Interactive charting for nucleotide composition and benchmark scaling curves.
- **Streamlit (v1.67)**: Modern interactive web application framework.
- **Pytest (v9.2)**: Automated unit testing and integration verification.

---

## 8. System Architecture
GeneWeaver is organized into three distinct architectural tiers:

1. **Input & Validation Tier**:
   - `input_handler.py`: Reads pasted text and FASTA files via `Bio.SeqIO`. Loads pre-packaged sample references and guide libraries.
   - `validator.py`: Enforces canonical base validity, flags degenerate IUPAC characters, and detects poly-T tracts.
   - `sequence_utils.py`: Normalizes text, computes GC content, calculates reverse-complements, and transcribes DNA to RNA.
2. **Computational Engine Tier**:
   - `crispr_alignment.py`: Defines data models (`CRISPRTargetMatch`, `AlignmentEngineResult`), PAM matcher, and seed-weighted scoring.
   - `cpu_engine.py`: Deterministic sliding-window reference scanner.
   - `gpu_engine.py`: Tensorized parallel scanner using PyTorch striding.
3. **Analytics & Dashboard Tier**:
   - `app.py`: Project overview and architectural specifications.
   - `pages/1_Sequence_Analyzer.py`: Sequence inspection and composition charts.
   - `pages/2_CRISPR_Alignment.py`: Target search, visual mismatch highlighter, and match table.
   - `pages/3_Performance_Benchmark.py`: Real-time latency and throughput profiling.
   - `pages/4_Dataset_Explorer.py`: Sample data browser and data dictionary.
   - `pages/5_Results.py`: Report viewer and CSV exporter.

---

## 9. Methodology

### 9.1 Sequence Normalization and Validation
Raw sequences are stripped of whitespace and numbers. The validator inspects every character against the canonical set $\{A, C, G, T\}$. If illegal characters are found, their 1-indexed positions and unique identities are reported.

### 9.2 Dual-Strand Sliding Window
For a guide of length $L$ and PAM of length $K$, the window size is $W = L + K$ (23 bp for standard SpCas9).
- **Forward Strand**: Slides across $R[i : i + W]$ for $i \in [0, N - W]$.
- **Reverse Strand**: Reverses and complements the reference sequence ($R_{rc}$) and runs the identical sliding window. When a match is found at $i_{rc}$, coordinates are mapped back to forward 1-based coordinates:
  $$\text{Start} = N - (i_{rc} + W) + 1, \quad \text{End} = N - i_{rc}$$

### 9.3 Seed-Weighted Scoring Formula
Mismatches in the PAM-proximal seed region (bases 11–20) receive a penalty weight of $2.5$, whereas PAM-distal mismatches (bases 1–10) receive a penalty weight of $1.0$:
$$\text{Penalty } P = (\text{Distal MM} \times 1.0) + (\text{Seed MM} \times 2.5)$$
$$\text{Score } S = 100.0 \times \max\left(0.0, 1.0 - \frac{P}{L \times 1.75}\right)$$

### 9.4 Tensor Acceleration Logic
1. Sequences are converted to integer arrays ($A=0, C=1, G=2, T=3$).
2. PyTorch's `unfold(dimension=0, size=W, step=1)` creates a 2D matrix of shape $[M, W]$ where $M = N - W + 1$.
3. A vectorized boolean mask extracts only windows where the PAM matches.
4. A broadcasted inequality comparison `(candidate_targets != guide_tensor)` calculates all mismatch counts across thousands of windows simultaneously.

---

## 10. Implementation Details
The application codebase is modularized cleanly in the `src/` directory. High-precision timing is conducted using `time.perf_counter()`. CUDA synchronization calls (`torch.cuda.synchronize()`) ensure that asynchronous GPU queues complete prior to timing readings. All UI pages follow a dark navy theme with custom CSS styling and responsive layout containers.

---

## 11. Testing & Quality Assurance
The project includes a comprehensive automated test suite in `tests/` executed via Pytest. Testing verified:
- Sequence cleaning, reverse complementation, and composition metrics (`test_validation.py`).
- Input validation on valid, invalid, empty, and short inputs (`test_validation.py`).
- Exact match forward alignment, reverse-complement alignment, and PAM recognition (`test_alignment.py`).
- Mismatch filtering and seed vs distal score calculations (`test_alignment.py`).
- Numerical and coordinate equivalence between CPU and GPU engines (`test_cpu_gpu.py`).
- Hardware fallback behavior when CUDA is absent (`test_cpu_gpu.py`).
- Single and scaling benchmarking suite execution and CSV export (`test_benchmark.py`).

**Test Execution Outcome:**
```
============================= test session starts =============================
platform win32 -- Python 3.13.12, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Gauri Dwivedi\Documents\GeneWeaver
collected 33 items

tests/test_alignment.py::test_match_pam_spcas9_ngg PASSED                [  3%]
tests/test_alignment.py::test_match_pam_degenerate_iupac PASSED          [  6%]
tests/test_alignment.py::test_compute_mismatches_exact PASSED            [  9%]
tests/test_alignment.py::test_compute_mismatches_seed_vs_distal PASSED   [ 12%]
tests/test_alignment.py::test_calculate_alignment_score PASSED           [ 15%]
tests/test_alignment.py::test_cpu_engine_exact_match_forward PASSED      [ 18%]
tests/test_alignment.py::test_cpu_engine_reverse_complement_match PASSED [ 21%]
tests/test_alignment.py::test_cpu_engine_mismatch_filtering PASSED       [ 24%]
tests/test_benchmark.py::test_benchmark_single_comparison PASSED         [ 27%]
tests/test_benchmark.py::test_benchmark_scaling PASSED                   [ 30%]
tests/test_benchmark.py::test_benchmark_csv_export PASSED                [ 33%]
tests/test_benchmark.py::test_report_generator PASSED                    [ 36%]
tests/test_cpu_gpu.py::test_gpu_device_detection PASSED                  [ 39%]
tests/test_cpu_gpu.py::test_cpu_gpu_result_equivalence PASSED            [ 42%]
tests/test_cpu_gpu.py::test_cpu_gpu_on_sample_fasta PASSED               [ 45%]
tests/test_validation.py::test_clean_sequence PASSED                     [ 48%]
tests/test_validation.py::test_reverse_complement_canonical PASSED       [ 51%]
tests/test_validation.py::test_reverse_complement_symmetry PASSED        [ 54%]
tests/test_validation.py::test_calculate_nucleotide_stats PASSED         [ 57%]
tests/test_validation.py::test_calculate_nucleotide_stats_empty PASSED   [ 60%]
tests/test_validation.py::test_transcribe_to_rna PASSED                  [ 63%]
tests/test_validation.py::test_validate_dna_sequence_valid PASSED        [ 66%]
tests/test_validation.py::test_validate_dna_sequence_case_and_whitespace PASSED [ 69%]
tests/test_validation.py::test_validate_dna_sequence_empty PASSED        [ 72%]
tests/test_validation.py::test_validate_dna_sequence_invalid_characters PASSED [ 75%]
tests/test_validation.py::test_validate_dna_sequence_min_length PASSED   [ 78%]
tests/test_validation.py::test_validate_guide_sequence_valid PASSED      [ 81%]
tests/test_validation.py::test_validate_guide_sequence_poly_t_warning PASSED [ 84%]
tests/test_validation.py::test_validate_pam_pattern PASSED               [ 87%]
tests/test_validation.py::test_parse_pasted_text_plain PASSED            [ 90%]
tests/test_validation.py::test_parse_pasted_text_fasta PASSED            [ 93%]
tests/test_validation.py::test_load_sample_reference_fasta PASSED        [ 96%]
tests/test_validation.py::test_load_sample_guides_csv PASSED             [100%]

============================= 33 passed in 7.82s ==============================
```

---

## 12. Results and Performance Measurements
All results reported here are **empirically measured** directly on the execution environment.

### Empirical Scaling Benchmark Results
| Sequence Length (bp) | Windows Evaluated | CPU Latency (ms) | GPU Engine Latency (ms) | CPU Throughput (win/s) | GPU Engine Throughput (win/s) | Measured Speedup | Active Device |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **500** | 956 | 1.171 ms | 0.487 ms | 816,500 | 1,961,025 | **2.40x** | CPU (PyTorch Fallback) |
| **2,500** | 4,956 | 7.337 ms | 3.299 ms | 675,448 | 1,502,227 | **2.22x** | CPU (PyTorch Fallback) |
| **10,000** | 19,956 | 34.947 ms | 6.273 ms | 571,037 | 3,181,354 | **5.57x** | CPU (PyTorch Fallback) |

### Key Observations:
1. **Result Equivalence**: The candidate targets identified by the GPU engine matched the CPU reference engine with 100% precision across all test runs.
2. **Throughput Scaling**: As the sequence length scaled from 500 bp to 10,000 bp, vectorized tensor striding maintained higher throughput, scaling to over 3.1 million windows scanned per second on the CPU vectorized backend.
3. **Hardware Transparency**: The engine reported that execution utilized the CPU fallback because dedicated CUDA hardware was not present in the runtime container.

---

## 13. Limitations
1. **Sequence Homology vs. Biological Cleavage**: GeneWeaver evaluates sequence matching and mismatch positioning. It does not model nucleosome occupancy, chromatin accessibility (DNase-seq / ATAC-seq), or cellular Cas9 kinetics. Computational homology does not equal guaranteed biological editing.
2. **Indel Alignments**: The current implementation utilizes sliding-window Hamming distance matching. It does not support gap insertions or deletions (indels) within the protospacer (which would require Smith-Waterman or Needleman-Wunsch dynamic programming).
3. **Whole-Genome Scale**: On full mammalian genomes (~3 billion base pairs), unrolling the entire genome into memory simultaneously would exceed system RAM; production whole-genome tools utilize indexed suffix arrays (e.g. FM-index / Burrows-Wheeler Transform).

---

## 14. Future Scope
1. **Dynamic Programming Extension**: Integrate GPU-accelerated Smith-Waterman algorithms to identify bulges, insertions, and deletions in off-target sites.
2. **Epigenetic Feature Integration**: Incorporate ENCODE chromatin accessibility and methylation data tracks to compute in vivo cleavage probabilities.
3. **Whole-Genome Suffix Array Indexing**: Implement FM-index or BWT pre-indexing to enable instant guide alignment against full human reference chromosomes (e.g., GRCh38).
4. **Machine Learning Specificity Models**: Integrate machine learning models (e.g. CFD score, Azimuth, or DeepCRISPR) to predict repair outcomes.

---

## 15. Conclusion
GeneWeaver successfully demonstrates how modern computational concepts, specifically tensor batching and vectorization, can be applied to solve bioinformatics problems in CRISPR guide alignment. By implementing a verified CPU reference engine, a hardware-aware PyTorch tensor engine, an automated 33-test quality suite, and an interactive Streamlit dashboard, this project delivers a functional and educational software product. The project reinforced vital computer science competencies in algorithmic complexity, PyTorch tensor striding, test-driven development, and scientific user experience design.

---

## 16. References
1. **Cong, L., et al.** (2013). Multiplex genome engineering using CRISPR/Cas systems. *Science*, 339(6121), 819–823.
2. **Fu, Y., et al.** (2013). High-frequency off-target mutagenesis induced by CRISPR-Cas nucleases in human cells. *Nature Biotechnology*, 31(9), 822–826.
3. **Hsu, P. D., et al.** (2013). DNA targeting specificity of RNA-guided Cas9 nucleases. *Nature Biotechnology*, 31(9), 827–832.
4. **Cock, P. J., et al.** (2009). Biopython: freely available Python tools for computational molecular biology and bioinformatics. *Bioinformatics*, 25(11), 1422–1423.
5. **Paszke, A., et al.** (2019). PyTorch: An imperative style, high-performance deep learning library. *Advances in Neural Information Processing Systems*, 32.

