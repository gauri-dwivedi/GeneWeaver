# GeneWeaver: GPU-Accelerated CRISPR Alignment Engine

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests Passing](https://img.shields.io/badge/tests-33%20passed-success.svg)](tests/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20PyTorch-orange.svg)](https://streamlit.io/)

**GeneWeaver** is an interactive, high-performance bioinformatics software application designed to search reference DNA sequences for candidate CRISPR-Cas9 target sites, calculate base mismatches, evaluate seed-weighted specificity scores, and benchmark CPU vs. GPU computing throughput.

Developed as an individual internship project at **Infotact Solutions**, GeneWeaver pairs an interpretable, deterministic Python reference engine on the CPU with a parallelized PyTorch tensor execution engine, providing both biological precision and modern hardware acceleration.

---

## 📑 Table of Contents
1. [Key Features](#-key-features)
2. [Project Architecture](#-project-architecture)
3. [Technology Stack](#-technology-stack)
4. [Installation & Windows Setup](#-installation--windows-setup)
5. [Running the Application](#-running-the-application)
6. [Application Modules](#-application-modules)
7. [Automated Testing](#-automated-testing)
8. [Real Empirical Benchmarks](#-real-empirical-benchmarks)
9. [Project Directory Structure](#-project-directory-structure)
10. [Student Internship Viva & Presentation Toolkit](#-student-internship-viva--presentation-toolkit)
    - [2-Minute Project Pitch](#1-2-minute-project-elevator-pitch)
    - [5-Minute Presentation Script](#2-5-minute-slide-by-slide-presentation-script)
    - [Top 10 Technical Viva Q&A](#3-top-10-technical-viva-questions--answers)
    - [Mentor Questions & Defenses](#4-common-mentor-questions--strategic-answers)
11. [Scientific Limitations & Caveats](#-scientific-limitations--caveats)
12. [License & Acknowledgments](#-license--acknowledgments)

---

## 🌟 Key Features

- **DNA Sequence Analysis & Validation**: Clean, format, and validate canonical nucleotides (A, C, G, T) with exact coordinate identification of non-standard characters.
- **Dual-Strand Sliding Window**: Scans both the forward (sense) and reverse-complement (antisense) strands with exact coordinate mapping.
- **Configurable PAM Recognition**: Evaluates candidate targets against standard SpCas9 (`NGG`) or custom IUPAC motif codes (`NAG`, `NNGRRT`).
- **Seed-Weighted Specificity Scoring**: Distinguishes between PAM-proximal seed mismatches (bases 11–20) and PAM-distal mismatches (bases 1–10) based on Cas9 biophysics.
- **Dual Computational Engine**:
  - **CPU Reference**: Deterministic character-by-character sliding-window scanner for ground-truth verification.
  - **GPU/Tensor Engine**: Integer nucleotide encoding ($A=0, C=1, G=2, T=3$) and zero-copy tensor striding via PyTorch's `unfold()`.
- **Automatic Hardware Sensing**: Automatically probes for NVIDIA CUDA hardware. If absent, falls back gracefully to CPU vectorization without fabricating GPU execution.
- **Interactive Benchmarking Suite**: Profiles real-time execution latency (ms) and throughput (windows/second) across scalable sequence sizes.
- **Interactive Web Dashboard**: Built with Streamlit featuring a professional dark navy theme, visual mismatch connectors (`||||||.||||||`), Plotly charts, and CSV/Markdown report exports.

---

## 🏗️ Project Architecture

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

## 💻 Technology Stack

| Technology | Role & Purpose |
| :--- | :--- |
| **Python 3.13** | Primary programming language |
| **Biopython (v1.88)** | FASTA stream parsing (`Bio.SeqIO`) and biological validation |
| **PyTorch (v2.15)** | GPU tensor batching, sliding-window `unfold()` striding, and CUDA acceleration |
| **Streamlit (v1.67)** | Interactive multi-page web dashboard framework |
| **Pandas (v3.1)** | Tabular dataset filtering and CSV report processing |
| **NumPy (v2.5)** | Numerical arrays and random synthetic sequence generation |
| **Plotly (v7.1)** | Interactive performance scaling and nucleotide composition charts |
| **Pytest (v9.2)** | Automated unit and integration test suite |

---

## ⚙️ Installation & Windows Setup

### Step 1: Open PowerShell
Open Windows PowerShell and navigate to the project directory:
```powershell
cd "C:\Users\Gauri Dwivedi\Documents\GeneWeaver"
```

### Step 2: Create and Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell shows a script execution error, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and retry activation).*

### Step 3: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🚀 Running the Application

Launch the Streamlit web dashboard:
```powershell
streamlit run app.py
```
The application will launch in your browser at `http://localhost:8501`.

---

## 📱 Application Modules

1. **Home / Overview (`app.py`)**: High-level system overview, device detection badge, architecture diagrams, and usage instructions.
2. **DNA Sequence Analyzer (`pages/1_Sequence_Analyzer.py`)**: Input validation, GC/AT ratios, base composition charts, and forward-to-alignment feature.
3. **CRISPR Target Alignment (`pages/2_CRISPR_Alignment.py`)**: Dual-strand sliding window matching, configurable PAMs, seed-weighted scores, visual alignment (`||||||.||||||`), and match export.
4. **CPU vs GPU Benchmarking (`pages/3_Performance_Benchmark.py`)**: Single-run and multi-scale latency and throughput profiling with interactive Plotly curves.
5. **Dataset Explorer (`pages/4_Dataset_Explorer.py`)**: Browser for pre-packaged synthetic references (*EMX1*, *VEGFA*), guide library, and complete data dictionary.
6. **Results & Downloads (`pages/5_Results.py`)**: Access session history, preview generated Markdown reports, and download CSV result files.

---

## 🧪 Automated Testing

GeneWeaver includes 33 automated tests in `tests/`:
```powershell
.\.venv\Scripts\pytest tests/ -v
```

### What is Tested:
- **`test_validation.py` (18 tests)**: Valid/invalid/empty inputs, whitespace cleaning, reverse complement symmetry, nucleotide stats, IUPAC PAM patterns, and FASTA file parsing.
- **`test_alignment.py` (8 tests)**: Exact match forward alignment, reverse strand alignment, seed vs distal positioning, mismatch cutoffs, and alignment scoring.
- **`test_cpu_gpu.py` (3 tests)**: Bit-for-bit and coordinate equivalence between CPU and GPU engines, device sensing, and sample FASTA evaluation.
- **`test_benchmark.py` (4 tests)**: Timing calculation, scaling metrics, report generation, and CSV export.

---

## 📊 Real Empirical Benchmarks

*Measured on the local runtime environment (Intel Core CPU, PyTorch CPU Vectorized Engine):*

| Sequence Length | Windows Evaluated | CPU Time | GPU/Tensor Time | CPU Throughput | GPU Throughput | Measured Speedup | Active Device |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **500 bp** | 956 | 1.171 ms | 0.487 ms | 816,500 win/s | 1,961,025 win/s | **2.40x** | CPU (PyTorch Fallback) |
| **2,500 bp** | 4,956 | 7.337 ms | 3.299 ms | 675,448 win/s | 1,502,227 win/s | **2.22x** | CPU (PyTorch Fallback) |
| **10,000 bp** | 19,956 | 34.947 ms | 6.273 ms | 571,037 win/s | 3,181,354 win/s | **5.57x** | CPU (PyTorch Fallback) |

> [!NOTE]
> GeneWeaver enforces strict reporting integrity. When running on systems without dedicated NVIDIA CUDA hardware, the system explicitly reports **CPU (PyTorch Vectorized Fallback)**. Even on CPU, PyTorch's C++ vectorized backend delivers 2.2x to 5.5x speedups over Python's interpreted loops.

---

## 📂 Project Directory Structure

```
GeneWeaver/
│
├── app.py                     # Main dashboard entry point & overview
├── requirements.txt           # Python dependency specifications
├── README.md                  # Comprehensive documentation & student viva guide
├── PROJECT_REPORT.md          # 16-section formal academic project report
├── .gitignore                 # Git ignore rules (.venv, caches, etc.)
│
├── data/                      # Standardized sample datasets & documentation
│   ├── sample_reference.fasta # Synthetic human EMX1, VEGFA, and repeat arrays
│   ├── sample_guides.csv      # Curated 8-guide library (exact, 1MM, 3MM, controls)
│   └── README.md              # Dataset descriptions and data dictionary
│
├── src/                       # Core modular application logic
│   ├── __init__.py            # Package initialization
│   ├── sequence_utils.py      # Normalization, reverse-complements, GC content
│   ├── validator.py           # DNA validation, IUPAC PAM parsing, error reporting
│   ├── input_handler.py       # FASTA stream ingestion & dataset loading
│   ├── crispr_alignment.py    # Match data models, PAM matcher, seed scoring
│   ├── cpu_engine.py          # Deterministic CPU reference engine
│   ├── gpu_engine.py          # PyTorch tensor-accelerated alignment engine
│   ├── benchmark.py           # Latency, throughput, and scaling suite
│   ├── report_generator.py    # Markdown and CSV report generation
│   └── ui_components.py       # Custom theme CSS, badges, and visual connectors
│
├── pages/                     # Interactive Streamlit multi-page interface
│   ├── 1_Sequence_Analyzer.py # Validation, nucleotide charts, transcription
│   ├── 2_CRISPR_Alignment.py  # Dual-strand search, visualizer, candidate table
│   ├── 3_Performance_Benchmark.py # CPU vs GPU latency, throughput, speedup
│   ├── 4_Dataset_Explorer.py  # Sample dataset browser & data dictionary
│   └── 5_Results.py           # Session run history & exported CSV downloads
│
├── tests/                     # Automated Pytest test suite
│   ├── __init__.py
│   ├── test_validation.py     # Module 1 & 2 tests
│   ├── test_alignment.py      # Module 3 & 4 tests
│   ├── test_cpu_gpu.py        # Module 5 CPU/GPU equivalence & fallback tests
│   └── test_benchmark.py      # Module 6 & reporting tests
│
├── results/                   # Destination for exported benchmark & alignment CSVs
│   ├── .gitkeep
│   ├── benchmark_run_*.csv
│   └── alignment_results_*.csv
│
└── docs/                      # Technical documentation
    ├── architecture.md        # Modular design, data flows, and hardware boundaries
    ├── methodology.md         # CRISPR biology, scoring equations, tensor unfolding
    └── user_guide.md          # Step-by-step user instructions
```

---

## 🎓 Student Internship Viva & Presentation Toolkit

### 1. 2-Minute Project Elevator Pitch
> *"Good morning, everyone. My internship project is **GeneWeaver: GPU-Accelerated CRISPR Alignment Engine**.*  
> *In CRISPR-Cas9 genome editing, a 20-nucleotide RNA guide directs the Cas9 enzyme to cut DNA at matching genomic target locations. However, Cas9 can accidentally bind to unintended off-target sites that contain base mismatches, which poses a serious safety concern in gene therapy.*  
> *The bioinformatic challenge is that searching for matching target sites across millions of base pairs is computationally expensive when done sequentially in standard Python.*  
> *To solve this, I built GeneWeaver from scratch. It features a dual-engine architecture: a deterministic CPU reference engine that establishes biological ground truth, and a GPU/tensor engine built in PyTorch that encodes DNA into integer tensors and uses memory striding via `unfold()` to evaluate thousands of candidate windows in parallel.*  
> *The system scans both forward and reverse-complement strands, recognizes standard SpCas9 `NGG` and custom PAM motifs, and calculates a seed-weighted specificity score that reflects Cas9 biology.*  
> *Everything is wrapped in an interactive Streamlit dashboard with real-time benchmarking, mismatch visualizers, and downloadable reports. All 33 automated tests pass, and our real benchmarks demonstrate up to a 5.5x speedup on larger sequence datasets."*

---

### 2. 5-Minute Slide-by-Slide Presentation Script

- **Slide 1: Title & Introduction**
  - *"Hello respected mentors. Today I present GeneWeaver, a dual-engine bioinformatics application developed during my individual internship at Infotact Solutions."*
- **Slide 2: The Problem**
  - *"CRISPR-Cas9 target site search requires sliding an overlapping 23 bp window (20 bp protospacer + 3 bp PAM) across every base pair of double-stranded DNA. In interpreted languages like Python, this iterative loop can become a performance bottleneck."*
- **Slide 3: Proposed Architecture**
  - *"GeneWeaver solves this through modular separation. Data enters through Biopython-validated FASTA handlers. We then provide two interchangeable engines: a pure Python CPU reference engine and a PyTorch tensor engine that leverages zero-copy matrix unfolding."*
- **Slide 4: Biological Fidelity**
  - *"Unlike simplistic string finders, GeneWeaver models real CRISPR biophysics: it evaluates both sense and antisense strands with exact coordinate mapping, supports IUPAC degenerate PAMs, and weights mismatches in the PAM-proximal seed region (bases 11–20) 2.5 times more heavily than PAM-distal mismatches."*
- **Slide 5: Live Benchmarks & Results**
  - *"On our test machine, we verified 100% result equivalence between CPU and GPU engines. Benchmarking demonstrated that while small sequences have kernel setup overhead, larger sequences (>10,000 bp) achieve over 3.1 million windows per second and a 5.57x throughput speedup."*
- **Slide 6: Conclusion**
  - *"GeneWeaver bridges rigorous biological validation with high-performance computing concepts, backed by 33 passing automated tests and a production-grade Streamlit web interface."*

---

### 3. Top 10 Technical Viva Questions & Answers

**Q1: What is a PAM motif and why is it essential for CRISPR-Cas9?**  
**A:** PAM stands for Protospacer Adjacent Motif. For SpCas9, the canonical PAM is $5'\text{-NGG-}3'$ located immediately 3' of the target protospacer on the non-target DNA strand. Cas9 requires PAM binding to trigger local DNA duplex unwinding and allow the sgRNA to interrogate the target sequence. Without a compatible PAM, Cas9 will not cleave DNA.

**Q2: What is the "Seed Region" in CRISPR guide alignment?**  
**A:** The seed region comprises the 8 to 12 nucleotides directly adjacent to the PAM (positions 11–20 of a 20 nt guide). Because RNA-DNA heteroduplex nucleation begins here, mismatches in the seed region severely destabilize Cas9 binding and cleavage. GeneWeaver assigns seed mismatches a 2.5x penalty weight compared to distal mismatches.

**Q3: How does GeneWeaver search the reverse (antisense) strand?**  
**A:** In double-stranded DNA, Cas9 can target either strand. GeneWeaver generates the reverse complement of the reference sequence ($A \leftrightarrow T, C \leftrightarrow G$ reversed). After scanning this strand, it maps the coordinates back to the forward reference using:
$$\text{Start}_{\text{forward}} = N - (i_{rc} + W) + 1, \quad \text{End}_{\text{forward}} = N - i_{rc}$$

**Q4: How does PyTorch accelerate sliding window alignment without a neural network?**  
**A:** We use PyTorch as a high-performance numerical tensor library. We encode DNA bases into integers ($A=0, C=1, G=2, T=3$). We then call `tensor.unfold(dimension=0, size=W, step=1)`, which creates a 2D strided view of all candidate windows in memory without copying data. We then perform parallel broadcast comparisons `(windows != guide_tensor)` to count mismatches across all windows simultaneously.

**Q5: What happens when a user runs GeneWeaver on a computer without an NVIDIA GPU?**  
**A:** GeneWeaver checks `torch.cuda.is_available()`. If False, it automatically activates the CPU tensor fallback engine, clearly reporting `CPU (PyTorch Vectorized Fallback)`. The application never crashes and never fakes GPU execution.

**Q6: Why is the CPU sometimes faster than the GPU for very small DNA sequences?**  
**A:** This is due to Amdahl's Law and kernel launch latency. Allocating GPU memory and transferring tensors over the PCIe bus introduces a fixed overhead of ~0.5 to 2.0 ms. For a 500 bp sequence, the computation is so small that the overhead exceeds the compute time. As sequence size grows, the parallel throughput easily overcomes this latency.

**Q7: What is the difference between Hamming distance and Levenshtein distance?**  
**A:** Hamming distance only measures substitution mismatches between strings of equal length. Levenshtein distance also accounts for insertions and deletions (indels). GeneWeaver currently uses sliding-window Hamming distance matching, as CRISPR off-target screening predominantly evaluates base substitutions.

**Q8: What are IUPAC degenerate nucleotide codes?**  
**A:** IUPAC codes represent nucleotide ambiguities. For example, `N` means any base (A, C, G, T), `R` means purines (A or G), and `Y` means pyrimidines (C or T). GeneWeaver supports all 15 IUPAC codes to allow scanning for engineered Cas variants like SaCas9 (`NNGRRT`).

**Q9: Does finding a computational sequence match prove that CRISPR will cut that genomic location?**  
**A:** No. Computational alignment only indicates sequence homology. In real cells, Cas9 cutting depends on chromatin accessibility (open vs closed heterochromatin), DNA methylation, nucleosome positioning, and local thermodynamic free energies ($\Delta G$).

**Q10: How are tests organized and executed?**  
**A:** Tests are organized into four dedicated test modules in `tests/` covering validation, alignment algorithms, CPU/GPU equivalence, and benchmarking. They are executed automatically using `pytest tests/ -v`.

---

### 4. Common Mentor Questions & Strategic Answers

- **Mentor:** *"Why did you use PyTorch instead of writing custom CUDA C++ kernels?"*  
  - **Answer:** *"Using PyTorch allows GeneWeaver to remain portable, easy to install via pip on Windows without requiring MSVC or the full CUDA Toolkit compiler, and automatically supports vectorized CPU fallback when a GPU is unavailable."*
- **Mentor:** *"Can this tool handle a full 3-billion-base human genome?"*  
  - **Answer:** *"GeneWeaver is designed for locus-scale and chromosome-scale analysis up to hundreds of thousands of base pairs. For full 3-gigabase genomes, unrolling the whole genome into RAM would exceed memory limits. Production whole-genome aligners use FM-index or Burrows-Wheeler Transform (BWT) indexing, which represents an exciting future extension for GeneWeaver."*
- **Mentor:** *"How did you ensure your test cases are biologically meaningful?"*  
  - **Answer:** *"Our pre-packaged sample dataset contains real, documented CRISPR target sites from foundational literature, including the canonical human EMX1 locus (Cong et al., Science 2013) and VEGFA locus (Fu et al., Nature Biotech 2013), alongside calibrated mismatch variants."*

---

## ⚠️ Scientific Limitations & Caveats

GeneWeaver is an in silico computational sequence-matching and educational profiling engine.
- A computational sequence match **does not guarantee** biological cutting efficiency or clinical off-target risk.
- True biological editing is modulated by epigenetic factors, chromatin structure, and cell-cycle dynamics.
- GeneWeaver results are intended for research, education, and candidate screening.

---

## 📄 License & Acknowledgments

This project is licensed under the **MIT License**.  
Developed during an individual bioinformatics and software engineering internship at **Infotact Solutions**.

