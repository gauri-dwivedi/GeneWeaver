# GeneWeaver User Guide & Manual

Welcome to **GeneWeaver**, an interactive bioinformatics engine for CRISPR-Cas9 target site scanning, off-target mismatch calculation, and CPU vs GPU performance profiling.

---

## 1. Quick Start Installation (Windows)

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- PowerShell or Command Prompt

### Step 1: Clone or Navigate to the Project Directory
```powershell
cd "C:\Users\Gauri Dwivedi\Documents\GeneWeaver"
```

### Step 2: Activate the Virtual Environment
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell displays an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### Step 3: Verify Dependencies
```powershell
python -m pip install -r requirements.txt
```

### Step 4: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
The application will launch in your default web browser at `http://localhost:8501`.

---

## 2. Navigating the Application

### 🏠 Home / Overview (`app.py`)
- Provides high-level project summary and system metrics.
- Displays the active processing hardware detected (CUDA GPU or CPU Fallback).
- Contains interactive tabs for Project Overview, Architecture Diagram, and Evaluation Guide.

### 🔬 Page 1: DNA Sequence Analyzer (`pages/1_Sequence_Analyzer.py`)
1. **Choose Input Mode**:
   - *Built-in Sample Reference*: Select from pre-packaged synthetic *EMX1*, *VEGFA*, or repeat arrays.
   - *Paste DNA / FASTA Text*: Paste raw sequence text or multi-FASTA strings.
   - *Upload FASTA File*: Drag and drop `.fasta`, `.fa`, or `.fna` files.
2. **Review Validation Status**:
   - Checks for non-canonical characters (flags exact character and index).
   - Computes sequence length, GC content (%), AT content (%), and purine/pyrimidine ratios.
3. **Explore Visualizations**:
   - View color-coded nucleotide sequences (A=Green, C=Blue, G=Yellow, T=Red).
   - Inspect base composition bar and pie charts.
   - Review 5'->3' reverse complement and mRNA transcription views.
4. **Send to Alignment**: Click *"🎯 Use This Sequence for CRISPR Alignment"* to store the sequence in active session state.

### 🎯 Page 2: CRISPR Target Alignment (`pages/2_CRISPR_Alignment.py`)
1. **Configure Guide RNA**:
   - Choose a preset guide (e.g. `G01_EMX1_Exact`) or enter a custom 20 bp sequence.
2. **Set PAM Pattern**:
   - Default is `NGG` (SpCas9). Supports any IUPAC ambiguity code (e.g. `NAG`, `NNGRRT`).
3. **Set Thresholds & Engine**:
   - Configure mismatch cutoff (0 to 6 mismatches).
   - Choose compute engine: **GPU (PyTorch Vectorized Batching)** or **CPU (Native Reference)**.
4. **Execute**: Click *"🚀 Execute CRISPR Alignment"*.
5. **Analyze Results**:
   - View latency, throughput, and target count metrics.
   - Sort and filter the candidate target table by strand, mismatches, or score.
   - Select any candidate target to view the **3-line visual alignment breakdown** (`||||||.||||||`).
6. **Download Results**: Export candidate matches as CSV.

### ⚡ Page 3: CPU vs GPU Benchmarking (`pages/3_Performance_Benchmark.py`)
1. **Single Sequence Direct Comparison**:
   - Select sequence size (500 to 50,000 bp).
   - Runs CPU and GPU engines on identical inputs.
   - Displays latency (ms), throughput (win/s), and speedup factor.
2. **Multi-Scale Scaling Suite**:
   - Select multiple sequence lengths (e.g. 500, 2000, 5000, 15000 bp).
   - Click *"⚡ Run Multi-Scale Scaling Benchmark"*.
   - View interactive Plotly scaling curves: Latency Scaling, Speedup Curve, and Throughput Bar charts.
   - Download benchmark dataset as CSV.

### 📁 Page 4: Dataset Explorer (`pages/4_Dataset_Explorer.py`)
- Inspect the contents of `data/sample_reference.fasta` and `data/sample_guides.csv`.
- Review the complete **Data Dictionary**.
- Study literature provenance (Cong et al. 2013, Fu et al. 2013).

### 📥 Page 5: Results and Downloads (`pages/5_Results.py`)
- Browse history of alignment executions during the current session.
- Preview and download comprehensive Markdown evaluation reports.
- Inspect and download previously exported CSV files stored on disk in `results/`.

---

## 3. Running Automated Tests

To run the complete test suite:
```powershell
.\.venv\Scripts\pytest tests/ -v
```
All 33 unit and integration tests should pass.
