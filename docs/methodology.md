# GeneWeaver Methodology & Biological Algorithms

## 1. Biological Foundations of CRISPR-Cas9 Target Recognition

Clustered Regularly Interspaced Short Palindromic Repeats (CRISPR) and the Cas9 endonuclease form an adaptive bacterial immune system widely adapted for precision genome engineering.

### Target Binding Mechanism
1. **PAM Interrogation**: The Cas9-sgRNA ribonucleoprotein (RNP) complex surveys genomic DNA for a short adjacent sequence termed the **Protospacer Adjacent Motif (PAM)**. For *Streptococcus pyogenes* Cas9 (SpCas9), this motif is canonical $5'\text{-NGG-}3'$ (where N is any nucleotide).
2. **Local DNA Unwinding**: Upon PAM recognition, Cas9 destabilizes the DNA duplex directly adjacent to the PAM.
3. **RNA-DNA Heteroduplex Hybridization**: The single-guide RNA (sgRNA) spacer (typically 20 nt) hybridizes to the target DNA strand through Watson-Crick base pairing.
4. **Seed Region Sensitivity**: Hybridization initiates at the PAM-proximal bases (nucleotides 11–20 of the spacer, termed the **Seed Region**). Mismatches within this 8–10 bp seed drastically impair Cas9 binding and cleavage. Conversely, mismatches located in the PAM-distal region (nucleotides 1–10) are frequently tolerated, resulting in potential **off-target cleavage**.

---

## 2. Sliding Window Alignment Algorithm

Given a reference DNA sequence $R$ of length $N$, a guide RNA $G$ of length $L$ (default 20 bp), and a PAM pattern $P$ of length $K$ (default 3 bp for `NGG`):

### Forward Strand Search
The algorithm evaluates a sliding window of width $W = L + K$ across all valid start indices $i \in [0, N - W]$:
$$\text{Window}_i = R[i : i + W]$$
$$\text{Target}_i = R[i : i + L], \quad \text{PAM}_i = R[i + L : i + W]$$

1. **PAM Check**: Check if $\text{PAM}_i$ matches pattern $P$ using IUPAC degenerate nucleotide definitions.
2. **Mismatch Counting**: If PAM matches, calculate character-by-character mismatches against guide $G$:
   $$\text{Mismatches} = \sum_{j=0}^{L-1} \mathbb{I}(G[j] \neq \text{Target}_i[j])$$
3. **Thresholding**: If $\text{Mismatches} \le \text{max\_mismatches}$, record candidate match at 1-based biological coordinates $[i + 1, i + W]$.

### Reverse-Complement (Antisense) Strand Search
In double-stranded DNA, Cas9 can target the negative strand. GeneWeaver generates the reverse complement $R_{rc} = \text{reverse\_complement}(R)$ and slides the exact same window logic across $R_{rc}$.

For each qualifying match found at index $i_{rc}$ in $R_{rc}$, biological coordinates are mapped back to the forward reference frame:
$$\text{Start}_{\text{forward}} = N - (i_{rc} + W) + 1$$
$$\text{End}_{\text{forward}} = N - i_{rc}$$

---

## 3. Seed-Weighted Alignment Specificity Scoring

To reflect Cas9 biophysics, GeneWeaver weights mismatches according to their position relative to the PAM:
- **Distal Region (Positions 1 to 10)**: Weight $w_{\text{distal}} = 1.0$
- **Seed Region (Positions 11 to 20)**: Weight $w_{\text{seed}} = 2.5$

The penalty metric $P$ is computed as:
$$P = (N_{\text{distal\_mm}} \times 1.0) + (N_{\text{seed\_mm}} \times 2.5)$$

Normalized specificity score $S \in [0.0, 100.0]$:
$$S = 100.0 \times \max\left(0.0, 1.0 - \frac{P}{L \times 1.75}\right)$$

*For a perfect match ($P = 0$), $S = 100.0$. A target with seed mutations is penalized 2.5x more heavily than one with distal mutations.*

---

## 4. GPU Tensor Parallelization Architecture

### Integer Encoding
DNA characters are mapped to integer values:
$$A \mapsto 0, \quad C \mapsto 1, \quad G \mapsto 2, \quad T \mapsto 3$$
The guide sequence is converted into a 1D tensor $\mathbf{g} \in \mathbb{Z}^L$.

### Zero-Copy Sliding Window Unfolding
Using PyTorch's `unfold()` method, the reference sequence tensor $\mathbf{r} \in \mathbb{Z}^N$ is partitioned into all overlapping windows of size $W$:
$$\mathbf{W} = \text{unfold}(\mathbf{r}, \text{dimension}=0, \text{size}=W, \text{step}=1) \in \mathbb{Z}^{M \times W}$$
where $M = N - W + 1$. This operation creates a strided matrix view without copying raw memory.

### Vectorized Matrix Arithmetic
1. **Batch PAM Filtering**: Slices $\mathbf{W}[:, L:W]$ and executes parallel boolean mask filtering on CUDA cores.
2. **Batch Difference Tensor**:
   $$\mathbf{D} = (\mathbf{W}_{\text{valid}}[:, :L] \neq \mathbf{g}) \in \{0, 1\}^{B \times L}$$
3. **Reduction**:
   $$\mathbf{m} = \sum_{k=0}^{L-1} \mathbf{D}[:, k] \in \mathbb{Z}^B$$
   All windows satisfying $\mathbf{m} \le \text{max\_mismatches}$ are returned in a single batch operation.

---

## 5. Benchmarking & Latency Measurement

- **High-Precision Timers**: All latency measurements utilize `time.perf_counter()`, offering sub-microsecond precision on Windows.
- **CUDA Synchronization**: Calls `torch.cuda.synchronize()` before starting and stopping timers to prevent asynchronous queue skew.
- **Warmup Iteration**: Runs a discardable initial pass to eliminate cold-start GPU kernel compilation and memory pool overhead.
- **Throughput Formula**:
  $$\text{Throughput} = \frac{\text{Total Windows Evaluated}}{\text{Elapsed Time (seconds)}} \quad [\text{windows/sec}]$$
- **Speedup Ratio**:
  $$\text{Speedup} = \frac{T_{\text{CPU Latency}}}{T_{\text{GPU Latency}}}$$

