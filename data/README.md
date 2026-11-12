# GeneWeaver Sample Datasets & Data Dictionary

This directory contains standardized sample datasets packaged with GeneWeaver to ensure immediate, zero-configuration usability and reproducible testing.

> [!NOTE]
> All reference loci in `sample_reference.fasta` are **synthetic reference sequences** inspired by well-characterized human genomic loci (such as human *EMX1* and *VEGFA* loci frequently used in CRISPR-Cas9 literature like Cong et al. 2013 and Fu et al. 2013). They are clearly marked with `[organism=synthetic]`.

---

## 1. File Descriptions

### `sample_reference.fasta`
Contains three multi-FASTA reference sequences formatted for alignment and benchmarking:
1. **`synthetic_ref_01_emx1_locus`** (~840 bp): Modeled after the human *EMX1* exon 3 region. Contains the canonical `GAGTCCGAGCAGAAGAAGAA GGG` SpCas9 target site.
2. **`synthetic_ref_02_vegfa_locus`** (~600 bp): Modeled after the human *VEGFA* promoter region, featuring the GC-rich target `GGGTGGGGGGAGTTTGCTCC TGG`.
3. **`synthetic_ref_03_benchmark_array`** (~1080 bp): An engineered array containing multiple forward and reverse target sites with 0, 1, and 3 mismatches specifically calibrated for performance benchmarking.

### `sample_guides.csv`
A curated library of 8 guide RNAs covering diverse test scenarios:
- **G01_EMX1_Exact**: Perfect exact match (0 mismatches) on forward strand.
- **G02_EMX1_1MM**: Single mismatch off-target test.
- **G03_EMX1_3MM**: Multiple mismatch (3 mismatches) off-target test.
- **G04_VEGFA_Target**: High-GC target sequence.
- **G05_ReverseStrand**: Tests antisense / reverse-complement strand detection.
- **G06_NonMatching**: Negative control (homopolymer C-tract, 0 matches).
- **G07_Invalid_Chars**: Negative validation test containing IUPAC `N` characters.
- **G08_Short_Length**: Negative validation test for length threshold sanity check.

---

## 2. Data Dictionary (`sample_guides.csv`)

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `guide_id` | String | Unique guide sequence identifier (e.g., G01_EMX1_Exact). |
| `sequence` | String | 5' to 3' nucleotide sequence of the CRISPR guide (crRNA spacer). |
| `target_locus` | String | ID of the reference sequence in `sample_reference.fasta` containing the target. |
| `expected_strand` | Categorical | Expected matching strand: `forward`, `reverse`, or `none`. |
| `expected_mismatches` | Integer | Expected number of base mismatches against target (99 for non-matching). |
| `pam_motif` | String | Protospacer Adjacent Motif required for Cas nuclease recognition (default: `NGG`). |
| `notes` | String | Biological context, publication reference, or testing purpose. |

---

## 3. Scientific Caveat and Limitations

GeneWeaver performs **sequence-matching and mismatch counting**. It is a computational alignment tool:
- A computational sequence match **does not** prove experimental cleavage or editing efficiency.
- Biological editing is governed by chromatin accessibility, nucleosome positioning, DNA methylation, cellular Cas9 concentration, and complex thermodynamic free energies ($\Delta G$).
- GeneWeaver results indicate candidate homology sites, not verified biological outcomes.

