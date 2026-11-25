\# Day 9 – CRISPR Alignment Workflow Audit



\## 1. Objective

Reviewed the CRISPR Alignment workflow and CPU–GPU alignment equivalence tests to identify gaps in test coverage.



\## 2. Areas Reviewed

\- CRISPR Alignment Streamlit page

\- CPU alignment engine

\- GPU/PyTorch alignment engine

\- Forward and reverse-strand matching

\- IUPAC PAM pattern matching

\- CPU–GPU result equivalence tests

\- Ambiguous and non-canonical reference characters



\## 3. Issue Identified

The test `test\_cpu\_gpu\_non\_canonical\_and\_ambiguous\_reference\_characters()` claimed to test both `N` and `X`, but its reference sequence contained only `N`.



\## 4. Improvement Implemented

\- Updated the test reference sequence to include both `N` and `X`.

\- Increased the allowed mismatch threshold from 1 to 2.

\- Updated mismatch-count assertions.

\- Added assertions to verify mismatch positions `\[1, 6]` for both CPU and GPU engines.



\## 5. Test Results

\- Targeted CPU–GPU tests: 9 passed.

\- Complete project test suite: 74 passed.

\- No test failures observed.



\## 6. Conclusion

Improved CPU–GPU parity test coverage for ambiguous and non-canonical reference characters. Both engines successfully produced consistent mismatch results.



\*\*Day 9 Status: Completed – Verified\*\*

