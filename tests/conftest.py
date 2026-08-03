"""Shared pytest bootstrap for the tennis-ml test suite.

Sets KMP_DUPLICATE_LIB_OK so torch and faiss can coexist in one test process.

On macOS both torch and faiss-cpu ship their own libomp (LLVM OpenMP runtime).
The first to initialize wins; the second aborts the interpreter with
"OMP: Error #15". The fast suite imports torch (test_nn.py) and faiss
(test_similarity.py) in the same process, so faiss's first native search
crashes the whole run unless this env var is set. This is the documented
PyTorch workaround for exactly this duplicate-libomp situation.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
