# Central runtime tweaks to reduce noisy warnings and improve stability
try:  # pragma: no cover - environment dependent
    import os
    # Hint OpenBLAS to use a single thread to avoid nested threadpools and warnings at import time
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    from threadpoolctl import threadpool_limits
    # Ensure BLAS libraries (e.g., OpenBLAS) run single-threaded to avoid nested pools
    threadpool_limits(1, "blas")
except Exception:
    pass


