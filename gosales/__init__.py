# Central runtime tweaks to reduce noisy warnings and improve stability
try:  # pragma: no cover - environment dependent
    from threadpoolctl import threadpool_limits
    # Ensure BLAS libraries (e.g., OpenBLAS) run single-threaded to avoid nested pools
    threadpool_limits(1, "blas")
except Exception:
    pass


