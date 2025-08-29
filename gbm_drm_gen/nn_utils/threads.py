import os

# simple idempotent guard so we don’t set twice
_SET = False

def configure_threads(n_intra=16, n_interop=1,
                      env_openblas=True, env_mkl=True, env_omp=True, env_numexpr=True):
    global _SET
    if _SET:
        return
    # Set env vars before importing torch or numpy (best-effort if called late)
    if env_omp:
        os.environ.setdefault("OMP_NUM_THREADS", str(n_intra))
    if env_mkl:
        os.environ.setdefault("MKL_NUM_THREADS", str(n_intra))
    if env_openblas:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", str(n_intra))
    if env_numexpr:
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(n_intra)
        torch.set_num_interop_threads(n_interop)
    except Exception:
        # If torch was already in parallel work, we skip; call this earlier next time
        pass
    _SET = True