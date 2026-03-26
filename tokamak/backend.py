"""
Compute backend abstraction — CUDA (CuPy) or CPU (NumPy).

Usage:
    from tokamak.backend import xp, is_cuda, to_host
"""

import sys
import numpy as np

# ── Detect CUDA availability ──
_backend_name = "cpu"
_use_cuda = False
xp = np  # default

def init_backend(requested: str = "auto"):
    """
    Initialize compute backend.
    
    Args:
        requested: 'cuda', 'cpu', or 'auto' (try CUDA, fall back to CPU).
    """
    global xp, _use_cuda, _backend_name

    if requested in ("cuda", "auto"):
        try:
            import cupy as cp
            # Quick sanity check
            _ = cp.zeros(1)
            xp = cp
            _use_cuda = True
            _backend_name = "cuda"
            dev = cp.cuda.Device()
            mem_total = dev.mem_info[1] / 1e9
            try:
                dev_name = cp.cuda.runtime.getDeviceProperties(dev.id)['name']
                if isinstance(dev_name, bytes):
                    dev_name = dev_name.decode()
            except Exception:
                dev_name = f"GPU:{dev.id}"
            print(f"  [BACKEND] CUDA enabled — {dev_name} "
                  f"({mem_total:.1f} GB)")
            return
        except Exception as e:
            if requested == "cuda":
                print(f"  [BACKEND] CUDA requested but failed: {e}")
                sys.exit(1)
            # auto mode: fall through to CPU

    xp = np
    _use_cuda = False
    _backend_name = "cpu"
    print("  [BACKEND] CPU mode (NumPy + Numba)")


def is_cuda() -> bool:
    """Return True if CUDA backend is active."""
    return _use_cuda


def to_host(arr):
    """Move array to host (CPU). No-op if already on CPU."""
    if _use_cuda:
        import cupy as cp
        if isinstance(arr, cp.ndarray):
            return arr.get()
    return np.asarray(arr)


def to_device(arr):
    """Move array to device (GPU). No-op if on CPU backend."""
    if _use_cuda:
        import cupy as cp
        return cp.asarray(arr)
    return arr


def sync():
    """Synchronize GPU. No-op on CPU."""
    if _use_cuda:
        import cupy as cp
        cp.cuda.stream.get_current_stream().synchronize()
