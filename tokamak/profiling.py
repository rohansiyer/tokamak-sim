"""
GPU/CPU profiling utilities for simulation phases.

Provides PhaseTimer context manager for timing and memory tracking,
plus a Profiler class to collect and report results.
"""

import time
import numpy as np

from tokamak.backend import is_cuda


class PhaseTimer:
    """
    Context manager that records elapsed time and GPU memory for a phase.

    Uses cupy.cuda.Event for GPU timing when CUDA is active,
    falls back to time.perf_counter for CPU.
    """

    def __init__(self, name: str, profiler: 'Profiler'):
        self.name = name
        self.profiler = profiler
        self._start_time = None
        self._start_mem = None
        self._gpu_start = None
        self._gpu_end = None

    def __enter__(self):
        if is_cuda():
            import cupy as cp
            cp.cuda.stream.get_current_stream().synchronize()
            self._gpu_start = cp.cuda.Event()
            self._gpu_end = cp.cuda.Event()
            dev = cp.cuda.Device()
            self._start_mem = dev.mem_info[0]  # free memory
            self._gpu_start.record()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed_wall = time.perf_counter() - self._start_time
        gpu_time = None
        mem_used_mb = None

        if is_cuda():
            import cupy as cp
            self._gpu_end.record()
            self._gpu_end.synchronize()
            gpu_time = cp.cuda.get_elapsed_time(
                self._gpu_start, self._gpu_end) / 1000.0  # ms → s

            dev = cp.cuda.Device()
            free_now = dev.mem_info[0]
            mem_used_mb = (self._start_mem - free_now) / 1e6  # bytes → MB

        self.profiler._record(self.name, elapsed_wall, gpu_time, mem_used_mb)
        return False


class Profiler:
    """Collects PhaseTimer results and prints a formatted report."""

    def __init__(self):
        self.phases: list[dict] = []

    def phase(self, name: str) -> PhaseTimer:
        """Create a PhaseTimer context manager for the named phase."""
        return PhaseTimer(name, self)

    def _record(self, name, wall_time, gpu_time, mem_delta_mb):
        self.phases.append({
            'name': name,
            'wall': wall_time,
            'gpu': gpu_time,
            'mem_mb': mem_delta_mb,
        })

    def report(self):
        """Print a formatted profiling report."""
        cuda = is_cuda()

        total_wall = sum(p['wall'] for p in self.phases)
        total_gpu = sum(p['gpu'] for p in self.phases if p['gpu'] is not None)

        print()
        print("━━━ PROFILING REPORT ━━━")

        # Header
        if cuda:
            print(f"  {'Phase':<35} {'Wall [s]':>9} {'GPU [s]':>9} "
                  f"{'VRAM Δ [MB]':>12} {'% Total':>8}")
            print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*12} {'─'*8}")
        else:
            print(f"  {'Phase':<35} {'Wall [s]':>9} {'% Total':>8}")
            print(f"  {'─'*35} {'─'*9} {'─'*8}")

        for p in self.phases:
            pct = 100.0 * p['wall'] / max(total_wall, 1e-9)
            if cuda:
                gpu_str = f"{p['gpu']:9.3f}" if p['gpu'] is not None else "      N/A"
                mem_str = (f"{p['mem_mb']:+10.1f} MB"
                           if p['mem_mb'] is not None else "         N/A")
                print(f"  {p['name']:<35} {p['wall']:9.3f} {gpu_str} "
                      f"{mem_str:>12} {pct:7.1f}%")
            else:
                print(f"  {p['name']:<35} {p['wall']:9.3f} {pct:7.1f}%")

        # Totals
        if cuda:
            print(f"  {'─'*35} {'─'*9} {'─'*9} {'─'*12} {'─'*8}")
            print(f"  {'TOTAL':<35} {total_wall:9.3f} {total_gpu:9.3f} "
                  f"{'':>12} {'100.0%':>8}")

            # GPU memory summary
            import cupy as cp
            dev = cp.cuda.Device()
            free, total = dev.mem_info
            used = total - free
            print(f"\n  GPU Memory: {used/1e9:.2f} / {total/1e9:.2f} GB "
                  f"({100*used/total:.1f}% utilised)")
        else:
            print(f"  {'─'*35} {'─'*9} {'─'*8}")
            print(f"  {'TOTAL':<35} {total_wall:9.3f} {'100.0%':>8}")
