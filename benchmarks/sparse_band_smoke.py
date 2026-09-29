"""Manual smoke benchmark for the O(K) sparse banded solver.

Run from the repository root:

    python benchmarks/sparse_band_smoke.py

This is intentionally not part of the pytest suite.  Its purpose is to catch
accidental dense allocations and to report elapsed time against the number K of
allowed cells.
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hellinger_etw_sparse import RowBand, etw_align_sparse_banded


def diagonal_band(n: int, radius: int) -> RowBand:
    row = np.arange(n, dtype=np.int64)
    return RowBand(np.maximum(0, row - radius), np.minimum(n, row + radius + 1), n)


def main() -> None:
    # Warm Numba before measuring.
    warm = RowBand.full(2, 2)
    etw_align_sparse_banded(
        [0, 1],
        [0.0, 0.5, 1.0],
        [0, 1],
        [0.0, 0.5, 1.0],
        band=warm,
        similarity_values=np.ones(4),
        use_numba=True,
    )

    print("n\tradius\tK\tseconds\tK/second")
    for n in (1_000, 2_000, 4_000, 8_000, 16_000, 32_000):
        radius = 2
        band = diagonal_band(n, radius)
        similarities = np.ones(band.allowed_cells, dtype=np.float64)
        times = np.linspace(0.0, 1.0, n + 1)
        start = time.perf_counter()
        result = etw_align_sparse_banded(
            np.arange(n),
            times,
            np.arange(n),
            times,
            band=band,
            similarity_values=similarities,
            use_numba=True,
        )
        elapsed = time.perf_counter() - start
        rate = band.allowed_cells / elapsed
        print(f"{n}\t{radius}\t{band.allowed_cells}\t{elapsed:.6f}\t{rate:.0f}")
        assert result.diagnostics.state_cells == band.allowed_cells


if __name__ == "__main__":
    main()
