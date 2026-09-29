"""Structural and quality smoke benchmark for coarse-to-fine ETW.

This is not a publication benchmark.  It checks that the number of evaluated
pair cells grows roughly linearly for a fixed corridor width and records the
score gap to the unrestricted exact algorithm at moderate sizes.

Run from the repository root:

    python benchmarks/multiscale_infill_smoke.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hellinger_etw import etw_align as etw_align_dense
from hellinger_etw_multiscale import etw_align_multiscale
from simulate_etw_data import rbf_similarity_matrix, simulate_pair


def row_rbf(sigma: float):
    def similarity(x, ys) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        diff = y_arr - x_arr
        return np.exp(-0.5 * np.sum(diff * diff, axis=-1) / (sigma * sigma))

    return similarity


def run(args: argparse.Namespace) -> list[dict[str, object]]:
    # Warm Numba before timing.
    warm = simulate_pair(32, 32, seed=1, noise=args.noise, warp_strength=args.warp)
    etw_align_multiscale(
        warm.f_values,
        warm.f_times,
        warm.g_values,
        warm.g_times,
        row_similarity=row_rbf(args.sigma),
        coarsest_size=16,
        initial_radius=args.radius,
        adaptive=False,
        use_numba=True,
    )

    rows: list[dict[str, object]] = []
    for n in args.sizes:
        pair = simulate_pair(
            n,
            n,
            seed=args.seed + n,
            noise=args.noise,
            warp_strength=args.warp,
            grid_size=max(2048, 2 * n),
        )

        t0 = perf_counter()
        fixed = etw_align_multiscale(
            pair.f_values,
            pair.f_times,
            pair.g_values,
            pair.g_times,
            row_similarity=row_rbf(args.sigma),
            coarsest_size=args.coarsest_size,
            initial_radius=args.radius,
            adaptive=False,
            use_numba=True,
        )
        fixed_seconds = perf_counter() - t0

        exact_score = np.nan
        exact_seconds = np.nan
        score_gap = np.nan
        if n <= args.exact_through:
            t0 = perf_counter()
            C = rbf_similarity_matrix(pair.f_values, pair.g_values, sigma=args.sigma)
            exact = etw_align_dense(
                pair.f_values,
                pair.f_times,
                pair.g_values,
                pair.g_times,
                similarity_matrix=C,
                use_numba=True,
            )
            exact_seconds = perf_counter() - t0
            exact_score = exact.score
            score_gap = exact.score - fixed.score

        rows.append(
            {
                "n": n,
                "method": "fixed_multiscale",
                "seconds": fixed_seconds,
                "score": fixed.score,
                "exact_score": exact_score,
                "score_gap": score_gap,
                "evaluated_cells": fixed.diagnostics.total_similarity_evaluations,
                "cells_per_n": fixed.diagnostics.total_similarity_evaluations / n,
                "final_band_cells": fixed.final_band.allowed_cells,
                "final_band_width_equivalent": fixed.final_band.allowed_cells / n,
                "exact_seconds": exact_seconds,
            }
        )

        if args.adaptive:
            t0 = perf_counter()
            adaptive = etw_align_multiscale(
                pair.f_values,
                pair.f_times,
                pair.g_values,
                pair.g_times,
                row_similarity=row_rbf(args.sigma),
                coarsest_size=args.coarsest_size,
                initial_radius=max(1, args.radius // 2),
                adaptive=True,
                stability_rtol=args.stability_tolerance,
                stability_atol=args.stability_tolerance,
                max_attempts=args.max_attempts,
                use_numba=True,
            )
            adaptive_seconds = perf_counter() - t0
            finest = next(level for level in adaptive.diagnostics.levels if level.level == 0)
            adaptive_gap = np.nan if np.isnan(exact_score) else exact_score - adaptive.score
            rows.append(
                {
                    "n": n,
                    "method": "adaptive_multiscale",
                    "seconds": adaptive_seconds,
                    "score": adaptive.score,
                    "exact_score": exact_score,
                    "score_gap": adaptive_gap,
                    "evaluated_cells": adaptive.diagnostics.total_similarity_evaluations,
                    "cells_per_n": adaptive.diagnostics.total_similarity_evaluations / n,
                    "final_band_cells": adaptive.final_band.allowed_cells,
                    "final_band_width_equivalent": adaptive.final_band.allowed_cells / n,
                    "exact_seconds": exact_seconds,
                    "accepted_radius": finest.accepted.radius,
                    "stopping_reason": finest.stopping_reason,
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 256, 512, 1024, 2048, 4096])
    parser.add_argument("--exact-through", type=int, default=1024)
    parser.add_argument("--coarsest-size", type=int, default=32)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--warp", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--stability-tolerance", type=float, default=1.0e-4)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = run(args)
    fieldnames = sorted({key for row in rows for key in row})
    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as handle:
            out = csv.DictWriter(handle, fieldnames=fieldnames)
            out.writeheader()
            out.writerows(rows)


if __name__ == "__main__":
    main()
