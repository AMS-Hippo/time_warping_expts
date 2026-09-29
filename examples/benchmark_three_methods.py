"""Reproducible benchmark for the three ETW implementations.

The benchmark compares

1. the direct cubic recurrence in :mod:`hellinger_etw_original`;
2. the exact dense quadratic upper-envelope solver in :mod:`hellinger_etw`; and
3. the adaptive multiscale sparse solver in :mod:`hellinger_etw_multiscale`.

Two non-adversarial regimes are included by default:

``smooth_distinctive``
    Moderate time warp and low observation noise.  Coarse paths are informative,
    so multiscale refinement should remain narrow.

``noisy_strong_warp``
    Stronger time warp and substantially more observation noise.  This is still
    generated from one smooth latent curve, but coarse paths are less precise,
    so adaptive refinement generally evaluates wider corridors.

The dense methods are timed end-to-end by adding a measured vectorized dense
similarity-matrix construction time to their DP time.  The multiscale timing is
measured end-to-end directly; it computes only the similarities requested by
its sparse corridors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable, Optional, Sequence

import csv
import gc
import json
import math
import os
import platform
import shutil
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hellinger_etw import etw_align as etw_align_quadratic
from hellinger_etw_multiscale import MultiscaleETWResult, etw_align_multiscale
from hellinger_etw_original import etw_align as etw_align_cubic
from simulate_etw_data import SimulatedPair, rbf_similarity_matrix, simulate_pair


@dataclass(frozen=True)
class BenchmarkRegime:
    """Parameters for one synthetic benchmark regime."""

    name: str
    label: str
    description: str
    noise: float
    warp_strength: float
    sigma: float
    kind: str = "integrated_random_walk"
    min_spacing: float = 0.0


DEFAULT_REGIMES: tuple[BenchmarkRegime, ...] = (
    BenchmarkRegime(
        name="smooth_distinctive",
        label="Smooth, distinctive signal",
        description=(
            "Moderate warp and low noise.  Coarse averages preserve the main "
            "landmarks, so a narrow multiscale corridor should work well."
        ),
        noise=0.02,
        warp_strength=0.50,
        sigma=0.25,
        kind="integrated_random_walk",
        min_spacing=0.0,
    ),
    BenchmarkRegime(
        name="noisy_strong_warp",
        label="Noisy signal with stronger warp",
        description=(
            "The same smooth latent-curve model, but with stronger warp, more "
            "noise, and a sharper similarity kernel.  This is realistic rather "
            "than adversarial, but coarse paths are less precise and adaptive "
            "refinement usually widens."
        ),
        noise=0.08,
        warp_strength=1.20,
        sigma=0.18,
        kind="integrated_random_walk",
        min_spacing=0.0,
    ),
)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration recorded with every benchmark bundle."""

    n_max: int = 1024
    n_min: int = 64
    seed: int = 356
    repeats: int = 3
    cubic_max_n: int = 512
    quadratic_max_n: int = 2048
    coarsest_size: int = 32
    initial_radius: int = 4
    adaptive: bool = True
    stability_tolerance: float = 1.0e-4
    max_attempts: int = 8
    fixed_radius_diagnostic: bool = True
    use_numba: bool = True


@dataclass(frozen=True)
class BenchmarkRun:
    """In-memory output of :func:`run_benchmark`."""

    config: BenchmarkConfig
    regimes: tuple[BenchmarkRegime, ...]
    sizes: tuple[int, ...]
    raw_rows: tuple[dict[str, Any], ...]
    summary_rows: tuple[dict[str, Any], ...]
    scaling_fits: tuple[dict[str, Any], ...]
    largest_instances: dict[str, SimulatedPair]


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def power_of_two_sizes(n_min: int, n_max: int) -> tuple[int, ...]:
    """Return all powers of two from ``n_min`` through ``n_max`` inclusive."""

    if not _is_power_of_two(n_min) or not _is_power_of_two(n_max):
        raise ValueError("n_min and n_max must both be powers of two.")
    if n_min > n_max:
        raise ValueError("n_min cannot exceed n_max.")
    sizes: list[int] = []
    value = n_min
    while value <= n_max:
        sizes.append(value)
        value *= 2
    return tuple(sizes)


def validate_config(config: BenchmarkConfig) -> None:
    """Validate a benchmark configuration before any expensive allocation."""

    power_of_two_sizes(config.n_min, config.n_max)
    if config.n_min < 16:
        raise ValueError("n_min must be at least 16.")
    if config.repeats < 1:
        raise ValueError("repeats must be positive.")
    if config.cubic_max_n < config.n_min:
        raise ValueError("cubic_max_n must be at least n_min.")
    if config.quadratic_max_n < config.n_min:
        raise ValueError("quadratic_max_n must be at least n_min.")
    if not _is_power_of_two(config.coarsest_size):
        raise ValueError("coarsest_size must be a power of two.")
    if config.coarsest_size > config.n_min:
        raise ValueError("coarsest_size cannot exceed n_min.")
    if config.initial_radius < 0:
        raise ValueError("initial_radius must be nonnegative.")
    if config.stability_tolerance < 0:
        raise ValueError("stability_tolerance must be nonnegative.")
    if config.max_attempts < 1:
        raise ValueError("max_attempts must be positive.")


def estimate_dense_peak_gb(n: int, bytes_per_cell: float = 64.0) -> float:
    """Conservative order-of-magnitude memory estimate for a dense solve."""

    return float(bytes_per_cell * n * n / 1.0e9)


def row_rbf(sigma: float):
    """Return the vectorized row-similarity callback used by multiscale ETW."""

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def similarity(x: Any, ys: Sequence[Any]) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        delta = y_arr - x_arr
        return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / (sigma * sigma))

    return similarity


def make_instance(n: int, regime: BenchmarkRegime, *, seed: int) -> SimulatedPair:
    """Generate one deterministic problem from a benchmark regime."""

    # Each size gets an independent, reproducible draw from the same statistical
    # regime.  This avoids making one unusually easy or difficult latent curve
    # determine the entire scaling plot.
    regime_code = sum((index + 1) * ord(char) for index, char in enumerate(regime.name))
    instance_seed = int(seed + 10_000 * regime_code + n)
    return simulate_pair(
        n,
        n,
        kind=regime.kind,
        seed=instance_seed,
        noise=regime.noise,
        warp_strength=regime.warp_strength,
        grid_size=max(2048, 2 * n),
        min_spacing=regime.min_spacing,
    )


def _time_once(function, /, *args, **kwargs):
    gc.collect()
    start = perf_counter()
    value = function(*args, **kwargs)
    return value, perf_counter() - start


def warm_numba(
    regimes: Sequence[BenchmarkRegime] = DEFAULT_REGIMES,
    *,
    seed: int = 356,
    coarsest_size: int = 16,
    initial_radius: int = 2,
) -> None:
    """Compile all kernels before measurements are taken."""

    regime = regimes[0]
    pair = make_instance(32, regime, seed=seed)
    dense = rbf_similarity_matrix(pair.f_values, pair.g_values, sigma=regime.sigma)
    etw_align_cubic(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity_matrix=dense,
        use_numba=True,
    )
    etw_align_quadratic(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity_matrix=dense,
        use_numba=True,
    )
    etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=row_rbf(regime.sigma),
        coarsest_size=coarsest_size,
        initial_radius=initial_radius,
        adaptive=True,
        stability_rtol=1.0e-4,
        stability_atol=1.0e-4,
        max_attempts=4,
        use_numba=True,
    )


def _finest_level(result: MultiscaleETWResult):
    return next(level for level in result.diagnostics.levels if level.level == 0)


def _multiscale_fields(result: MultiscaleETWResult, n: int) -> dict[str, Any]:
    finest = _finest_level(result)
    accepted = finest.accepted
    return {
        "evaluated_cells": int(result.diagnostics.total_similarity_evaluations),
        "evaluated_cell_fraction": float(
            result.diagnostics.total_similarity_evaluations / (n * n)
        ),
        "cells_per_n": float(result.diagnostics.total_similarity_evaluations / n),
        "final_band_cells": int(result.final_band.allowed_cells),
        "final_band_fraction": float(result.final_band.allowed_cells / (n * n)),
        "accepted_radius": int(accepted.radius),
        "finest_attempts": int(len(finest.attempts)),
        "boundary_touches": int(accepted.boundary_touches),
        "stopping_reason": finest.stopping_reason,
        "heuristic_converged": bool(result.diagnostics.heuristic_converged),
        "full_grid": bool(result.diagnostics.final_full_grid),
    }


def _append_method_rows(
    rows: list[dict[str, Any]],
    *,
    regime: BenchmarkRegime,
    n: int,
    method: str,
    method_label: str,
    score: float,
    solve_times: Sequence[float],
    similarity_seconds: float,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    extra = {} if extra is None else dict(extra)
    for repeat, solve_seconds in enumerate(solve_times):
        row: dict[str, Any] = {
            "regime": regime.name,
            "regime_label": regime.label,
            "n": int(n),
            "repeat": int(repeat),
            "method": method,
            "method_label": method_label,
            "similarity_seconds": float(similarity_seconds),
            "solve_seconds": float(solve_seconds),
            "total_seconds": float(similarity_seconds + solve_seconds),
            "score": float(score),
        }
        row.update(extra)
        rows.append(row)


def _run_repeated(function, repeats: int):
    times: list[float] = []
    result = None
    for _ in range(repeats):
        result, elapsed = _time_once(function)
        times.append(float(elapsed))
    assert result is not None
    return result, times


def _quantile(values: Sequence[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, q))


def summarize_rows(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Aggregate repeat-level rows into medians and quartiles."""

    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("method") == "fixed_radius_diagnostic":
            continue
        key = (str(row["regime"]), int(row["n"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    summary: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        regime, n, method = key
        total = [float(row["total_seconds"]) for row in group]
        solve = [float(row["solve_seconds"]) for row in group]
        first = group[0]
        out: dict[str, Any] = {
            "regime": regime,
            "regime_label": first["regime_label"],
            "n": n,
            "method": method,
            "method_label": first["method_label"],
            "repeats": len(group),
            "median_total_seconds": float(np.median(total)),
            "q25_total_seconds": _quantile(total, 0.25),
            "q75_total_seconds": _quantile(total, 0.75),
            "median_solve_seconds": float(np.median(solve)),
            "score": float(first["score"]),
        }
        for field in (
            "score_gap_to_exact",
            "relative_score_gap",
            "evaluated_cells",
            "evaluated_cell_fraction",
            "cells_per_n",
            "final_band_cells",
            "final_band_fraction",
            "accepted_radius",
            "finest_attempts",
            "boundary_touches",
            "heuristic_converged",
            "full_grid",
            "stopping_reason",
        ):
            if field in first and first[field] not in (None, ""):
                out[field] = first[field]
        summary.append(out)
    return tuple(summary)


def fit_scaling_exponents(
    summary_rows: Sequence[dict[str, Any]], *, minimum_points: int = 3
) -> tuple[dict[str, Any], ...]:
    """Fit log-log slopes for the available timing curves."""

    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        key = (str(row["regime"]), str(row["method"]))
        groups.setdefault(key, []).append(row)

    fits: list[dict[str, Any]] = []
    for (regime, method), group in sorted(groups.items()):
        group = sorted(group, key=lambda row: int(row["n"]))
        if len(group) < minimum_points:
            continue
        # Use at most the last four points to emphasize the observed scaling at
        # the larger benchmark sizes rather than small-size overheads.
        tail = group[-4:]
        x = np.log(np.asarray([row["n"] for row in tail], dtype=np.float64))
        y = np.log(
            np.asarray([row["median_total_seconds"] for row in tail], dtype=np.float64)
        )
        slope, intercept = np.polyfit(x, y, 1)
        fits.append(
            {
                "regime": regime,
                "regime_label": tail[0]["regime_label"],
                "method": method,
                "method_label": tail[0]["method_label"],
                "points": len(tail),
                "n_min": int(tail[0]["n"]),
                "n_max": int(tail[-1]["n"]),
                "slope": float(slope),
                "intercept": float(intercept),
            }
        )
    return tuple(fits)


def run_benchmark(
    config: BenchmarkConfig,
    regimes: Sequence[BenchmarkRegime] = DEFAULT_REGIMES,
    *,
    progress: bool = True,
) -> BenchmarkRun:
    """Benchmark the cubic, quadratic, and adaptive multiscale methods."""

    validate_config(config)
    regimes_tuple = tuple(regimes)
    sizes = power_of_two_sizes(config.n_min, config.n_max)
    rows: list[dict[str, Any]] = []
    largest_instances: dict[str, SimulatedPair] = {}

    for regime in regimes_tuple:
        if progress:
            print(f"\n{regime.label}")
            print("-" * len(regime.label))
        for n in sizes:
            pair = make_instance(n, regime, seed=config.seed)
            if n == config.n_max:
                largest_instances[regime.name] = pair

            dense = None
            similarity_seconds = math.nan
            exact_score: Optional[float] = None

            need_dense = n <= max(config.cubic_max_n, config.quadratic_max_n)
            if need_dense:
                dense, similarity_seconds = _time_once(
                    rbf_similarity_matrix,
                    pair.f_values,
                    pair.g_values,
                    sigma=regime.sigma,
                )

            if n <= config.cubic_max_n:
                assert dense is not None

                def cubic_call():
                    return etw_align_cubic(
                        pair.f_values,
                        pair.f_times,
                        pair.g_values,
                        pair.g_times,
                        similarity_matrix=dense,
                        use_numba=config.use_numba,
                    )

                cubic_result, cubic_times = _run_repeated(cubic_call, config.repeats)
                _append_method_rows(
                    rows,
                    regime=regime,
                    n=n,
                    method="cubic",
                    method_label="Direct cubic",
                    score=cubic_result.score,
                    solve_times=cubic_times,
                    similarity_seconds=similarity_seconds,
                )

            if n <= config.quadratic_max_n:
                assert dense is not None

                def quadratic_call():
                    return etw_align_quadratic(
                        pair.f_values,
                        pair.f_times,
                        pair.g_values,
                        pair.g_times,
                        similarity_matrix=dense,
                        use_numba=config.use_numba,
                    )

                quadratic_result, quadratic_times = _run_repeated(
                    quadratic_call, config.repeats
                )
                exact_score = float(quadratic_result.score)
                _append_method_rows(
                    rows,
                    regime=regime,
                    n=n,
                    method="quadratic",
                    method_label="Exact quadratic",
                    score=quadratic_result.score,
                    solve_times=quadratic_times,
                    similarity_seconds=similarity_seconds,
                )

                if n <= config.cubic_max_n:
                    cubic_score = next(
                        float(row["score"])
                        for row in reversed(rows)
                        if row["regime"] == regime.name
                        and row["n"] == n
                        and row["method"] == "cubic"
                    )
                    if not np.isclose(cubic_score, exact_score, rtol=2e-11, atol=2e-12):
                        raise AssertionError(
                            f"Cubic and quadratic scores disagree for {regime.name}, "
                            f"N={n}: {cubic_score} versus {exact_score}."
                        )

            row_similarity = row_rbf(regime.sigma)

            def multiscale_call():
                return etw_align_multiscale(
                    pair.f_values,
                    pair.f_times,
                    pair.g_values,
                    pair.g_times,
                    row_similarity=row_similarity,
                    coarsest_size=config.coarsest_size,
                    initial_radius=config.initial_radius,
                    adaptive=config.adaptive,
                    stability_rtol=config.stability_tolerance,
                    stability_atol=config.stability_tolerance,
                    max_attempts=config.max_attempts,
                    use_numba=config.use_numba,
                )

            multiscale_result, multiscale_times = _run_repeated(
                multiscale_call, config.repeats
            )
            multi_extra = _multiscale_fields(multiscale_result, n)
            if exact_score is None:
                score_gap = math.nan
                relative_gap = math.nan
            else:
                score_gap = float(exact_score - multiscale_result.score)
                if score_gap < -2e-10:
                    raise AssertionError(
                        f"A band-constrained score exceeded the exact score for "
                        f"{regime.name}, N={n}: gap={score_gap}."
                    )
                score_gap = max(0.0, score_gap)
                relative_gap = score_gap / max(abs(exact_score), 1.0e-15)
            multi_extra.update(
                {
                    "score_gap_to_exact": score_gap,
                    "relative_score_gap": relative_gap,
                }
            )
            _append_method_rows(
                rows,
                regime=regime,
                n=n,
                method="multiscale",
                method_label="Adaptive multiscale",
                score=multiscale_result.score,
                solve_times=multiscale_times,
                similarity_seconds=0.0,
                extra=multi_extra,
            )

            if config.fixed_radius_diagnostic:
                fixed, fixed_seconds = _time_once(
                    etw_align_multiscale,
                    pair.f_values,
                    pair.f_times,
                    pair.g_values,
                    pair.g_times,
                    row_similarity=row_similarity,
                    coarsest_size=config.coarsest_size,
                    initial_radius=config.initial_radius,
                    adaptive=False,
                    use_numba=config.use_numba,
                )
                if exact_score is None:
                    fixed_gap = math.nan
                    fixed_relative_gap = math.nan
                else:
                    fixed_gap = max(0.0, float(exact_score - fixed.score))
                    fixed_relative_gap = fixed_gap / max(abs(exact_score), 1.0e-15)
                fixed_extra = _multiscale_fields(fixed, n)
                fixed_extra.update(
                    {
                        "score_gap_to_exact": fixed_gap,
                        "relative_score_gap": fixed_relative_gap,
                    }
                )
                _append_method_rows(
                    rows,
                    regime=regime,
                    n=n,
                    method="fixed_radius_diagnostic",
                    method_label="Fixed-radius diagnostic",
                    score=fixed.score,
                    solve_times=[fixed_seconds],
                    similarity_seconds=0.0,
                    extra=fixed_extra,
                )

            if progress:
                multi_median = float(np.median(multiscale_times))
                radius = multi_extra["accepted_radius"]
                cells = multi_extra["evaluated_cells"]
                exact_text = (
                    ""
                    if exact_score is None
                    else f", gap={float(multi_extra['score_gap_to_exact']):.3g}"
                )
                print(
                    f"N={n:5d}: multiscale {multi_median:8.4f}s, "
                    f"radius={radius:4d}, cells={cells:,}{exact_text}"
                )

            del dense
            gc.collect()

    summary = summarize_rows(rows)
    fits = fit_scaling_exponents(summary)
    return BenchmarkRun(
        config=config,
        regimes=regimes_tuple,
        sizes=sizes,
        raw_rows=tuple(rows),
        summary_rows=summary,
        scaling_fits=fits,
        largest_instances=largest_instances,
    )


def _rows_to_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _summary_lookup(run: BenchmarkRun, regime: str, method: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in run.summary_rows
            if row["regime"] == regime and row["method"] == method
        ],
        key=lambda row: int(row["n"]),
    )


def _fixed_diagnostic_lookup(run: BenchmarkRun, regime: str) -> list[dict[str, Any]]:
    rows = [
        row
        for row in run.raw_rows
        if row["regime"] == regime and row["method"] == "fixed_radius_diagnostic"
    ]
    return sorted(rows, key=lambda row: int(row["n"]))


def _regime_by_name(run: BenchmarkRun, name: str) -> BenchmarkRegime:
    return next(regime for regime in run.regimes if regime.name == name)


def plot_regime_examples(run: BenchmarkRun, output: str | Path):
    """Show the two benchmark regimes at the largest requested size."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(run.regimes), figsize=(12, 7), squeeze=False)
    for column, regime in enumerate(run.regimes):
        pair = run.largest_instances[regime.name]
        n = len(pair.f_values)
        stride = max(1, n // 700)
        f_mid = 0.5 * (pair.f_times[:-1] + pair.f_times[1:])
        g_mid = 0.5 * (pair.g_times[:-1] + pair.g_times[1:])

        ax = axes[0, column]
        ax.plot(pair.f_values[::stride, 0], pair.f_values[::stride, 1], label="f")
        ax.plot(pair.g_values[::stride, 0], pair.g_values[::stride, 1], label="g")
        ax.set_title(regime.label)
        ax.set_xlabel("value coordinate 1")
        ax.set_ylabel("value coordinate 2")
        ax.legend()
        ax.grid(True, alpha=0.25)

        ax = axes[1, column]
        ax.plot(f_mid[::stride], pair.f_values[::stride, 1], label="f")
        ax.plot(g_mid[::stride], pair.g_values[::stride, 1], label="g")
        ax.set_xlabel("observed time")
        ax.set_ylabel("value coordinate 2")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Benchmark regimes at N={run.config.n_max:,}",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    return fig


def plot_wall_clock(run: BenchmarkRun, output: str | Path):
    """Plot end-to-end wall clock against problem size for all three methods."""

    import matplotlib.pyplot as plt

    methods = ("cubic", "quadratic", "multiscale")
    markers = {"cubic": "o", "quadratic": "s", "multiscale": "^"}
    fig, axes = plt.subplots(1, len(run.regimes), figsize=(12.5, 4.8), squeeze=False)

    fit_map = {
        (row["regime"], row["method"]): float(row["slope"])
        for row in run.scaling_fits
    }
    for ax, regime in zip(axes[0], run.regimes):
        for method in methods:
            data = _summary_lookup(run, regime.name, method)
            if not data:
                continue
            n = np.asarray([row["n"] for row in data], dtype=float)
            median = np.asarray([row["median_total_seconds"] for row in data], dtype=float)
            q25 = np.asarray([row["q25_total_seconds"] for row in data], dtype=float)
            q75 = np.asarray([row["q75_total_seconds"] for row in data], dtype=float)
            slope = fit_map.get((regime.name, method))
            label = str(data[0]["method_label"])
            if slope is not None:
                label += f"  (slope {slope:.2f})"
            (line,) = ax.plot(n, median, marker=markers[method], linewidth=2.0, label=label)
            ax.fill_between(n, q25, q75, alpha=0.16, color=line.get_color())

        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("intervals per series, N")
        ax.set_title(regime.label)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("end-to-end wall clock (seconds)")
    fig.suptitle("Elastic time warping: measured scaling", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    return fig


def plot_accuracy_and_work(run: BenchmarkRun, output: str | Path):
    """Plot multiscale score gaps and the amount of sparse work performed."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(run.regimes), figsize=(12.5, 8.0), squeeze=False)
    for column, regime in enumerate(run.regimes):
        adaptive = _summary_lookup(run, regime.name, "multiscale")
        fixed = _fixed_diagnostic_lookup(run, regime.name)

        ax = axes[0, column]
        exact_adaptive = [
            row
            for row in adaptive
            if "relative_score_gap" in row
            and math.isfinite(float(row.get("relative_score_gap", math.nan)))
        ]
        if exact_adaptive:
            n = np.asarray([row["n"] for row in exact_adaptive], dtype=float)
            gap = np.asarray([row["relative_score_gap"] for row in exact_adaptive], dtype=float)
            ax.plot(n, np.maximum(gap, 1.0e-16), marker="^", linewidth=2.0, label="adaptive")
        exact_fixed = [
            row
            for row in fixed
            if row.get("relative_score_gap") is not None
            and math.isfinite(float(row.get("relative_score_gap", math.nan)))
        ]
        if exact_fixed:
            n = np.asarray([row["n"] for row in exact_fixed], dtype=float)
            gap = np.asarray([row["relative_score_gap"] for row in exact_fixed], dtype=float)
            ax.plot(
                n,
                np.maximum(gap, 1.0e-16),
                marker="x",
                linestyle="--",
                linewidth=1.8,
                label=f"fixed radius {run.config.initial_radius}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(regime.label)
        ax.set_ylabel("relative score gap to exact")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)

        ax = axes[1, column]
        n = np.asarray([row["n"] for row in adaptive], dtype=float)
        fraction = np.asarray(
            [row.get("evaluated_cell_fraction", math.nan) for row in adaptive],
            dtype=float,
        )
        radius = np.asarray(
            [row.get("accepted_radius", math.nan) for row in adaptive], dtype=float
        )
        (line,) = ax.plot(
            n,
            100.0 * fraction,
            marker="^",
            linewidth=2.0,
            label="evaluated cells",
        )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("intervals per series, N")
        ax.set_ylabel("cells evaluated (% of full N² grid)")
        ax.grid(True, which="both", alpha=0.25)
        twin = ax.twinx()
        twin.plot(
            n,
            radius,
            marker=".",
            linestyle=":",
            linewidth=1.5,
            label="accepted radius",
        )
        twin.set_ylabel("finest accepted radius")
        # Combine legends from the two y-axes.
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = twin.get_legend_handles_labels()
        ax.legend(handles + handles2, labels + labels2, fontsize=8, loc="best")

    fig.suptitle("Multiscale accuracy and adaptive work", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    return fig


def system_information() -> dict[str, Any]:
    """Collect enough environment metadata to interpret timing results."""

    try:
        import numba

        numba_version = numba.__version__
    except Exception:
        numba_version = None
    try:
        import matplotlib

        matplotlib_version = matplotlib.__version__
    except Exception:
        matplotlib_version = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "numba": numba_version,
        "matplotlib": matplotlib_version,
    }


def write_benchmark_bundle(
    run: BenchmarkRun,
    *,
    output_dir: str | Path,
    prefix: Optional[str] = None,
    notebook_path: Optional[str | Path] = None,
) -> dict[str, Path]:
    """Write raw data, figures, metadata, and one shareable return ZIP."""

    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"etw_three_method_benchmark_N{run.config.n_max}_seed{run.config.seed}"

    raw_csv = directory / f"{prefix}_raw.csv"
    summary_csv = directory / f"{prefix}_summary.csv"
    config_json = directory / f"{prefix}_config.json"
    fits_json = directory / f"{prefix}_scaling_fits.json"
    system_json = directory / f"{prefix}_system.json"
    instances_npz = directory / f"{prefix}_largest_instances.npz"
    regime_png = directory / f"{prefix}_regimes.png"
    timing_png = directory / f"{prefix}_wall_clock.png"
    quality_png = directory / f"{prefix}_quality_work.png"
    readme = directory / f"{prefix}_README.txt"
    return_zip = directory / f"{prefix}_RETURN_THIS.zip"

    _rows_to_csv(run.raw_rows, raw_csv)
    _rows_to_csv(run.summary_rows, summary_csv)
    config_json.write_text(
        json.dumps(
            _json_ready(
                {
                    "config": asdict(run.config),
                    "sizes": run.sizes,
                    "regimes": [asdict(regime) for regime in run.regimes],
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    fits_json.write_text(
        json.dumps(_json_ready(run.scaling_fits), indent=2), encoding="utf-8"
    )
    system_json.write_text(
        json.dumps(_json_ready(system_information()), indent=2), encoding="utf-8"
    )

    payload: dict[str, np.ndarray] = {}
    for regime in run.regimes:
        pair = run.largest_instances[regime.name]
        prefix_key = regime.name
        payload[f"{prefix_key}_f_values"] = np.asarray(pair.f_values)
        payload[f"{prefix_key}_f_times"] = np.asarray(pair.f_times)
        payload[f"{prefix_key}_g_values"] = np.asarray(pair.g_values)
        payload[f"{prefix_key}_g_times"] = np.asarray(pair.g_times)
        payload[f"{prefix_key}_latent_times"] = np.asarray(pair.latent_times)
        payload[f"{prefix_key}_latent_values"] = np.asarray(pair.latent_values)
    np.savez_compressed(instances_npz, **payload)

    fig = plot_regime_examples(run, regime_png)
    plt.close(fig)
    fig = plot_wall_clock(run, timing_png)
    plt.close(fig)
    fig = plot_accuracy_and_work(run, quality_png)
    plt.close(fig)

    notebook_copy: Optional[Path] = None
    if notebook_path is not None:
        source = Path(notebook_path)
        if source.exists():
            notebook_copy = directory / source.name
            shutil.copy2(source, notebook_copy)

    common_sizes = [
        n
        for n in run.sizes
        if n <= run.config.cubic_max_n and n <= run.config.quadratic_max_n
    ]
    readme.write_text(
        "ETW three-method benchmark bundle\n"
        "=================================\n\n"
        f"Maximum requested size: N={run.config.n_max}\n"
        f"All-three-method range: {common_sizes[0] if common_sizes else 'none'}"
        f" through {common_sizes[-1] if common_sizes else 'none'}\n"
        f"Direct cubic cap: N={run.config.cubic_max_n}\n"
        f"Exact quadratic cap: N={run.config.quadratic_max_n}\n"
        f"Timing repeats: {run.config.repeats}\n\n"
        "The wall-clock figure reports end-to-end time.  For the two dense\n"
        "methods this is measured dense-similarity time plus DP time.  The\n"
        "multiscale time is measured externally around the complete hierarchy.\n\n"
        "The fixed-radius curve in the quality figure is a diagnostic only;\n"
        "the headline third method is adaptive multiscale ETW.\n\n"
        "Send back this ZIP.  It contains raw repeat-level timing data, summary\n"
        "tables, figures, configuration, environment metadata, and the largest\n"
        "synthetic instances used in the run.\n",
        encoding="utf-8",
    )

    bundle_items = [
        raw_csv,
        summary_csv,
        config_json,
        fits_json,
        system_json,
        instances_npz,
        regime_png,
        regime_png.with_suffix(".svg"),
        timing_png,
        timing_png.with_suffix(".svg"),
        quality_png,
        quality_png.with_suffix(".svg"),
        readme,
    ]
    if notebook_copy is not None:
        bundle_items.append(notebook_copy)

    with zipfile.ZipFile(return_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_items:
            archive.write(path, arcname=path.name)

    return {
        "raw_csv": raw_csv,
        "summary_csv": summary_csv,
        "config_json": config_json,
        "scaling_fits_json": fits_json,
        "system_json": system_json,
        "largest_instances_npz": instances_npz,
        "regime_figure": regime_png,
        "timing_figure": timing_png,
        "quality_figure": quality_png,
        "readme": readme,
        "return_zip": return_zip,
    }


def summary_html(run: BenchmarkRun) -> str:
    """Compact HTML table for notebook display."""

    headers = (
        "regime",
        "N",
        "method",
        "median total (s)",
        "score gap",
        "cells / N²",
        "radius",
    )
    rows_html: list[str] = []
    for row in run.summary_rows:
        gap = row.get("score_gap_to_exact")
        gap_text = "—" if gap is None or not math.isfinite(float(gap)) else f"{float(gap):.3g}"
        fraction = row.get("evaluated_cell_fraction")
        fraction_text = (
            "—"
            if fraction is None or not math.isfinite(float(fraction))
            else f"{100.0 * float(fraction):.3f}%"
        )
        radius = row.get("accepted_radius", "—")
        rows_html.append(
            "<tr>"
            f"<td>{row['regime_label']}</td>"
            f"<td>{int(row['n']):,}</td>"
            f"<td>{row['method_label']}</td>"
            f"<td>{float(row['median_total_seconds']):.5g}</td>"
            f"<td>{gap_text}</td>"
            f"<td>{fraction_text}</td>"
            f"<td>{radius}</td>"
            "</tr>"
        )
    header = "".join(f"<th>{item}</th>" for item in headers)
    return (
        "<div style='max-height:520px;overflow:auto'>"
        "<table style='border-collapse:collapse'>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody>"
        "</table></div>"
    )


__all__ = [
    "BenchmarkConfig",
    "BenchmarkRegime",
    "BenchmarkRun",
    "DEFAULT_REGIMES",
    "estimate_dense_peak_gb",
    "fit_scaling_exponents",
    "make_instance",
    "plot_accuracy_and_work",
    "plot_regime_examples",
    "plot_wall_clock",
    "power_of_two_sizes",
    "row_rbf",
    "run_benchmark",
    "summary_html",
    "validate_config",
    "warm_numba",
    "write_benchmark_bundle",
]
