"""Illustrate sampling-density sensitivity of DTW and Hellinger ETW.

The example deliberately fixes the continuous curves and true warp, then changes
only the sampling densities.  Hellinger ETW uses the timestamp intervals in its
block score.  Classical DTW is run with the same RBF local geometry, but treats
the sampled points as an unweighted index sequence.

This is an illustrative stress test, not a sampling-consistency theorem for the
restricted block ETW recurrence.  The latter can itself depend on segmentation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import csv
import json
import math
import zipfile

import matplotlib.pyplot as plt
import numpy as np

from hellinger_etw import ETWBlock, etw_align


@dataclass(frozen=True)
class SamplingExperimentResult:
    label: str
    beta_f: float
    beta_g: float
    f_breaks: np.ndarray
    g_breaks: np.ndarray
    f_values: np.ndarray
    g_values: np.ndarray
    similarity_matrix: np.ndarray
    dtw_path: tuple[tuple[int, int], ...]
    hellinger_blocks: tuple[ETWBlock, ...]
    evaluation_grid: np.ndarray
    true_warp: np.ndarray
    hellinger_warp: np.ndarray
    dtw_warp: np.ndarray
    hellinger_rmse: float
    dtw_rmse: float


@dataclass(frozen=True)
class SamplingComparison:
    n: int
    beta: float
    sigma: float
    experiments: tuple[SamplingExperimentResult, SamplingExperimentResult]
    hellinger_between_rmse: float
    dtw_between_rmse: float

    def summary_rows(self) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for result in self.experiments:
            rows.append(
                {
                    "experiment": result.label,
                    "beta_f": result.beta_f,
                    "beta_g": result.beta_g,
                    "hellinger_rmse_to_true": result.hellinger_rmse,
                    "dtw_rmse_to_true": result.dtw_rmse,
                    "hellinger_between_experiments": self.hellinger_between_rmse,
                    "dtw_between_experiments": self.dtw_between_rmse,
                }
            )
        return rows


def landmark_curve(t: Sequence[float] | np.ndarray) -> np.ndarray:
    """A scalar continuous curve with three asymmetric landmarks."""

    x = np.asarray(t, dtype=np.float64)
    y = (
        np.exp(-((x - 0.22) / 0.055) ** 2)
        - 0.80 * np.exp(-((x - 0.52) / 0.090) ** 2)
        + 1.10 * np.exp(-((x - 0.79) / 0.040) ** 2)
        + 0.15 * x
    )
    return y[..., None]


def true_warp(t: Sequence[float] | np.ndarray) -> np.ndarray:
    """Smooth increasing map alpha with alpha(0)=0 and alpha(1)=1."""

    query = np.asarray(t, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, 20_001)
    speed = np.exp(
        0.65 * np.sin(2.0 * np.pi * grid + 0.40)
        + 0.15 * np.cos(4.0 * np.pi * grid - 0.30)
    )
    cumulative = np.empty_like(grid)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(
        0.5 * (speed[1:] + speed[:-1]) * np.diff(grid)
    )
    cumulative /= cumulative[-1]
    return np.interp(np.clip(query, 0.0, 1.0), grid, cumulative)


def exponential_density(t: Sequence[float] | np.ndarray, beta: float) -> np.ndarray:
    """Density proportional to exp(beta * (2 t - 1))."""

    x = np.asarray(t, dtype=np.float64)
    values = np.exp(float(beta) * (2.0 * x - 1.0))
    return values / np.trapezoid(values, x)


def density_quantile_breaks(n: int, beta: float, *, grid_size: int = 50_001) -> np.ndarray:
    """Return n interval breakpoints whose density is ``exponential_density``."""

    if n <= 0:
        raise ValueError("n must be positive.")
    grid = np.linspace(0.0, 1.0, int(grid_size))
    density = exponential_density(grid, beta)
    cdf = np.empty_like(grid)
    cdf[0] = 0.0
    cdf[1:] = np.cumsum(0.5 * (density[1:] + density[:-1]) * np.diff(grid))
    cdf /= cdf[-1]
    breaks = np.interp(np.linspace(0.0, 1.0, n + 1), cdf, grid)
    breaks[0] = 0.0
    breaks[-1] = 1.0
    return breaks


def rbf_similarity_matrix(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    difference = x_arr[:, None, :] - y_arr[None, :, :]
    squared = np.sum(difference * difference, axis=-1)
    return np.exp(-0.5 * squared / (float(sigma) ** 2))


def standard_dtw_path(similarity_matrix: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Classical three-neighbour DTW using local cost 1 - similarity."""

    similarity = np.asarray(similarity_matrix, dtype=np.float64)
    if similarity.ndim != 2 or min(similarity.shape) == 0:
        raise ValueError("similarity_matrix must be a nonempty matrix.")
    cost = 1.0 - similarity
    n, m = cost.shape
    values = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    values[0, 0] = 0.0
    predecessor = np.zeros((n, m), dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            candidates = (
                values[i - 1, j - 1],
                values[i - 1, j],
                values[i, j - 1],
            )
            move = int(np.argmin(candidates))
            values[i, j] = cost[i - 1, j - 1] + candidates[move]
            predecessor[i - 1, j - 1] = move

    i = n - 1
    j = m - 1
    reversed_path: list[tuple[int, int]] = []
    while True:
        reversed_path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(predecessor[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return tuple(reversed(reversed_path))


def hellinger_warp_from_blocks(
    blocks: Sequence[ETWBlock],
    f_breaks: np.ndarray,
    g_breaks: np.ndarray,
    similarity_matrix: np.ndarray,
    evaluation_grid: np.ndarray,
) -> np.ndarray:
    """Reconstruct the optimized map alpha: g-time -> f-time from ETW blocks."""

    ds = np.diff(f_breaks)
    dt = np.diff(g_breaks)
    knot_t = [0.0]
    knot_s = [0.0]

    for block in blocks:
        if block.kind == "one_f_to_many_g":
            i = block.f_start
            q = block.g_start
            j = block.g_stop
            weights = dt[q:j] * similarity_matrix[i, q:j] ** 2
            total = float(np.sum(weights))
            if total <= 0.0:
                weights = dt[q:j]
                total = float(np.sum(weights))
            cumulative = 0.0
            for ell, weight in zip(range(q, j), weights):
                cumulative += float(weight)
                knot_t.append(float(g_breaks[ell + 1]))
                knot_s.append(
                    float(
                        f_breaks[i]
                        + ds[i] * cumulative / total
                    )
                )
        elif block.kind == "many_f_to_one_g":
            h = block.f_start
            i = block.f_stop
            j = block.g_start
            weights = ds[h:i] * similarity_matrix[h:i, j] ** 2
            total = float(np.sum(weights))
            if total <= 0.0:
                weights = ds[h:i]
                total = float(np.sum(weights))
            cumulative = 0.0
            for r, weight in zip(range(h, i), weights):
                cumulative += float(weight)
                knot_t.append(
                    float(
                        g_breaks[j]
                        + dt[j] * cumulative / total
                    )
                )
                knot_s.append(float(f_breaks[r + 1]))
        else:
            raise ValueError(
                "This visualization assumes the no-skip Hellinger recurrence."
            )

    knot_t_array = np.asarray(knot_t, dtype=np.float64)
    knot_s_array = np.asarray(knot_s, dtype=np.float64)
    knot_t_array[-1] = 1.0
    knot_s_array[-1] = 1.0
    knot_s_array = np.maximum.accumulate(knot_s_array)
    return np.interp(evaluation_grid, knot_t_array, knot_s_array)


def dtw_warp_from_path(
    path: Sequence[tuple[int, int]],
    f_breaks: np.ndarray,
    g_breaks: np.ndarray,
    evaluation_grid: np.ndarray,
) -> np.ndarray:
    """Visualize a DTW path as a map from physical g-time to physical f-time.

    For each g sample, the displayed f-time is the mean midpoint time of all f
    samples matched to it.  This is only a visualization convention; the DTW
    optimization itself is the standard index-based recurrence above.
    """

    f_mid = 0.5 * (f_breaks[:-1] + f_breaks[1:])
    g_mid = 0.5 * (g_breaks[:-1] + g_breaks[1:])
    matched: list[list[float]] = [[] for _ in range(len(g_mid))]
    for i, j in path:
        matched[j].append(float(f_mid[i]))

    x = [0.0]
    y = [0.0]
    for j, values in enumerate(matched):
        if not values:
            raise RuntimeError("A standard DTW path should visit every g index.")
        x.append(float(g_mid[j]))
        y.append(float(np.mean(values)))
    x.append(1.0)
    y.append(1.0)
    y_array = np.maximum.accumulate(np.asarray(y, dtype=np.float64))
    return np.interp(evaluation_grid, np.asarray(x), y_array)


def run_experiment(
    *,
    n: int,
    beta_f: float,
    beta_g: float,
    sigma: float,
    label: str,
    evaluation_points: int = 2001,
    use_numba: bool = True,
) -> SamplingExperimentResult:
    f_breaks = density_quantile_breaks(n, beta_f)
    g_breaks = density_quantile_breaks(n, beta_g)
    f_mid = 0.5 * (f_breaks[:-1] + f_breaks[1:])
    g_mid = 0.5 * (g_breaks[:-1] + g_breaks[1:])
    f_values = landmark_curve(f_mid)
    g_values = landmark_curve(true_warp(g_mid))
    similarity = rbf_similarity_matrix(f_values, g_values, sigma)

    hellinger = etw_align(
        f_values,
        f_breaks,
        g_values,
        g_breaks,
        similarity_matrix=similarity,
        use_numba=use_numba,
    )
    dtw_path = standard_dtw_path(similarity)
    grid = np.linspace(0.0, 1.0, int(evaluation_points))
    truth = true_warp(grid)
    hellinger_warp = hellinger_warp_from_blocks(
        hellinger.blocks, f_breaks, g_breaks, similarity, grid
    )
    dtw_warp = dtw_warp_from_path(dtw_path, f_breaks, g_breaks, grid)

    return SamplingExperimentResult(
        label=label,
        beta_f=float(beta_f),
        beta_g=float(beta_g),
        f_breaks=f_breaks,
        g_breaks=g_breaks,
        f_values=f_values,
        g_values=g_values,
        similarity_matrix=similarity,
        dtw_path=dtw_path,
        hellinger_blocks=tuple(hellinger.blocks),
        evaluation_grid=grid,
        true_warp=truth,
        hellinger_warp=hellinger_warp,
        dtw_warp=dtw_warp,
        hellinger_rmse=float(np.sqrt(np.mean((hellinger_warp - truth) ** 2))),
        dtw_rmse=float(np.sqrt(np.mean((dtw_warp - truth) ** 2))),
    )


def run_swapped_density_comparison(
    *,
    n: int = 80,
    beta: float = 2.0,
    sigma: float = 0.25,
    use_numba: bool = True,
) -> SamplingComparison:
    first = run_experiment(
        n=n,
        beta_f=beta,
        beta_g=-beta,
        sigma=sigma,
        label="A: f dense late; g dense early",
        use_numba=use_numba,
    )
    second = run_experiment(
        n=n,
        beta_f=-beta,
        beta_g=beta,
        sigma=sigma,
        label="B: f dense early; g dense late",
        use_numba=use_numba,
    )
    h_between = float(
        np.sqrt(np.mean((first.hellinger_warp - second.hellinger_warp) ** 2))
    )
    d_between = float(np.sqrt(np.mean((first.dtw_warp - second.dtw_warp) ** 2)))
    return SamplingComparison(
        n=int(n),
        beta=float(beta),
        sigma=float(sigma),
        experiments=(first, second),
        hellinger_between_rmse=h_between,
        dtw_between_rmse=d_between,
    )


def density_contrast_sweep(
    *,
    n: int = 80,
    beta_values: Iterable[float] = tuple(np.linspace(0.0, 2.4, 13)),
    sigma: float = 0.25,
    use_numba: bool = True,
) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for beta in beta_values:
        comparison = run_swapped_density_comparison(
            n=n, beta=float(beta), sigma=sigma, use_numba=use_numba
        )
        rows.append(
            {
                "beta": float(beta),
                "hellinger_worst_rmse": max(
                    result.hellinger_rmse for result in comparison.experiments
                ),
                "dtw_worst_rmse": max(
                    result.dtw_rmse for result in comparison.experiments
                ),
                "hellinger_between_rmse": comparison.hellinger_between_rmse,
                "dtw_between_rmse": comparison.dtw_between_rmse,
            }
        )
    return rows


def plot_main_comparison(
    comparison: SamplingComparison,
    output: str | Path | None = None,
):
    dense_grid = np.linspace(0.0, 1.0, 2001)
    continuous_f = landmark_curve(dense_grid)[:, 0]
    continuous_g = landmark_curve(true_warp(dense_grid))[:, 0]
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.3), sharex=False)

    for row, result in enumerate(comparison.experiments):
        f_mid = 0.5 * (result.f_breaks[:-1] + result.f_breaks[1:])
        g_mid = 0.5 * (result.g_breaks[:-1] + result.g_breaks[1:])
        axis = axes[row, 0]
        axis.plot(dense_grid, continuous_f, linewidth=2.0, label="f(s)")
        axis.scatter(f_mid, result.f_values[:, 0], s=13, alpha=0.75, label="f samples")
        axis.plot(dense_grid, continuous_g, linewidth=2.0, label="g(t)")
        axis.scatter(g_mid, result.g_values[:, 0], s=13, alpha=0.75, label="g samples")
        axis.set_title(result.label)
        axis.set_xlabel("physical time")
        axis.set_ylabel("signal value")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=9)

        axis = axes[row, 1]
        axis.plot(
            result.evaluation_grid,
            result.true_warp,
            color="black",
            linewidth=2.4,
            label="true α",
        )
        axis.plot(
            result.evaluation_grid,
            result.hellinger_warp,
            linewidth=2.2,
            label=f"Hellinger ETW  RMSE={result.hellinger_rmse:.3f}",
        )
        axis.plot(
            result.evaluation_grid,
            result.dtw_warp,
            linewidth=2.2,
            label=f"standard DTW  RMSE={result.dtw_rmse:.3f}",
        )
        axis.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, color="0.70")
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("g-time t")
        axis.set_ylabel("estimated f-time α(t)")
        axis.set_title("Recovered physical-time warp")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best", fontsize=9)

    fig.suptitle(
        "Same continuous curves and warp; only the sampling densities are swapped",
        fontsize=16,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    if output is not None:
        path = Path(output)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    return fig


def plot_stability_summary(
    comparison: SamplingComparison,
    sweep_rows: Sequence[dict[str, float]],
    output: str | Path | None = None,
):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6))
    grid = comparison.experiments[0].evaluation_grid
    truth = comparison.experiments[0].true_warp

    axes[0].plot(grid, truth, color="black", linewidth=2.4, label="true α")
    for result in comparison.experiments:
        axes[0].plot(grid, result.hellinger_warp, linewidth=2.0, label=result.label[0])
    axes[0].set_title(
        f"Hellinger: between-run RMSE {comparison.hellinger_between_rmse:.3f}"
    )
    axes[0].set_xlabel("g-time t")
    axes[0].set_ylabel("f-time")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(title="experiment", fontsize=9)

    axes[1].plot(grid, truth, color="black", linewidth=2.4, label="true α")
    for result in comparison.experiments:
        axes[1].plot(grid, result.dtw_warp, linewidth=2.0, label=result.label[0])
    axes[1].set_title(f"DTW: between-run RMSE {comparison.dtw_between_rmse:.3f}")
    axes[1].set_xlabel("g-time t")
    axes[1].set_ylabel("f-time")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(title="experiment", fontsize=9)

    beta = [row["beta"] for row in sweep_rows]
    axes[2].plot(
        beta,
        [row["hellinger_worst_rmse"] for row in sweep_rows],
        marker="o",
        linewidth=2.0,
        label="Hellinger ETW",
    )
    axes[2].plot(
        beta,
        [row["dtw_worst_rmse"] for row in sweep_rows],
        marker="o",
        linewidth=2.0,
        label="standard DTW",
    )
    axes[2].axvline(comparison.beta, linestyle="--", linewidth=1.0, color="0.5")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("sampling-density contrast β")
    axes[2].set_ylabel("worst warp RMSE across A/B")
    axes[2].set_title("Density-contrast stress sweep")
    axes[2].grid(True, which="both", alpha=0.25)
    axes[2].legend(fontsize=9)

    fig.tight_layout()
    if output is not None:
        path = Path(output)
        fig.savefig(path, dpi=220, bbox_inches="tight")
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    return fig


def write_result_bundle(
    comparison: SamplingComparison,
    sweep_rows: Sequence[dict[str, float]],
    *,
    output_dir: str | Path,
    prefix: str = "sampling_rate_comparison",
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    main_png = output / f"{prefix}_main.png"
    summary_png = output / f"{prefix}_stability.png"
    fig_main = plot_main_comparison(comparison, main_png)
    fig_summary = plot_stability_summary(comparison, sweep_rows, summary_png)
    plt.close(fig_main)
    plt.close(fig_summary)

    summary = {
        "n": comparison.n,
        "beta": comparison.beta,
        "sigma": comparison.sigma,
        "hellinger_between_rmse": comparison.hellinger_between_rmse,
        "dtw_between_rmse": comparison.dtw_between_rmse,
        "experiments": comparison.summary_rows(),
        "sweep": list(sweep_rows),
        "caveat": (
            "This is an illustrative stress test. The restricted block ETW "
            "recurrence is not universally invariant to segmentation."
        ),
    }
    json_path = output / f"{prefix}.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = output / f"{prefix}_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sweep_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sweep_rows)

    arrays: dict[str, np.ndarray] = {}
    for index, result in enumerate(comparison.experiments, start=1):
        prefix_i = f"experiment_{index}"
        arrays[f"{prefix_i}_f_breaks"] = result.f_breaks
        arrays[f"{prefix_i}_g_breaks"] = result.g_breaks
        arrays[f"{prefix_i}_grid"] = result.evaluation_grid
        arrays[f"{prefix_i}_true_warp"] = result.true_warp
        arrays[f"{prefix_i}_hellinger_warp"] = result.hellinger_warp
        arrays[f"{prefix_i}_dtw_warp"] = result.dtw_warp
    npz_path = output / f"{prefix}.npz"
    np.savez_compressed(npz_path, **arrays)

    zip_path = output / f"{prefix}_RETURN_THIS.zip"
    files = [
        main_png,
        main_png.with_suffix(".svg"),
        summary_png,
        summary_png.with_suffix(".svg"),
        json_path,
        csv_path,
        npz_path,
    ]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, arcname=path.name)
    return {
        "main_figure": main_png,
        "stability_figure": summary_png,
        "summary_json": json_path,
        "sweep_csv": csv_path,
        "arrays_npz": npz_path,
        "return_zip": zip_path,
    }


__all__ = [
    "SamplingComparison",
    "SamplingExperimentResult",
    "density_contrast_sweep",
    "density_quantile_breaks",
    "landmark_curve",
    "plot_main_comparison",
    "plot_stability_summary",
    "run_experiment",
    "run_swapped_density_comparison",
    "standard_dtw_path",
    "true_warp",
    "write_result_bundle",
]
