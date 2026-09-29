"""Record and visualize one coarse-to-fine ETW run.

The example is intentionally deterministic.  It is used by the accompanying
notebook and by the presentation-slide builder so that the multiscale picture
reflects the actual implementation rather than a hand-drawn schematic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hellinger_etw import etw_align as etw_align_dense
from hellinger_etw_multiscale import (
    PairScale,
    build_dyadic_hierarchy,
    lift_pairs_to_row_band,
)
from hellinger_etw_sparse import RowBand, etw_align_sparse_banded
from simulate_etw_data import SimulatedPair, rbf_similarity_matrix, simulate_pair


@dataclass(frozen=True)
class TraceLevel:
    """One accepted level of a fixed-radius coarse-to-fine run."""

    hierarchy_index: int
    n: int
    m: int
    radius: Optional[int]
    band: RowBand
    score: float
    pairs: tuple[tuple[int, int], ...]

    @property
    def full_cells(self) -> int:
        return self.n * self.m

    @property
    def allowed_cells(self) -> int:
        return self.band.allowed_cells

    @property
    def density(self) -> float:
        return self.allowed_cells / self.full_cells


@dataclass(frozen=True)
class MultiscaleTrace:
    """Recorded hierarchy and accepted paths in coarse-to-fine order."""

    pair: SimulatedPair
    hierarchy: tuple[PairScale, ...]
    levels: tuple[TraceLevel, ...]
    exact_score: Optional[float]
    sigma: float
    radius: int
    seed: int
    noise: float
    warp_strength: float
    coarsest_size: int

    @property
    def final_score(self) -> float:
        return self.levels[-1].score

    @property
    def total_allowed_cells(self) -> int:
        return sum(level.allowed_cells for level in self.levels)

    @property
    def finest_full_cells(self) -> int:
        return self.levels[-1].full_cells

    def summary_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for order, level in enumerate(self.levels):
            rows.append(
                {
                    "stage": order,
                    "hierarchy_index": level.hierarchy_index,
                    "shape": f"{level.n} x {level.m}",
                    "radius": "full" if level.radius is None else level.radius,
                    "allowed_cells": level.allowed_cells,
                    "full_cells": level.full_cells,
                    "density": level.density,
                    "score": level.score,
                    "matched_pairs": len(level.pairs),
                }
            )
        return rows

    def summary_dict(self) -> dict[str, object]:
        score_gap = None if self.exact_score is None else self.exact_score - self.final_score
        return {
            "seed": self.seed,
            "noise": self.noise,
            "warp_strength": self.warp_strength,
            "sigma": self.sigma,
            "radius": self.radius,
            "coarsest_size": self.coarsest_size,
            "n": self.levels[-1].n,
            "m": self.levels[-1].m,
            "exact_computed": self.exact_score is not None,
            "exact_score": self.exact_score,
            "final_score": self.final_score,
            "score_gap": score_gap,
            "total_allowed_cells": self.total_allowed_cells,
            "finest_full_cells": self.finest_full_cells,
            "reduction_factor": self.finest_full_cells / self.total_allowed_cells,
            "levels": self.summary_rows(),
        }

    def write_summary_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.summary_dict(), indent=2), encoding="utf-8")

    def write_trace_npz(self, path: str | Path) -> None:
        """Write bands and selected paths in a compact, non-pickle archive.

        This contains everything needed to redraw the multiscale corridor/path
        panels without rerunning the solver.
        """

        payload: dict[str, np.ndarray] = {
            "level_count": np.asarray([len(self.levels)], dtype=np.int64),
            "seed": np.asarray([self.seed], dtype=np.int64),
            "sigma": np.asarray([self.sigma], dtype=np.float64),
            "radius": np.asarray([self.radius], dtype=np.int64),
            "noise": np.asarray([self.noise], dtype=np.float64),
            "warp_strength": np.asarray([self.warp_strength], dtype=np.float64),
            "coarsest_size": np.asarray([self.coarsest_size], dtype=np.int64),
            "final_score": np.asarray([self.final_score], dtype=np.float64),
            "exact_score": np.asarray(
                [np.nan if self.exact_score is None else self.exact_score],
                dtype=np.float64,
            ),
        }
        for index, level in enumerate(self.levels):
            prefix = f"level_{index}"
            payload[f"{prefix}_hierarchy_index"] = np.asarray(
                [level.hierarchy_index], dtype=np.int64
            )
            payload[f"{prefix}_shape"] = np.asarray([level.n, level.m], dtype=np.int64)
            payload[f"{prefix}_radius"] = np.asarray(
                [-1 if level.radius is None else level.radius], dtype=np.int64
            )
            payload[f"{prefix}_score"] = np.asarray([level.score], dtype=np.float64)
            payload[f"{prefix}_lo"] = np.asarray(level.band.lo, dtype=np.int64)
            payload[f"{prefix}_hi"] = np.asarray(level.band.hi, dtype=np.int64)
            payload[f"{prefix}_pairs"] = np.asarray(level.pairs, dtype=np.int64).reshape(-1, 2)
        np.savez_compressed(path, **payload)


def row_rbf(sigma: float):
    """Return a vectorized row-wise RBF similarity callback."""

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def similarity(x, ys) -> np.ndarray:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(ys, dtype=np.float64)
        delta = y_arr - x_arr
        return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / (sigma * sigma))

    return similarity


def record_fixed_radius_run(
    *,
    n: int = 256,
    m: Optional[int] = None,
    seed: int = 356,
    noise: float = 0.02,
    warp_strength: float = 0.5,
    sigma: float = 0.25,
    coarsest_size: int = 32,
    radius: int = 4,
    use_numba: bool = True,
    compute_exact: bool = True,
) -> MultiscaleTrace:
    """Record a deterministic fixed-radius run, including every accepted band.

    This function intentionally spells out the public coarse-to-fine operations
    rather than reaching into private solver internals.  It is therefore also a
    compact executable description of the algorithm.
    """

    if m is None:
        m = n
    pair = simulate_pair(
        n,
        m,
        seed=seed,
        noise=noise,
        warp_strength=warp_strength,
        grid_size=max(2048, 2 * max(n, m)),
    )
    hierarchy = build_dyadic_hierarchy(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        coarsest_size=coarsest_size,
    )

    coarse_index = len(hierarchy) - 1
    coarse = hierarchy[coarse_index]
    coarse_similarity = rbf_similarity_matrix(
        np.asarray(coarse.f.values),
        np.asarray(coarse.g.values),
        sigma=sigma,
    )
    current = etw_align_dense(
        coarse.f.values,
        coarse.f.breaks,
        coarse.g.values,
        coarse.g.breaks,
        similarity_matrix=coarse_similarity,
        use_numba=use_numba,
    )

    levels: list[TraceLevel] = [
        TraceLevel(
            hierarchy_index=coarse_index,
            n=coarse.f.n,
            m=coarse.g.n,
            radius=None,
            band=RowBand.full(coarse.f.n, coarse.g.n),
            score=float(current.score),
            pairs=tuple((int(i), int(j)) for i, j in current.pairs),
        )
    ]

    row_similarity = row_rbf(sigma)
    for hierarchy_index in range(coarse_index - 1, -1, -1):
        fine = hierarchy[hierarchy_index]
        parent = hierarchy[hierarchy_index + 1]
        assert parent.f.child_start is not None
        assert parent.f.child_stop is not None
        assert parent.g.child_start is not None
        assert parent.g.child_stop is not None
        band = lift_pairs_to_row_band(
            fine.f.n,
            fine.g.n,
            current.pairs,
            parent.f.child_start,
            parent.f.child_stop,
            parent.g.child_start,
            parent.g.child_stop,
            radius=radius,
        )
        current = etw_align_sparse_banded(
            fine.f.values,
            fine.f.breaks,
            fine.g.values,
            fine.g.breaks,
            band=band,
            row_similarity=row_similarity,
            use_numba=use_numba,
        )
        levels.append(
            TraceLevel(
                hierarchy_index=hierarchy_index,
                n=fine.f.n,
                m=fine.g.n,
                radius=radius,
                band=band,
                score=float(current.score),
                pairs=tuple((int(i), int(j)) for i, j in current.pairs),
            )
        )

    exact_score: Optional[float] = None
    if compute_exact:
        finest_similarity = rbf_similarity_matrix(
            pair.f_values,
            pair.g_values,
            sigma=sigma,
        )
        exact = etw_align_dense(
            pair.f_values,
            pair.f_times,
            pair.g_values,
            pair.g_times,
            similarity_matrix=finest_similarity,
            use_numba=use_numba,
        )
        exact_score = float(exact.score)

    return MultiscaleTrace(
        pair=pair,
        hierarchy=hierarchy,
        levels=tuple(levels),
        exact_score=exact_score,
        sigma=sigma,
        radius=radius,
        seed=seed,
        noise=noise,
        warp_strength=warp_strength,
        coarsest_size=coarsest_size,
    )


def _compressed_path(pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    """Remove duplicate and collinear points from a monotone cell path."""

    if not pairs:
        return np.empty((0, 2), dtype=np.float64)
    points: list[tuple[int, int]] = []
    for point in pairs:
        if not points or point != points[-1]:
            points.append(point)
    if len(points) <= 2:
        return np.asarray(points, dtype=np.float64)

    keep: list[tuple[int, int]] = [points[0]]
    previous_direction: Optional[tuple[int, int]] = None
    for index in range(1, len(points)):
        di = points[index][0] - points[index - 1][0]
        dj = points[index][1] - points[index - 1][1]
        direction = (int(np.sign(di)), int(np.sign(dj)))
        if previous_direction is None:
            previous_direction = direction
            continue
        if direction != previous_direction:
            keep.append(points[index - 1])
            previous_direction = direction
    keep.append(points[-1])
    return np.asarray(keep, dtype=np.float64)


def plot_trace(
    trace: MultiscaleTrace,
    *,
    output: Optional[str | Path] = None,
    levels: Optional[Iterable[int]] = None,
):
    """Plot the recorded hierarchy with evaluated corridors and refined paths."""

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    chosen = list(trace.levels if levels is None else [trace.levels[i] for i in levels])
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.3 * len(chosen), 3.6), squeeze=False)
    axes_row = axes[0]

    for stage, (ax, level) in enumerate(zip(axes_row, chosen)):
        for row in range(level.n):
            lo = int(level.band.lo[row])
            hi = int(level.band.hi[row])
            ax.add_patch(
                Rectangle(
                    (lo, row),
                    hi - lo,
                    1.0,
                    facecolor="#DDEEFF",
                    edgecolor="none",
                )
            )
        path = _compressed_path(level.pairs)
        if path.size:
            ax.plot(path[:, 1] + 0.5, path[:, 0] + 0.5, color="#F06400", linewidth=2.2)
        ax.set_xlim(0, level.m)
        ax.set_ylim(0, level.n)
        ax.set_aspect("equal")
        ax.set_xlabel("g interval")
        if stage == 0:
            ax.set_ylabel("f interval")
        else:
            ax.set_yticklabels([])
        label = "full grid" if level.radius is None else f"radius {level.radius}"
        ax.set_title(
            f"{level.n} x {level.m}: {label}\n"
            f"{level.allowed_cells:,} / {level.full_cells:,} cells",
            fontsize=10,
        )
        ax.grid(True, linewidth=0.35, alpha=0.25)

    fig.suptitle(
        "Coarse-to-fine ETW: shaded cells are evaluated; orange is the selected path",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    if output is not None:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, bbox_inches="tight")
    return fig


def choose_display_levels(level_count: int, max_levels: int = 4) -> list[int]:
    """Choose hierarchy levels that make a four-panel explanation legible.

    For a long hierarchy, evenly spaced levels make the first displayed corridor
    unnecessarily thin.  The presentation-friendly choice is instead:

    1. the complete coarsest grid;
    2. the first refined grid, where the corridor is still visibly wide;
    3. a late refinement, but not so late that it is indistinguishable from the
       final panel;
    4. the finest grid.

    With ten levels this gives indices ``[0, 1, 6, 9]``.
    """

    if level_count <= 0:
        raise ValueError("level_count must be positive.")
    if max_levels <= 0:
        raise ValueError("max_levels must be positive.")
    if level_count <= max_levels:
        return list(range(level_count))
    if max_levels != 4:
        # Retain a predictable generic fallback for non-presentation uses.
        raw = np.linspace(0, level_count - 1, max_levels)
        selected = sorted({int(round(value)) for value in raw})
        selected[0] = 0
        selected[-1] = level_count - 1
        candidate = 0
        while len(selected) < max_levels:
            if candidate not in selected:
                selected.append(candidate)
                selected.sort()
            candidate += 1
        return selected[:max_levels]

    late = max(2, level_count - 4)
    selected = [0, 1, late, level_count - 1]
    # Very short hierarchies can make ``late`` collide with an endpoint.
    if len(set(selected)) < 4:
        selected = [0, 1, level_count - 2, level_count - 1]
    return selected


def write_shareable_run_bundle(
    trace: MultiscaleTrace,
    *,
    output_dir: str | Path,
    prefix: Optional[str] = None,
    max_slide_levels: int = 4,
) -> dict[str, Path]:
    """Write figures, summaries, raw corridor/path arrays, and a return ZIP.

    The ZIP is intended to be sent back for deterministic incorporation into a
    presentation slide.
    """

    from zipfile import ZIP_DEFLATED, ZipFile
    import matplotlib.pyplot as plt

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    if prefix is None:
        prefix = f"multiscale_run_N{trace.levels[-1].n}_seed{trace.seed}"

    selected = choose_display_levels(len(trace.levels), max_levels=max_slide_levels)
    summary_path = directory / f"{prefix}.json"
    archive_path = directory / f"{prefix}.npz"
    all_figure_path = directory / f"{prefix}_all_levels.png"
    slide_figure_path = directory / f"{prefix}_slide_levels.png"
    readme_path = directory / f"{prefix}_README.txt"
    zip_path = directory / f"{prefix}_RETURN_THIS.zip"

    trace.write_summary_json(summary_path)
    trace.write_trace_npz(archive_path)
    fig = plot_trace(trace, output=all_figure_path)
    plt.close(fig)
    fig = plot_trace(trace, output=slide_figure_path, levels=selected)
    plt.close(fig)

    selected_shapes = [f"{trace.levels[index].n} x {trace.levels[index].m}" for index in selected]
    exact_line = (
        "An unrestricted dense comparison was computed."
        if trace.exact_score is not None
        else "No unrestricted dense comparison was requested."
    )
    readme_path.write_text(
        "ETW multiscale slide-run bundle\n"
        "================================\n\n"
        f"Finest grid: {trace.levels[-1].n} x {trace.levels[-1].m}\n"
        f"Levels selected for the slide preview: {', '.join(selected_shapes)}\n"
        f"Total evaluated cells: {trace.total_allowed_cells:,}\n"
        f"Finest full grid: {trace.finest_full_cells:,} cells\n"
        f"{exact_line}\n\n"
        "Send this ZIP file back.  The NPZ contains every row-band interval and\n"
        "selected path, so the slide can be rebuilt without rerunning the model.\n",
        encoding="utf-8",
    )

    with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as bundle:
        for item in [summary_path, archive_path, all_figure_path, slide_figure_path, readme_path]:
            bundle.write(item, arcname=item.name)

    return {
        "summary_json": summary_path,
        "trace_npz": archive_path,
        "all_levels_png": all_figure_path,
        "slide_levels_png": slide_figure_path,
        "readme": readme_path,
        "return_zip": zip_path,
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    trace = record_fixed_radius_run()
    plot_trace(trace, output=root / "MULTISCALE_ILLUSTRATION_RUN.png")
    trace.write_summary_json(root / "MULTISCALE_ILLUSTRATION_RUN.json")
    for row in trace.summary_rows():
        print(row)
    message = (
        f"total cells={trace.total_allowed_cells:,}; "
        f"finest full grid={trace.finest_full_cells:,}"
    )
    if trace.exact_score is not None:
        message += f"; score gap={trace.exact_score - trace.final_score:.3g}"
    print(message)
