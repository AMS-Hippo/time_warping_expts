"""Coarse-to-fine Hellinger Elastic Time Warping.

This module builds a dyadic hierarchy, solves the coarsest problem exactly, and
then refines the matching inside sparse row-interval corridors.  The restricted
solve at every level is exact for its band; the overall multiscale method is a
heuristic unless the finest band expands to the full grid.

The implementation is designed for the usual infill regime.  If a corridor of
half-width ``r`` contains ``K = O(r(n+m))`` pair cells, both similarity work and
the sparse dynamic program are proportional to ``K`` rather than ``n*m``.

For numerical values, adjacent intervals are coarsened by their
duration-weighted mean.  For values in an arbitrary metric space, callers must
provide ``coarsen_values`` because there is no canonical mean in a general
metric space.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, List, Optional, Sequence, Tuple

import math
import warnings

import numpy as np

from hellinger_etw import ETWBlock, ETWResult, etw_align as etw_align_dense
from hellinger_etw_sparse import (
    InfeasibleBandError,
    RowBand,
    SparseETWResult,
    etw_align_sparse_banded,
)


ValueReducer = Callable[[Sequence[Any], np.ndarray], Any]
ScalarSimilarity = Callable[[Any, Any], float]
RowSimilarity = Callable[[Any, Sequence[Any]], Sequence[float]]


@dataclass(frozen=True)
class SeriesScale:
    """One series at one hierarchy level.

    ``child_start[k]:child_stop[k]`` gives the intervals at the next finer
    level represented by coarse interval ``k``.  These arrays are ``None`` at
    the original finest level.
    """

    values: Sequence[Any]
    breaks: np.ndarray
    child_start: Optional[np.ndarray] = None
    child_stop: Optional[np.ndarray] = None

    @property
    def n(self) -> int:
        return len(self.values)


@dataclass(frozen=True)
class PairScale:
    """The two time series at one hierarchy level."""

    f: SeriesScale
    g: SeriesScale

    @property
    def shape(self) -> Tuple[int, int]:
        return self.f.n, self.g.n


@dataclass(frozen=True)
class BandAttemptDiagnostics:
    """Diagnostics for one radius attempted at one refinement level."""

    radius: int
    allowed_cells: int
    density: float
    score: Optional[float]
    boundary_touches: int
    full_grid: bool
    similarity_evaluations: int
    row_similarity_calls: int
    elapsed_seconds: float
    status: str
    score_change: Optional[float] = None


@dataclass(frozen=True)
class MultiscaleLevelDiagnostics:
    """Diagnostics for one hierarchy level, indexed with 0 = finest."""

    level: int
    n: int
    m: int
    coarsest: bool
    attempts: Tuple[BandAttemptDiagnostics, ...]
    accepted_attempt: int
    stopping_reason: str
    score_stable: bool
    boundary_free: bool

    @property
    def accepted(self) -> BandAttemptDiagnostics:
        return self.attempts[self.accepted_attempt]


@dataclass(frozen=True)
class MultiscaleDiagnostics:
    """Diagnostics for an entire multiscale solve."""

    levels: Tuple[MultiscaleLevelDiagnostics, ...]
    total_similarity_evaluations: int
    total_allowed_cells_evaluated: int
    total_elapsed_seconds: float
    final_full_grid: bool
    heuristic_converged: bool


@dataclass(frozen=True)
class MultiscaleETWResult:
    """Result returned by :func:`etw_align_multiscale`."""

    score: float
    blocks: List[ETWBlock]
    pairs: List[Tuple[int, int]]
    unmatched_f: List[int]
    unmatched_g: List[int]
    final_band: RowBand
    diagnostics: MultiscaleDiagnostics

    @property
    def exact(self) -> bool:
        """Whether the finest solve covered the full grid."""

        return self.diagnostics.final_full_grid


def etw_align_multiscale(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    *,
    similarity: Optional[ScalarSimilarity] = None,
    row_similarity: Optional[RowSimilarity] = None,
    coarsen_values: Optional[ValueReducer] = None,
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
    coarsest_size: int = 64,
    initial_radius: int = 4,
    adaptive: bool = True,
    propagate_radius: bool = True,
    stability_rtol: float = 1.0e-8,
    stability_atol: float = 1.0e-10,
    require_boundary_free: bool = True,
    min_successful_attempts: int = 2,
    max_radius: Optional[int] = None,
    max_attempts: Optional[int] = None,
    use_numba: bool = True,
    check_nonnegative: bool = True,
) -> MultiscaleETWResult:
    """Align two series by coarse-to-fine sparse refinement.

    Exactly one of ``similarity`` and ``row_similarity`` must be supplied.

    The coarsest level is solved on its full grid with the exact dense
    quadratic algorithm.  Each finer level lifts the accepted matching from
    the preceding coarse level, dilates it by an index radius, and solves the
    original no-skip recurrence exactly inside that band.

    When ``adaptive`` is true, the radius is doubled until two successful
    radii have stable scores and the accepted path does not touch an artificial
    band boundary, or until a full-grid solve / user stopping limit is reached.
    By default, a radius required at one refinement level is propagated to the
    next finer level: the next doubling schedule starts at half that radius, so
    it is guaranteed to test at least the previously accepted width without
    forcing the radius to double at every level.

    Score stability and boundary freedom are diagnostics, not a proof of global
    optimality.  The returned result is certified exact only when the finest
    band is the full grid.
    """

    if (similarity is None) == (row_similarity is None):
        raise ValueError("Provide exactly one of similarity or row_similarity.")
    if coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
    if initial_radius < 0:
        raise ValueError("initial_radius must be nonnegative.")
    if stability_rtol < 0.0 or stability_atol < 0.0:
        raise ValueError("stability tolerances must be nonnegative.")
    if min_successful_attempts < 1:
        raise ValueError("min_successful_attempts must be at least one.")
    if max_radius is not None and max_radius < initial_radius:
        raise ValueError("max_radius cannot be smaller than initial_radius.")
    if max_attempts is not None and max_attempts < 1:
        raise ValueError("max_attempts must be positive when supplied.")

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both input series must contain at least one value.")

    f_breaks = _normalize_breaks(f_times, n, end_f, "f_times")
    g_breaks = _normalize_breaks(g_times, m, end_g, "g_times")
    hierarchy = build_dyadic_hierarchy(
        f_values,
        f_breaks,
        g_values,
        g_breaks,
        coarsest_size=coarsest_size,
        coarsen_values=coarsen_values,
    )

    t_total = perf_counter()
    level_diags: List[MultiscaleLevelDiagnostics] = []

    coarsest_index = len(hierarchy) - 1
    coarsest = hierarchy[coarsest_index]
    t0 = perf_counter()
    C_coarse, coarse_evals, coarse_row_calls = _dense_similarity_matrix(
        coarsest.f.values,
        coarsest.g.values,
        similarity=similarity,
        row_similarity=row_similarity,
    )
    coarse_result = etw_align_dense(
        coarsest.f.values,
        coarsest.f.breaks,
        coarsest.g.values,
        coarsest.g.breaks,
        similarity_matrix=C_coarse,
        use_numba=use_numba,
        check_nonnegative=check_nonnegative,
    )
    coarse_elapsed = perf_counter() - t0
    cn, cm = coarsest.shape
    coarse_attempt = BandAttemptDiagnostics(
        radius=max(cn, cm),
        allowed_cells=cn * cm,
        density=1.0,
        score=float(coarse_result.score),
        boundary_touches=0,
        full_grid=True,
        similarity_evaluations=coarse_evals,
        row_similarity_calls=coarse_row_calls,
        elapsed_seconds=coarse_elapsed,
        status="ok",
        score_change=None,
    )
    level_diags.append(
        MultiscaleLevelDiagnostics(
            level=coarsest_index,
            n=cn,
            m=cm,
            coarsest=True,
            attempts=(coarse_attempt,),
            accepted_attempt=0,
            stopping_reason="coarsest_exact",
            score_stable=True,
            boundary_free=True,
        )
    )

    current_result: ETWResult | SparseETWResult = coarse_result
    current_band = RowBand.full(cn, cm)
    refinement_radius = int(initial_radius)

    for level_index in range(coarsest_index - 1, -1, -1):
        fine = hierarchy[level_index]
        parent = hierarchy[level_index + 1]
        if parent.f.child_start is None or parent.g.child_start is None:
            raise RuntimeError("Hierarchy mapping is missing at a refinement step.")

        result, band, diag = _solve_refinement_level(
            level=level_index,
            fine=fine,
            parent=parent,
            coarse_pairs=current_result.pairs,
            similarity=similarity,
            row_similarity=row_similarity,
            initial_radius=refinement_radius,
            adaptive=adaptive,
            stability_rtol=stability_rtol,
            stability_atol=stability_atol,
            require_boundary_free=require_boundary_free,
            min_successful_attempts=min_successful_attempts,
            max_radius=max_radius,
            max_attempts=max_attempts,
            use_numba=use_numba,
            check_nonnegative=check_nonnegative,
        )
        current_result = result
        current_band = band
        level_diags.append(diag)
        if propagate_radius:
            accepted_radius = diag.accepted.radius
            carried_start = 0 if accepted_radius == 0 else (accepted_radius + 1) // 2
            refinement_radius = max(int(initial_radius), carried_start)

    # Present diagnostics in natural coarse-to-fine execution order.  The level
    # field itself still uses 0 = finest, which is convenient for analysis.
    levels_tuple = tuple(level_diags)
    total_evals = sum(
        attempt.similarity_evaluations
        for level in levels_tuple
        for attempt in level.attempts
    )
    total_cells = sum(
        attempt.allowed_cells for level in levels_tuple for attempt in level.attempts
    )
    final_diag = next(level for level in levels_tuple if level.level == 0)
    final_full = final_diag.accepted.full_grid
    converged = all(
        level.stopping_reason in {"coarsest_exact", "full_grid", "score_stable"}
        for level in levels_tuple
    )
    diagnostics = MultiscaleDiagnostics(
        levels=levels_tuple,
        total_similarity_evaluations=total_evals,
        total_allowed_cells_evaluated=total_cells,
        total_elapsed_seconds=perf_counter() - t_total,
        final_full_grid=final_full,
        heuristic_converged=converged,
    )

    return MultiscaleETWResult(
        score=float(current_result.score),
        blocks=current_result.blocks,
        pairs=current_result.pairs,
        unmatched_f=current_result.unmatched_f,
        unmatched_g=current_result.unmatched_g,
        final_band=current_band,
        diagnostics=diagnostics,
    )


def build_dyadic_hierarchy(
    f_values: Sequence[Any],
    f_breaks: Sequence[float],
    g_values: Sequence[Any],
    g_breaks: Sequence[float],
    *,
    coarsest_size: int = 64,
    coarsen_values: Optional[ValueReducer] = None,
) -> Tuple[PairScale, ...]:
    """Build fine-to-coarse dyadic levels and child maps."""

    if coarsest_size < 1:
        raise ValueError("coarsest_size must be positive.")
    f0 = SeriesScale(f_values, _validated_breaks(f_breaks, len(f_values), "f_breaks"))
    g0 = SeriesScale(g_values, _validated_breaks(g_breaks, len(g_values), "g_breaks"))
    levels: List[PairScale] = [PairScale(f0, g0)]

    while levels[-1].f.n > coarsest_size or levels[-1].g.n > coarsest_size:
        current = levels[-1]
        next_f = _coarsen_series_once(
            current.f,
            reduce=current.f.n > coarsest_size,
            reducer=coarsen_values,
            name="f_values",
        )
        next_g = _coarsen_series_once(
            current.g,
            reduce=current.g.n > coarsest_size,
            reducer=coarsen_values,
            name="g_values",
        )
        levels.append(PairScale(next_f, next_g))

    return tuple(levels)


def lift_pairs_to_row_band(
    fine_n: int,
    fine_m: int,
    coarse_pairs: Sequence[Tuple[int, int]],
    f_child_start: np.ndarray,
    f_child_stop: np.ndarray,
    g_child_start: np.ndarray,
    g_child_stop: np.ndarray,
    *,
    radius: int,
) -> RowBand:
    """Lift coarse matched cells to a dilated row-interval corridor."""

    if fine_n <= 0 or fine_m <= 0:
        raise ValueError("fine_n and fine_m must be positive.")
    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    if len(coarse_pairs) == 0:
        raise ValueError("coarse_pairs must not be empty.")

    lo = np.full(fine_n, fine_m, dtype=np.int64)
    hi = np.zeros(fine_n, dtype=np.int64)
    r = int(radius)

    for coarse_i_raw, coarse_j_raw in coarse_pairs:
        coarse_i = int(coarse_i_raw)
        coarse_j = int(coarse_j_raw)
        if not (0 <= coarse_i < f_child_start.size):
            raise ValueError(f"Invalid coarse f index {coarse_i}.")
        if not (0 <= coarse_j < g_child_start.size):
            raise ValueError(f"Invalid coarse g index {coarse_j}.")
        row_a = max(0, int(f_child_start[coarse_i]) - r)
        row_b = min(fine_n, int(f_child_stop[coarse_i]) + r)
        col_a = max(0, int(g_child_start[coarse_j]) - r)
        col_b = min(fine_m, int(g_child_stop[coarse_j]) + r)
        for row in range(row_a, row_b):
            if col_a < lo[row]:
                lo[row] = col_a
            if col_b > hi[row]:
                hi[row] = col_b

    empty = hi == 0
    if np.any(empty):
        first = int(np.flatnonzero(empty)[0])
        raise ValueError(
            f"Lifted coarse matching did not cover fine row {first}; hierarchy is inconsistent."
        )
    band = RowBand(lo, hi, fine_m)
    if not band.contains(0, 0) or not band.contains(fine_n - 1, fine_m - 1):
        raise ValueError("Lifted band does not contain both endpoint pair cells.")
    return band


def _solve_refinement_level(
    *,
    level: int,
    fine: PairScale,
    parent: PairScale,
    coarse_pairs: Sequence[Tuple[int, int]],
    similarity: Optional[ScalarSimilarity],
    row_similarity: Optional[RowSimilarity],
    initial_radius: int,
    adaptive: bool,
    stability_rtol: float,
    stability_atol: float,
    require_boundary_free: bool,
    min_successful_attempts: int,
    max_radius: Optional[int],
    max_attempts: Optional[int],
    use_numba: bool,
    check_nonnegative: bool,
) -> Tuple[SparseETWResult, RowBand, MultiscaleLevelDiagnostics]:
    n, m = fine.shape
    effective_max_radius = max(n, m) if max_radius is None else int(max_radius)
    radius = min(int(initial_radius), effective_max_radius)
    attempts: List[BandAttemptDiagnostics] = []
    successful_scores: List[float] = []
    last_result: Optional[SparseETWResult] = None
    last_band: Optional[RowBand] = None
    stopping_reason = "fixed_radius"
    stable = False
    boundary_free = False

    attempt_no = 0
    while True:
        attempt_no += 1
        band = lift_pairs_to_row_band(
            n,
            m,
            coarse_pairs,
            parent.f.child_start,  # type: ignore[arg-type]
            parent.f.child_stop,  # type: ignore[arg-type]
            parent.g.child_start,  # type: ignore[arg-type]
            parent.g.child_stop,  # type: ignore[arg-type]
            radius=radius,
        )
        full = band.allowed_cells == n * m
        t0 = perf_counter()
        try:
            result = etw_align_sparse_banded(
                fine.f.values,
                fine.f.breaks,
                fine.g.values,
                fine.g.breaks,
                band=band,
                similarity=similarity,
                row_similarity=row_similarity,
                use_numba=use_numba,
                check_nonnegative=check_nonnegative,
            )
        except InfeasibleBandError as exc:
            elapsed = perf_counter() - t0
            attempts.append(
                BandAttemptDiagnostics(
                    radius=radius,
                    allowed_cells=band.allowed_cells,
                    density=band.density,
                    score=None,
                    boundary_touches=0,
                    full_grid=full,
                    similarity_evaluations=band.allowed_cells,
                    row_similarity_calls=n if row_similarity is not None else 0,
                    elapsed_seconds=elapsed,
                    status=f"infeasible: {exc}",
                    score_change=None,
                )
            )
            if full or radius >= effective_max_radius:
                raise
            if max_attempts is not None and attempt_no >= max_attempts:
                raise RuntimeError(
                    "Multiscale refinement exhausted max_attempts without a finite alignment."
                ) from exc
            radius = _next_radius(radius, effective_max_radius)
            continue

        elapsed = perf_counter() - t0
        touches = _count_artificial_boundary_touches(result.pairs, band)
        boundary_free = touches == 0
        change = None
        if successful_scores:
            previous = successful_scores[-1]
            if result.score + _score_tolerance(
                result.score, previous, stability_rtol, stability_atol
            ) < previous:
                raise RuntimeError(
                    "A wider nested band produced a smaller score; this indicates "
                    "an implementation or numerical error."
                )
            change = float(result.score - previous)
            stable = abs(change) <= _score_tolerance(
                result.score, previous, stability_rtol, stability_atol
            )
        successful_scores.append(float(result.score))
        attempts.append(
            BandAttemptDiagnostics(
                radius=radius,
                allowed_cells=band.allowed_cells,
                density=band.density,
                score=float(result.score),
                boundary_touches=touches,
                full_grid=full,
                similarity_evaluations=result.diagnostics.similarity_evaluations,
                row_similarity_calls=result.diagnostics.row_similarity_calls,
                elapsed_seconds=elapsed,
                status="ok",
                score_change=change,
            )
        )
        last_result = result
        last_band = band

        if not adaptive:
            stopping_reason = "fixed_radius"
            break
        if full:
            stable = True
            boundary_free = True
            stopping_reason = "full_grid"
            break
        stability_ready = stable or (
            min_successful_attempts == 1 and len(successful_scores) == 1
        )
        if (
            len(successful_scores) >= min_successful_attempts
            and stability_ready
            and (boundary_free or not require_boundary_free)
        ):
            stable = stability_ready
            stopping_reason = "score_stable"
            break
        if radius >= effective_max_radius:
            stopping_reason = "max_radius"
            break
        if max_attempts is not None and attempt_no >= max_attempts:
            stopping_reason = "max_attempts"
            break
        radius = _next_radius(radius, effective_max_radius)

    assert last_result is not None and last_band is not None
    accepted_attempt = len(attempts) - 1
    if adaptive and stopping_reason in {"max_radius", "max_attempts"}:
        warnings.warn(
            f"Level {level} stopped at {stopping_reason} before the adaptive "
            "stability criterion was met; the returned banded result is heuristic.",
            RuntimeWarning,
            stacklevel=2,
        )
    diag = MultiscaleLevelDiagnostics(
        level=level,
        n=n,
        m=m,
        coarsest=False,
        attempts=tuple(attempts),
        accepted_attempt=accepted_attempt,
        stopping_reason=stopping_reason,
        score_stable=stable,
        boundary_free=boundary_free,
    )
    return last_result, last_band, diag


def _normalize_breaks(
    times: Sequence[float], n: int, end: Optional[float], name: str
) -> np.ndarray:
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if arr.size == n + 1:
        out = arr.copy()
    elif arr.size == n:
        endpoint = 1.0 if end is None else float(end)
        out = np.empty(n + 1, dtype=np.float64)
        out[:-1] = arr
        out[-1] = endpoint
    else:
        raise ValueError(
            f"{name} must contain {n} starts or {n + 1} breakpoints; got {arr.size}."
        )
    return _validated_breaks(out, n, name)


def _validated_breaks(breaks: Sequence[float], n: int, name: str) -> np.ndarray:
    arr = np.asarray(breaks, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n + 1:
        raise ValueError(f"{name} must be a length-{n + 1} one-dimensional array.")
    if not np.all(np.isfinite(arr)) or np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return arr.copy()


def _coarsen_series_once(
    series: SeriesScale,
    *,
    reduce: bool,
    reducer: Optional[ValueReducer],
    name: str,
) -> SeriesScale:
    n = series.n
    if not reduce:
        starts = np.arange(n, dtype=np.int64)
        stops = starts + 1
        return SeriesScale(series.values, series.breaks.copy(), starts, stops)

    starts = np.arange(0, n, 2, dtype=np.int64)
    stops = np.minimum(starts + 2, n)
    durations = np.diff(series.breaks)
    coarse_values: List[Any] = []
    for a, b in zip(starts, stops):
        vals = [series.values[k] for k in range(int(a), int(b))]
        weights = durations[int(a) : int(b)]
        if reducer is None:
            coarse_values.append(_duration_weighted_numeric_mean(vals, weights, name))
        else:
            coarse_values.append(reducer(vals, weights.copy()))

    coarse_breaks = np.empty(starts.size + 1, dtype=np.float64)
    coarse_breaks[:-1] = series.breaks[starts]
    coarse_breaks[-1] = series.breaks[-1]
    packed_values = _pack_values(coarse_values)
    return SeriesScale(packed_values, coarse_breaks, starts, stops)


def _duration_weighted_numeric_mean(
    values: Sequence[Any], weights: np.ndarray, name: str
) -> Any:
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} cannot be coarsened automatically.  Supply coarsen_values "
            "for non-numerical or metric-space observations."
        ) from exc
    if arr.ndim < 1 or arr.shape[0] != len(values):
        raise ValueError(
            f"{name} cannot be coarsened automatically; supply coarsen_values."
        )
    w = np.asarray(weights, dtype=np.float64)
    normalized = w / float(np.sum(w))
    return np.tensordot(normalized, arr, axes=(0, 0))


def _pack_values(values: List[Any]) -> Sequence[Any]:
    try:
        arr = np.asarray(values)
        if arr.dtype != object and arr.shape[0] == len(values):
            return arr
    except (TypeError, ValueError):
        pass
    return values


def _dense_similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    *,
    similarity: Optional[ScalarSimilarity],
    row_similarity: Optional[RowSimilarity],
) -> Tuple[np.ndarray, int, int]:
    n = len(f_values)
    m = len(g_values)
    out = np.empty((n, m), dtype=np.float64)
    if row_similarity is not None:
        for i in range(n):
            try:
                targets = g_values[:m]  # type: ignore[index]
            except (TypeError, AttributeError):
                targets = [g_values[j] for j in range(m)]
            values = np.asarray(row_similarity(f_values[i], targets), dtype=np.float64)
            if values.ndim != 1 or values.size != m:
                raise ValueError(
                    f"row_similarity returned shape {values.shape} at the coarsest "
                    f"level; expected length {m}."
                )
            out[i] = values
        return out, n * m, n

    assert similarity is not None
    for i in range(n):
        for j in range(m):
            out[i, j] = float(similarity(f_values[i], g_values[j]))
    return out, n * m, 0


def _count_artificial_boundary_touches(
    pairs: Sequence[Tuple[int, int]], band: RowBand
) -> int:
    touches = 0
    for i_raw, j_raw in pairs:
        i = int(i_raw)
        j = int(j_raw)
        at_left = j == int(band.lo[i]) and int(band.lo[i]) > 0
        at_right = j == int(band.hi[i]) - 1 and int(band.hi[i]) < band.m
        if at_left or at_right:
            touches += 1
    return touches


def _score_tolerance(a: float, b: float, rtol: float, atol: float) -> float:
    return float(atol + rtol * max(1.0, abs(a), abs(b)))


def _next_radius(radius: int, maximum: int) -> int:
    if radius >= maximum:
        return maximum
    return min(maximum, 1 if radius == 0 else 2 * radius)


__all__ = [
    "BandAttemptDiagnostics",
    "MultiscaleDiagnostics",
    "MultiscaleETWResult",
    "MultiscaleLevelDiagnostics",
    "PairScale",
    "SeriesScale",
    "build_dyadic_hierarchy",
    "etw_align_multiscale",
    "lift_pairs_to_row_band",
]
