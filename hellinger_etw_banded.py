"""
Banded Elastic Time Warping with Hellinger block scores.

This module implements an exact solver for the band-constrained version of the
Hellinger Elastic Time Warping recurrence.  A band is a boolean mask
``band_mask[i, j]`` indicating whether interval ``f[i]`` is allowed to be
matched to interval ``g[j]``.  Every many-to-one or one-to-many block is required
to use only allowed interval-pair cells.

The public API mirrors ``hellinger_etw.py`` as closely as possible.  The main
entry points are

    etw_align_banded(..., band_mask=...)
    etw_align(..., band_mask=...)       # alias

When no band is supplied, the full grid is allowed.

Notes
-----
* With infinite skip penalties, the Numba path uses a sparse run-length scan of
  the allowed cells.  This is the intended fast banded mode.
* With finite skip penalties, the solver falls back to a dense masked DP, since
  skip transitions can move through states that are not adjacent to allowed
  match cells.  This is still exact, but no longer subquadratic in the band size.
* The score is exact for the band-constrained problem.  It is an exact global
  score only if the supplied band contains at least one globally optimal
  matching.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple

import math
import warnings

import numpy as np

try:  # Optional acceleration.  The public API works without numba.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - depends on optional dependency
    njit = None  # type: ignore[assignment]
    _HAVE_NUMBA = False


NEG_INF = -np.inf
_EPS = 1.0e-12

# Traceback op codes.  These match hellinger_etw.py.
_OP_NONE = np.int8(0)
_OP_MANY_F_TO_ONE_G = np.int8(1)
_OP_ONE_F_TO_MANY_G = np.int8(2)
_OP_SKIP_F = np.int8(3)
_OP_SKIP_G = np.int8(4)


@dataclass(frozen=True)
class ETWBlock:
    """One block in the optimal traceback.

    Indices use Python half-open intervals.  For example,
    ``f_start=2, f_stop=5, g_start=7, g_stop=8`` means
    ``f[2], f[3], f[4]`` are matched to ``g[7]``.

    ``kind`` is one of
        * ``"many_f_to_one_g"``
        * ``"one_f_to_many_g"``
        * ``"skip_f"``
        * ``"skip_g"``

    ``contribution`` is the score added by this traceback step.  For a skip it
    is negative and equal to minus the corresponding skip penalty.
    """

    kind: str
    f_start: int
    f_stop: int
    g_start: int
    g_stop: int
    contribution: float


@dataclass(frozen=True)
class ETWResult:
    """Result returned by :func:`etw_align_banded`."""

    score: float
    blocks: List[ETWBlock]
    pairs: List[Tuple[int, int]]
    unmatched_f: List[int]
    unmatched_g: List[int]
    score_table: Optional[np.ndarray] = None
    band_mask: Optional[np.ndarray] = None


def etw_align(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    **kwargs: Any,
) -> ETWResult:
    """Alias for :func:`etw_align_banded`.

    This makes the module usable in the same style as ``hellinger_etw.py``:

        from hellinger_etw_banded import etw_align
    """

    return etw_align_banded(f_values, f_times, g_values, g_times, **kwargs)


def etw_align_banded(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    *,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    band_mask: Optional[np.ndarray] = None,
    time_radius: Optional[float] = None,
    path: Optional[Sequence[Tuple[int, int]]] = None,
    path_radius: Optional[int] = None,
    distance_radius: Optional[float] = None,
    band_combine: str = "and",
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
    skip_f_penalty: Optional[Sequence[float] | float] = None,
    skip_g_penalty: Optional[Sequence[float] | float] = None,
    use_numba: bool = True,
    check_nonnegative: bool = True,
    return_score_table: bool = False,
    return_band_mask: bool = False,
) -> ETWResult:
    """Align two timestamped series under a band-constrained ETW objective.

    Parameters are the same as ``hellinger_etw.etw_align`` with these additions:

    band_mask:
        Boolean array of shape ``(n, m)``.  ``band_mask[i, j]`` means interval
        ``f[i]`` may be matched to interval ``g[j]``.  Every block in the
        traceback must consist only of allowed interval-pair cells.

    time_radius:
        Convenience band constructor.  Allows interval pairs whose midpoint
        timestamps differ by at most ``time_radius``.  This is usually the
        simplest Sakoe-Chiba-style band for normalized times.

    path, path_radius:
        Convenience band constructor.  ``path`` is a list of allowed center
        pairs, for example ``result.pairs`` from a coarse solve.  Cells with
        Chebyshev index distance at most ``path_radius`` from the path are
        allowed.

    distance_radius:
        Convenience band constructor for numeric Euclidean values.  Allows
        interval pairs with ``||f[i] - g[j]|| <= distance_radius``.  This is
        useful when the similarity is monotone decreasing in Euclidean distance.

    band_combine:
        How to combine multiple supplied band constructors: ``"and"`` or
        ``"or"``.  The default ``"and"`` is conservative when a time band and a
        distance band are both supplied.

    skip_f_penalty, skip_g_penalty:
        Same meaning as in ``hellinger_etw.py``.  ``None`` means infinite
        penalty, i.e. no skips.  Finite skips are supported exactly, but they
        use a dense masked DP rather than the sparse band scan.

    Returns
    -------
    ETWResult
        Contains the optimal band-constrained score, traceback blocks, expanded
        matched pairs, and unmatched indices.  If ``return_band_mask=True``, the
        effective band is attached as ``result.band_mask``.
    """

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both input series must contain at least one value.")

    ds, f_breaks = _interval_lengths(f_times, n, end_f, "f_times")
    dt, g_breaks = _interval_lengths(g_times, m, end_g, "g_times")

    effective_band = build_band_mask(
        n,
        m,
        f_breaks=f_breaks,
        g_breaks=g_breaks,
        f_values=f_values,
        g_values=g_values,
        band_mask=band_mask,
        time_radius=time_radius,
        path=path,
        path_radius=path_radius,
        distance_radius=distance_radius,
        combine=band_combine,
    )

    if not np.any(effective_band):
        raise ValueError("The effective band contains no allowed match cells.")

    C = _similarity_matrix(f_values, g_values, similarity, similarity_matrix)
    if C.shape != (n, m):
        raise ValueError(
            f"similarity_matrix must have shape {(n, m)}, got {C.shape}."
        )
    if not np.all(np.isfinite(C)):
        raise ValueError("All similarities must be finite real numbers.")
    if check_nonnegative:
        min_c = float(np.min(C))
        if min_c < -1.0e-12:
            raise ValueError(
                "Similarities must be nonnegative for this ETW recurrence. "
                f"Minimum observed value was {min_c}."
            )
        C = np.maximum(C, 0.0)

    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")
    has_finite_skips = bool(np.any(np.isfinite(skip_f)) or np.any(np.isfinite(skip_g)))

    pref_f, pref_g = _prefix_tables(C, ds, dt)
    sqrt_ds = np.sqrt(ds)
    sqrt_dt = np.sqrt(dt)

    if use_numba and _HAVE_NUMBA:
        if has_finite_skips:
            V, ptr_op, ptr_i, ptr_j = _etw_banded_dense_numba(
                pref_f, pref_g, sqrt_dt, sqrt_ds, skip_f, skip_g, effective_band
            )
        else:
            row_ptr, row_cols, col_ptr = _band_to_row_csr_and_col_ptr(effective_band)
            V, ptr_op, ptr_i, ptr_j = _etw_banded_sparse_noskip_numba(
                pref_f, pref_g, sqrt_dt, sqrt_ds, row_ptr, row_cols, col_ptr
            )
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python dense "
                "banded implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        V, ptr_op, ptr_i, ptr_j = _etw_banded_dense_python(
            pref_f, pref_g, sqrt_dt, sqrt_ds, skip_f, skip_g, effective_band
        )

    score = float(V[n, m])
    if not np.isfinite(score):
        raise ValueError(
            "No finite band-constrained alignment was found.  The band may be "
            "too narrow, or skipping may be disallowed while the band blocks all "
            "paths from the start to the end."
        )

    blocks, pairs, unmatched_f, unmatched_g = _traceback(V, ptr_op, ptr_i, ptr_j)
    return ETWResult(
        score=score,
        blocks=blocks,
        pairs=pairs,
        unmatched_f=unmatched_f,
        unmatched_g=unmatched_g,
        score_table=V if return_score_table else None,
        band_mask=effective_band if return_band_mask else None,
    )


def compute_similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    similarity: Callable[[Any, Any], float],
) -> np.ndarray:
    """Compute ``C[i, j] = similarity(f_values[i], g_values[j])``.

    This intentionally uses a simple Python double loop so that ``similarity``
    can be any Python callable.  For numeric data, prefer a vectorized matrix
    builder and pass the result through ``similarity_matrix``.
    """

    n = len(f_values)
    m = len(g_values)
    C = np.empty((n, m), dtype=np.float64)
    for i, x in enumerate(f_values):
        for j, y in enumerate(g_values):
            C[i, j] = float(similarity(x, y))
    return C


def euclidean_similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    *,
    sigma: float = 1.0,
    kind: str = "gaussian",
    block_size: int = 1024,
) -> np.ndarray:
    """Build a dense Euclidean similarity matrix in blocks.

    Parameters
    ----------
    sigma:
        Positive length scale.

    kind:
        ``"gaussian"`` gives ``exp(-||x-y||^2 / (2*sigma^2))``.
        ``"laplacian"`` gives ``exp(-||x-y|| / sigma)``.

    This is a convenience helper for high-dimensional numeric data.  It is still
    O(n*m*d), but avoids an O(n*m) Python callback loop.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    X = _as_2d_float_array(f_values, "f_values")
    Y = _as_2d_float_array(g_values, "g_values")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("f_values and g_values must have the same feature dimension.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    n = X.shape[0]
    m = Y.shape[0]
    Y_norm = np.sum(Y * Y, axis=1)
    C = np.empty((n, m), dtype=np.float64)
    for a in range(0, n, block_size):
        b = min(n, a + block_size)
        Xb = X[a:b]
        d2 = np.sum(Xb * Xb, axis=1)[:, None] + Y_norm[None, :] - 2.0 * (Xb @ Y.T)
        d2 = np.maximum(d2, 0.0)
        if kind == "gaussian":
            C[a:b] = np.exp(-d2 / (2.0 * sigma * sigma))
        elif kind == "laplacian":
            C[a:b] = np.exp(-np.sqrt(d2) / sigma)
        else:
            raise ValueError("kind must be 'gaussian' or 'laplacian'.")
    return C


def make_time_band(
    f_times: Sequence[float],
    n: int,
    g_times: Sequence[float],
    m: int,
    *,
    radius: float,
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
) -> np.ndarray:
    """Return a midpoint timestamp band ``abs(mid_f - mid_g) <= radius``."""

    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")
    _, f_breaks = _interval_lengths(f_times, n, end_f, "f_times")
    _, g_breaks = _interval_lengths(g_times, m, end_g, "g_times")
    f_mid = 0.5 * (f_breaks[:-1] + f_breaks[1:])
    g_mid = 0.5 * (g_breaks[:-1] + g_breaks[1:])
    return np.abs(f_mid[:, None] - g_mid[None, :]) <= float(radius)


def make_path_band(
    n: int,
    m: int,
    path: Sequence[Tuple[int, int]],
    *,
    radius: int,
) -> np.ndarray:
    """Return a Chebyshev-radius band around a list of matched index pairs."""

    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    mask = np.zeros((n, m), dtype=np.bool_)
    r = int(radius)
    for i, j in path:
        ii = int(i)
        jj = int(j)
        if ii < 0 or ii >= n or jj < 0 or jj >= m:
            continue
        a = max(0, ii - r)
        b = min(n, ii + r + 1)
        c = max(0, jj - r)
        d = min(m, jj + r + 1)
        mask[a:b, c:d] = True
    return mask


def make_euclidean_distance_band(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    *,
    radius: float,
    block_size: int = 1024,
) -> np.ndarray:
    """Return ``||f[i]-g[j]|| <= radius`` for numeric Euclidean values.

    This is useful when the similarity is monotone decreasing in Euclidean
    distance.  It can be combined with a time band or path band.
    """

    if radius < 0.0:
        raise ValueError("radius must be nonnegative.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    X = _as_2d_float_array(f_values, "f_values")
    Y = _as_2d_float_array(g_values, "g_values")
    if X.shape[1] != Y.shape[1]:
        raise ValueError("f_values and g_values must have the same feature dimension.")

    n = X.shape[0]
    m = Y.shape[0]
    r2 = float(radius) * float(radius)
    Y_norm = np.sum(Y * Y, axis=1)
    mask = np.empty((n, m), dtype=np.bool_)
    for a in range(0, n, block_size):
        b = min(n, a + block_size)
        Xb = X[a:b]
        d2 = np.sum(Xb * Xb, axis=1)[:, None] + Y_norm[None, :] - 2.0 * (Xb @ Y.T)
        mask[a:b] = d2 <= r2 + 1.0e-12
    return mask


def build_band_mask(
    n: int,
    m: int,
    *,
    f_breaks: Optional[np.ndarray] = None,
    g_breaks: Optional[np.ndarray] = None,
    f_values: Optional[Sequence[Any]] = None,
    g_values: Optional[Sequence[Any]] = None,
    band_mask: Optional[np.ndarray] = None,
    time_radius: Optional[float] = None,
    path: Optional[Sequence[Tuple[int, int]]] = None,
    path_radius: Optional[int] = None,
    distance_radius: Optional[float] = None,
    combine: str = "and",
) -> np.ndarray:
    """Build and combine band masks.

    ``combine='and'`` intersects supplied masks.  ``combine='or'`` unions them.
    If no masks are supplied, the full grid is returned.
    """

    masks: List[np.ndarray] = []

    if band_mask is not None:
        arr = np.asarray(band_mask, dtype=np.bool_)
        if arr.shape != (n, m):
            raise ValueError(f"band_mask must have shape {(n, m)}, got {arr.shape}.")
        masks.append(arr.copy())

    if time_radius is not None:
        if f_breaks is None or g_breaks is None:
            raise ValueError("f_breaks and g_breaks are required for time_radius.")
        if time_radius < 0.0:
            raise ValueError("time_radius must be nonnegative.")
        f_mid = 0.5 * (f_breaks[:-1] + f_breaks[1:])
        g_mid = 0.5 * (g_breaks[:-1] + g_breaks[1:])
        masks.append(np.abs(f_mid[:, None] - g_mid[None, :]) <= float(time_radius))

    if path is not None or path_radius is not None:
        if path is None or path_radius is None:
            raise ValueError("Provide both path and path_radius, or neither.")
        masks.append(make_path_band(n, m, path, radius=int(path_radius)))

    if distance_radius is not None:
        if f_values is None or g_values is None:
            raise ValueError("f_values and g_values are required for distance_radius.")
        masks.append(
            make_euclidean_distance_band(f_values, g_values, radius=float(distance_radius))
        )

    if not masks:
        return np.ones((n, m), dtype=np.bool_)

    mode = combine.lower()
    if mode == "and":
        out = masks[0].copy()
        for mask in masks[1:]:
            out &= mask
        return out
    if mode == "or":
        out = masks[0].copy()
        for mask in masks[1:]:
            out |= mask
        return out
    raise ValueError("combine must be 'and' or 'or'.")


def _similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    similarity: Optional[Callable[[Any, Any], float]],
    similarity_matrix: Optional[np.ndarray],
) -> np.ndarray:
    if similarity_matrix is not None:
        return np.asarray(similarity_matrix, dtype=np.float64)
    if similarity is None:
        raise ValueError("Provide either similarity or similarity_matrix.")
    return compute_similarity_matrix(f_values, g_values, similarity)


def _as_2d_float_array(values: Sequence[Any], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 1D or 2D numeric array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def _interval_lengths(
    times: Sequence[float], n: int, end: Optional[float], name: str
) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")

    if arr.size == n + 1:
        breaks = arr.copy()
    elif arr.size == n:
        final = 1.0 if end is None else float(end)
        breaks = np.empty(n + 1, dtype=np.float64)
        breaks[:n] = arr
        breaks[n] = final
    else:
        raise ValueError(
            f"{name} must contain either {n} start times or {n + 1} "
            f"breakpoints; got {arr.size}."
        )

    if not np.all(np.isfinite(breaks)):
        raise ValueError(f"{name} must contain only finite times.")
    diffs = np.diff(breaks)
    if np.any(diffs <= 0.0):
        raise ValueError(
            f"{name} must be strictly increasing after appending the endpoint."
        )
    return diffs.astype(np.float64), breaks


def _penalty_array(
    penalty: Optional[Sequence[float] | float], n: int, name: str
) -> np.ndarray:
    if penalty is None:
        return np.full(n, np.inf, dtype=np.float64)

    arr = np.asarray(penalty, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(n, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.size == n:
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar or a length-{n} array.")

    if np.any(np.isnan(out)) or np.any(out < 0.0):
        raise ValueError(f"{name} must contain nonnegative penalties or np.inf.")
    return out


def _prefix_tables(
    C: np.ndarray, ds: np.ndarray, dt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    C2 = C * C

    # pref_f[j, i] = sum_{r < i} ds[r] * C[r, j]^2.
    pref_f = np.empty((C.shape[1], C.shape[0] + 1), dtype=np.float64)
    pref_f[:, 0] = 0.0
    pref_f[:, 1:] = np.cumsum((ds[:, None] * C2).T, axis=1)

    # pref_g[i, j] = sum_{q < j} dt[q] * C[i, q]^2.
    pref_g = np.empty((C.shape[0], C.shape[1] + 1), dtype=np.float64)
    pref_g[:, 0] = 0.0
    pref_g[:, 1:] = np.cumsum(dt[None, :] * C2, axis=1)
    return pref_f, pref_g


def _band_to_row_csr_and_col_ptr(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, m = mask.shape
    row_counts = np.sum(mask, axis=1, dtype=np.int64)
    row_ptr = np.empty(n + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(row_counts, out=row_ptr[1:])
    row_cols = np.empty(int(row_ptr[-1]), dtype=np.int64)
    pos = 0
    for i in range(n):
        cols = np.flatnonzero(mask[i])
        row_cols[pos : pos + cols.size] = cols
        pos += cols.size

    col_counts = np.sum(mask, axis=0, dtype=np.int64)
    col_ptr = np.empty(m + 1, dtype=np.int64)
    col_ptr[0] = 0
    np.cumsum(col_counts, out=col_ptr[1:])
    return row_ptr, row_cols, col_ptr


class _Envelope:
    """Pure-Python monotone upper envelope for v + a*sqrt(x - p)."""

    __slots__ = ("a", "idx", "p", "v", "start", "head")

    def __init__(self, a: float) -> None:
        self.a = float(a)
        self.idx: List[int] = []
        self.p: List[float] = []
        self.v: List[float] = []
        self.start: List[float] = []
        self.head = 0

    def clear(self) -> None:
        self.idx.clear()
        self.p.clear()
        self.v.clear()
        self.start.clear()
        self.head = 0

    def add(self, idx: int, p_new: float, v_new: float, current_x: float) -> None:
        if not math.isfinite(v_new):
            return

        if self.head > 64 and self.head * 2 > len(self.idx):
            h = self.head
            self.idx = self.idx[h:]
            self.p = self.p[h:]
            self.v = self.v[h:]
            self.start = self.start[h:]
            self.head = 0

        while len(self.idx) > self.head:
            x0 = _crossing_start_py(
                self.v[-1], self.p[-1], float(v_new), float(p_new), self.a, current_x
            )
            if not math.isfinite(x0):
                return
            if x0 <= self.start[-1] + _EPS:
                self.idx.pop()
                self.p.pop()
                self.v.pop()
                self.start.pop()
            else:
                self.idx.append(int(idx))
                self.p.append(float(p_new))
                self.v.append(float(v_new))
                self.start.append(max(float(x0), float(current_x)))
                return

        if self.head > 0 and len(self.idx) == self.head:
            self.clear()
        self.idx.append(int(idx))
        self.p.append(float(p_new))
        self.v.append(float(v_new))
        self.start.append(float(current_x))

    def query(self, x: float) -> Tuple[float, int]:
        if len(self.idx) <= self.head:
            return NEG_INF, -1
        x = float(x)
        while self.head + 1 < len(self.idx) and self.start[self.head + 1] <= x + _EPS:
            self.head += 1
        dx = x - self.p[self.head]
        if dx < 0.0 and dx > -1.0e-12:
            dx = 0.0
        if dx < 0.0:
            return NEG_INF, -1
        return self.v[self.head] + self.a * math.sqrt(dx), self.idx[self.head]


def _crossing_start_py(
    old_v: float,
    old_p: float,
    new_v: float,
    new_p: float,
    a: float,
    current_x: float,
) -> float:
    if not math.isfinite(new_v):
        return math.inf
    if not math.isfinite(old_v):
        return current_x

    old_now = old_v + a * math.sqrt(max(0.0, current_x - old_p))
    new_now = new_v + a * math.sqrt(max(0.0, current_x - new_p))
    if new_now >= old_now - _EPS:
        return current_x

    d = new_p - old_p
    if d <= _EPS:
        return current_x if new_v >= old_v - _EPS else math.inf

    delta = (new_v - old_v) / a
    if delta <= 0.0:
        return math.inf

    root_d = math.sqrt(d)
    if delta >= root_d - _EPS:
        return current_x

    y = (d - delta * delta) / (2.0 * delta)
    return new_p + y * y


def _etw_banded_dense_python(
    pref_f: np.ndarray,
    pref_g: np.ndarray,
    sqrt_dt: np.ndarray,
    sqrt_ds: np.ndarray,
    skip_f: np.ndarray,
    skip_g: np.ndarray,
    band_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Dense masked DP.  Exact, simple, and supports finite skips."""

    m, n_plus = pref_f.shape
    n = n_plus - 1
    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
    ptr_i = np.full((n + 1, m + 1), -1, dtype=np.int64)
    ptr_j = np.full((n + 1, m + 1), -1, dtype=np.int64)
    V[0, 0] = 0.0

    for i in range(1, n + 1):
        if math.isfinite(V[i - 1, 0]) and math.isfinite(float(skip_f[i - 1])):
            V[i, 0] = V[i - 1, 0] - float(skip_f[i - 1])
            ptr_op[i, 0] = _OP_SKIP_F
            ptr_i[i, 0] = i - 1
            ptr_j[i, 0] = 0
    for j in range(1, m + 1):
        if math.isfinite(V[0, j - 1]) and math.isfinite(float(skip_g[j - 1])):
            V[0, j] = V[0, j - 1] - float(skip_g[j - 1])
            ptr_op[0, j] = _OP_SKIP_G
            ptr_i[0, j] = 0
            ptr_j[0, j] = j - 1

    col_envs = [_Envelope(float(sqrt_dt[j])) for j in range(m)]

    for i in range(1, n + 1):
        row_env = _Envelope(float(sqrt_ds[i - 1]))
        for j in range(1, m + 1):
            q = j - 1
            if bool(band_mask[i - 1, q]):
                x_a = float(pref_f[q, i])
                col_envs[q].add(
                    i - 1,
                    float(pref_f[q, i - 1]),
                    float(V[i - 1, j - 1]),
                    x_a,
                )
                a_val, h_best = col_envs[q].query(x_a)

                x_b = float(pref_g[i - 1, j])
                row_env.add(
                    j - 1,
                    float(pref_g[i - 1, j - 1]),
                    float(V[i - 1, j - 1]),
                    x_b,
                )
                b_val, l_best = row_env.query(x_b)
            else:
                # A vertical block in this column or a B horizontal block in this
                # row may not cross a forbidden interval-pair cell.
                col_envs[q].clear()
                row_env.clear()
                a_val = NEG_INF
                b_val = NEG_INF
                h_best = -1
                l_best = -1

            best = a_val
            best_op = _OP_MANY_F_TO_ONE_G if h_best >= 0 else _OP_NONE
            best_pi = h_best
            best_pj = j - 1

            if b_val > best:
                best = b_val
                best_op = _OP_ONE_F_TO_MANY_G if l_best >= 0 else _OP_NONE
                best_pi = i - 1
                best_pj = l_best

            if math.isfinite(float(skip_f[i - 1])) and math.isfinite(float(V[i - 1, j])):
                cand = float(V[i - 1, j]) - float(skip_f[i - 1])
                if cand > best:
                    best = cand
                    best_op = _OP_SKIP_F
                    best_pi = i - 1
                    best_pj = j

            if math.isfinite(float(skip_g[j - 1])) and math.isfinite(float(V[i, j - 1])):
                cand = float(V[i, j - 1]) - float(skip_g[j - 1])
                if cand > best:
                    best = cand
                    best_op = _OP_SKIP_G
                    best_pi = i
                    best_pj = j - 1

            V[i, j] = best
            ptr_op[i, j] = best_op
            ptr_i[i, j] = best_pi
            ptr_j[i, j] = best_pj

    return V, ptr_op, ptr_i, ptr_j


if _HAVE_NUMBA:

    @njit(cache=True)
    def _crossing_start_numba(
        old_v: float,
        old_p: float,
        new_v: float,
        new_p: float,
        a: float,
        current_x: float,
    ) -> float:
        if not np.isfinite(new_v):
            return np.inf
        if not np.isfinite(old_v):
            return current_x

        dx_old = current_x - old_p
        if dx_old < 0.0:
            if dx_old > -1.0e-12:
                dx_old = 0.0
            else:
                return np.inf
        dx_new = current_x - new_p
        if dx_new < 0.0:
            if dx_new > -1.0e-12:
                dx_new = 0.0
            else:
                return np.inf

        old_now = old_v + a * np.sqrt(dx_old)
        new_now = new_v + a * np.sqrt(dx_new)
        if new_now >= old_now - _EPS:
            return current_x

        d = new_p - old_p
        if d <= _EPS:
            if new_v >= old_v - _EPS:
                return current_x
            return np.inf

        delta = (new_v - old_v) / a
        if delta <= 0.0:
            return np.inf

        root_d = np.sqrt(d)
        if delta >= root_d - _EPS:
            return current_x

        y = (d - delta * delta) / (2.0 * delta)
        return new_p + y * y


    @njit(cache=True)
    def _hull_add_segment_numba(
        cand: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        start: np.ndarray,
        base: int,
        head: int,
        tail: int,
        idx_new: int,
        p_new: float,
        v_new: float,
        a: float,
        current_x: float,
    ) -> Tuple[int, int]:
        if not np.isfinite(v_new):
            return head, tail

        start_new = current_x
        while tail > head:
            last = base + tail - 1
            x0 = _crossing_start_numba(
                v[last], p[last], v_new, p_new, a, current_x
            )
            if not np.isfinite(x0):
                return head, tail
            if x0 <= start[last] + _EPS:
                tail -= 1
            else:
                start_new = x0
                if start_new < current_x:
                    start_new = current_x
                break

        if tail <= head:
            start_new = current_x

        pos = base + tail
        cand[pos] = idx_new
        p[pos] = p_new
        v[pos] = v_new
        start[pos] = start_new
        tail += 1
        return head, tail


    @njit(cache=True)
    def _hull_query_segment_numba(
        cand: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        start: np.ndarray,
        base: int,
        head: int,
        tail: int,
        a: float,
        x: float,
    ) -> Tuple[float, int, int]:
        if tail <= head:
            return NEG_INF, -1, head

        while head + 1 < tail and start[base + head + 1] <= x + _EPS:
            head += 1

        pos = base + head
        dx = x - p[pos]
        if dx < 0.0:
            if dx > -1.0e-12:
                dx = 0.0
            else:
                return NEG_INF, -1, head
        val = v[pos] + a * np.sqrt(dx)
        return val, cand[pos], head


    @njit(cache=True)
    def _hull_add_numba(
        cand: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        start: np.ndarray,
        head: int,
        tail: int,
        idx_new: int,
        p_new: float,
        v_new: float,
        a: float,
        current_x: float,
    ) -> Tuple[int, int]:
        if not np.isfinite(v_new):
            return head, tail

        start_new = current_x
        while tail > head:
            x0 = _crossing_start_numba(
                v[tail - 1], p[tail - 1], v_new, p_new, a, current_x
            )
            if not np.isfinite(x0):
                return head, tail
            if x0 <= start[tail - 1] + _EPS:
                tail -= 1
            else:
                start_new = x0
                if start_new < current_x:
                    start_new = current_x
                break

        if tail <= head:
            start_new = current_x

        cand[tail] = idx_new
        p[tail] = p_new
        v[tail] = v_new
        start[tail] = start_new
        tail += 1
        return head, tail


    @njit(cache=True)
    def _hull_query_numba(
        cand: np.ndarray,
        p: np.ndarray,
        v: np.ndarray,
        start: np.ndarray,
        head: int,
        tail: int,
        a: float,
        x: float,
    ) -> Tuple[float, int, int]:
        if tail <= head:
            return NEG_INF, -1, head

        while head + 1 < tail and start[head + 1] <= x + _EPS:
            head += 1

        dx = x - p[head]
        if dx < 0.0:
            if dx > -1.0e-12:
                dx = 0.0
            else:
                return NEG_INF, -1, head
        val = v[head] + a * np.sqrt(dx)
        return val, cand[head], head


    @njit(cache=True)
    def _etw_banded_sparse_noskip_numba(
        pref_f: np.ndarray,
        pref_g: np.ndarray,
        sqrt_dt: np.ndarray,
        sqrt_ds: np.ndarray,
        row_ptr: np.ndarray,
        row_cols: np.ndarray,
        col_ptr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sparse allowed-cell scan for the no-skip case.

        ``row_cols`` contains the allowed pair columns q for each pair row r.
        The DP state computed for pair cell (r, q) is V[r+1, q+1].
        """

        n = pref_g.shape[0]
        m = pref_f.shape[0]
        total_allowed = row_cols.shape[0]

        V = np.empty((n + 1, m + 1), dtype=np.float64)
        ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
        ptr_i = np.empty((n + 1, m + 1), dtype=np.int64)
        ptr_j = np.empty((n + 1, m + 1), dtype=np.int64)
        for i in range(n + 1):
            for j in range(m + 1):
                V[i, j] = NEG_INF
                ptr_i[i, j] = -1
                ptr_j[i, j] = -1
        V[0, 0] = 0.0

        col_cand = np.empty(total_allowed, dtype=np.int64)
        col_p = np.empty(total_allowed, dtype=np.float64)
        col_v = np.empty(total_allowed, dtype=np.float64)
        col_start = np.empty(total_allowed, dtype=np.float64)
        col_head = np.zeros(m, dtype=np.int64)
        col_tail = np.zeros(m, dtype=np.int64)
        last_allowed_row = np.full(m, -10, dtype=np.int64)

        row_cand = np.empty(m + 1, dtype=np.int64)
        row_p = np.empty(m + 1, dtype=np.float64)
        row_v = np.empty(m + 1, dtype=np.float64)
        row_start = np.empty(m + 1, dtype=np.float64)

        for r in range(n):
            i = r + 1
            row_head = 0
            row_tail = 0
            prev_q = -10
            start = row_ptr[r]
            stop = row_ptr[r + 1]
            for pos in range(start, stop):
                q = row_cols[pos]
                j = q + 1

                # Reset the row envelope at gaps in allowed cells.  This forces
                # one-f-to-many-g blocks to stay inside the band.
                if q != prev_q + 1:
                    row_head = 0
                    row_tail = 0
                prev_q = q

                # Reset this column envelope at gaps in allowed rows.  This
                # forces many-f-to-one-g blocks to stay inside the band.
                if last_allowed_row[q] != r - 1:
                    col_head[q] = 0
                    col_tail[q] = 0
                last_allowed_row[q] = r

                # A-transition: f[h:i] matched to g[q].
                base = col_ptr[q]
                head = col_head[q]
                tail = col_tail[q]
                x_a = pref_f[q, i]
                h_new = i - 1
                head, tail = _hull_add_segment_numba(
                    col_cand,
                    col_p,
                    col_v,
                    col_start,
                    base,
                    head,
                    tail,
                    h_new,
                    pref_f[q, h_new],
                    V[h_new, q],
                    sqrt_dt[q],
                    x_a,
                )
                a_val, h_best, head = _hull_query_segment_numba(
                    col_cand,
                    col_p,
                    col_v,
                    col_start,
                    base,
                    head,
                    tail,
                    sqrt_dt[q],
                    x_a,
                )
                col_head[q] = head
                col_tail[q] = tail

                # B-transition: f[r] matched to g[ell:j].
                x_b = pref_g[r, j]
                ell_new = q
                row_head, row_tail = _hull_add_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    ell_new,
                    pref_g[r, ell_new],
                    V[r, ell_new],
                    sqrt_ds[r],
                    x_b,
                )
                b_val, ell_best, row_head = _hull_query_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    sqrt_ds[r],
                    x_b,
                )

                best = a_val
                best_op = _OP_NONE
                best_pi = -1
                best_pj = -1
                if h_best >= 0:
                    best_op = _OP_MANY_F_TO_ONE_G
                    best_pi = h_best
                    best_pj = q

                if b_val > best:
                    best = b_val
                    best_op = _OP_NONE
                    best_pi = -1
                    best_pj = -1
                    if ell_best >= 0:
                        best_op = _OP_ONE_F_TO_MANY_G
                        best_pi = r
                        best_pj = ell_best

                V[i, j] = best
                ptr_op[i, j] = best_op
                ptr_i[i, j] = best_pi
                ptr_j[i, j] = best_pj

        return V, ptr_op, ptr_i, ptr_j


    @njit(cache=True)
    def _etw_banded_dense_numba(
        pref_f: np.ndarray,
        pref_g: np.ndarray,
        sqrt_dt: np.ndarray,
        sqrt_ds: np.ndarray,
        skip_f: np.ndarray,
        skip_g: np.ndarray,
        band_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Dense masked DP.  Exact and supports finite skips."""

        m = pref_f.shape[0]
        n = pref_f.shape[1] - 1

        V = np.empty((n + 1, m + 1), dtype=np.float64)
        ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
        ptr_i = np.empty((n + 1, m + 1), dtype=np.int64)
        ptr_j = np.empty((n + 1, m + 1), dtype=np.int64)

        for i in range(n + 1):
            for j in range(m + 1):
                V[i, j] = NEG_INF
                ptr_i[i, j] = -1
                ptr_j[i, j] = -1
        V[0, 0] = 0.0

        for i in range(1, n + 1):
            if np.isfinite(V[i - 1, 0]) and np.isfinite(skip_f[i - 1]):
                V[i, 0] = V[i - 1, 0] - skip_f[i - 1]
                ptr_op[i, 0] = _OP_SKIP_F
                ptr_i[i, 0] = i - 1
                ptr_j[i, 0] = 0
        for j in range(1, m + 1):
            if np.isfinite(V[0, j - 1]) and np.isfinite(skip_g[j - 1]):
                V[0, j] = V[0, j - 1] - skip_g[j - 1]
                ptr_op[0, j] = _OP_SKIP_G
                ptr_i[0, j] = 0
                ptr_j[0, j] = j - 1

        # Column envelopes, one for each g interval.  Dense allocation is used
        # only in the finite-skip fallback.
        col_cand = np.empty((m, n + 1), dtype=np.int64)
        col_p = np.empty((m, n + 1), dtype=np.float64)
        col_v = np.empty((m, n + 1), dtype=np.float64)
        col_start = np.empty((m, n + 1), dtype=np.float64)
        col_head = np.zeros(m, dtype=np.int64)
        col_tail = np.zeros(m, dtype=np.int64)

        row_cand = np.empty(m + 1, dtype=np.int64)
        row_p = np.empty(m + 1, dtype=np.float64)
        row_v = np.empty(m + 1, dtype=np.float64)
        row_start = np.empty(m + 1, dtype=np.float64)

        for i in range(1, n + 1):
            row_head = 0
            row_tail = 0
            for j in range(1, m + 1):
                q = j - 1

                a_val = NEG_INF
                b_val = NEG_INF
                h_best = -1
                ell_best = -1

                if band_mask[i - 1, q]:
                    # A-transition: f[h:i] matched to g[q].
                    x_a = pref_f[q, i]
                    h_new = i - 1
                    head = col_head[q]
                    tail = col_tail[q]
                    head, tail = _hull_add_numba(
                        col_cand[q],
                        col_p[q],
                        col_v[q],
                        col_start[q],
                        head,
                        tail,
                        h_new,
                        pref_f[q, h_new],
                        V[h_new, q],
                        sqrt_dt[q],
                        x_a,
                    )
                    a_val, h_best, head = _hull_query_numba(
                        col_cand[q],
                        col_p[q],
                        col_v[q],
                        col_start[q],
                        head,
                        tail,
                        sqrt_dt[q],
                        x_a,
                    )
                    col_head[q] = head
                    col_tail[q] = tail

                    # B-transition: f[i-1] matched to g[ell:j].
                    x_b = pref_g[i - 1, j]
                    ell_new = j - 1
                    row_head, row_tail = _hull_add_numba(
                        row_cand,
                        row_p,
                        row_v,
                        row_start,
                        row_head,
                        row_tail,
                        ell_new,
                        pref_g[i - 1, ell_new],
                        V[i - 1, ell_new],
                        sqrt_ds[i - 1],
                        x_b,
                    )
                    b_val, ell_best, row_head = _hull_query_numba(
                        row_cand,
                        row_p,
                        row_v,
                        row_start,
                        row_head,
                        row_tail,
                        sqrt_ds[i - 1],
                        x_b,
                    )
                else:
                    col_head[q] = 0
                    col_tail[q] = 0
                    row_head = 0
                    row_tail = 0

                best = a_val
                best_op = _OP_NONE
                best_pi = -1
                best_pj = -1
                if h_best >= 0:
                    best_op = _OP_MANY_F_TO_ONE_G
                    best_pi = h_best
                    best_pj = q

                if b_val > best:
                    best = b_val
                    best_op = _OP_NONE
                    best_pi = -1
                    best_pj = -1
                    if ell_best >= 0:
                        best_op = _OP_ONE_F_TO_MANY_G
                        best_pi = i - 1
                        best_pj = ell_best

                if np.isfinite(skip_f[i - 1]) and np.isfinite(V[i - 1, j]):
                    cand_score = V[i - 1, j] - skip_f[i - 1]
                    if cand_score > best:
                        best = cand_score
                        best_op = _OP_SKIP_F
                        best_pi = i - 1
                        best_pj = j

                if np.isfinite(skip_g[j - 1]) and np.isfinite(V[i, j - 1]):
                    cand_score = V[i, j - 1] - skip_g[j - 1]
                    if cand_score > best:
                        best = cand_score
                        best_op = _OP_SKIP_G
                        best_pi = i
                        best_pj = j - 1

                V[i, j] = best
                ptr_op[i, j] = best_op
                ptr_i[i, j] = best_pi
                ptr_j[i, j] = best_pj

        return V, ptr_op, ptr_i, ptr_j

else:

    def _etw_banded_sparse_noskip_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("Numba is not available.")

    def _etw_banded_dense_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("Numba is not available.")


def _traceback(
    V: np.ndarray, ptr_op: np.ndarray, ptr_i: np.ndarray, ptr_j: np.ndarray
) -> Tuple[List[ETWBlock], List[Tuple[int, int]], List[int], List[int]]:
    i = V.shape[0] - 1
    j = V.shape[1] - 1
    blocks_rev: List[ETWBlock] = []
    pairs_rev: List[Tuple[int, int]] = []
    unmatched_f_rev: List[int] = []
    unmatched_g_rev: List[int] = []

    while i > 0 or j > 0:
        op = int(ptr_op[i, j])
        pi = int(ptr_i[i, j])
        pj = int(ptr_j[i, j])
        if op == int(_OP_NONE) or pi < 0 or pj < 0:
            raise RuntimeError(
                f"Traceback failed at state ({i}, {j}); no predecessor stored."
            )
        contribution = float(V[i, j] - V[pi, pj])

        if op == int(_OP_MANY_F_TO_ONE_G):
            block = ETWBlock(
                kind="many_f_to_one_g",
                f_start=pi,
                f_stop=i,
                g_start=j - 1,
                g_stop=j,
                contribution=contribution,
            )
            for r in range(i - 1, pi - 1, -1):
                pairs_rev.append((r, j - 1))
        elif op == int(_OP_ONE_F_TO_MANY_G):
            block = ETWBlock(
                kind="one_f_to_many_g",
                f_start=i - 1,
                f_stop=i,
                g_start=pj,
                g_stop=j,
                contribution=contribution,
            )
            for q in range(j - 1, pj - 1, -1):
                pairs_rev.append((i - 1, q))
        elif op == int(_OP_SKIP_F):
            block = ETWBlock(
                kind="skip_f",
                f_start=i - 1,
                f_stop=i,
                g_start=j,
                g_stop=j,
                contribution=contribution,
            )
            unmatched_f_rev.append(i - 1)
        elif op == int(_OP_SKIP_G):
            block = ETWBlock(
                kind="skip_g",
                f_start=i,
                f_stop=i,
                g_start=j - 1,
                g_stop=j,
                contribution=contribution,
            )
            unmatched_g_rev.append(j - 1)
        else:
            raise RuntimeError(f"Unknown traceback op code {op} at state ({i}, {j}).")

        blocks_rev.append(block)
        i, j = pi, pj

    blocks = list(reversed(blocks_rev))
    pairs = list(reversed(pairs_rev))
    unmatched_f = list(reversed(unmatched_f_rev))
    unmatched_g = list(reversed(unmatched_g_rev))
    return blocks, pairs, unmatched_f, unmatched_g


__all__ = [
    "ETWBlock",
    "ETWResult",
    "build_band_mask",
    "compute_similarity_matrix",
    "etw_align",
    "etw_align_banded",
    "euclidean_similarity_matrix",
    "make_euclidean_distance_band",
    "make_path_band",
    "make_time_band",
]
