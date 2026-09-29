"""Memory-sparse banded Elastic Time Warping.

This module implements the no-skip Hellinger ETW recurrence on a band that is
represented by one contiguous interval of allowed columns per row.  If the band
contains ``K`` allowed interval pairs, the solver

* evaluates only those ``K`` similarities,
* stores only those ``K`` dynamic-programming states and predecessors, and
* uses ``O(K + n + m)`` working memory.

The recurrence inside the band is exact.  The returned score is the unrestricted
ETW optimum only when the band contains an unrestricted optimal matching.

The dense and arbitrary-mask implementations in :mod:`hellinger_etw_banded`
remain useful as reference implementations and for finite skip penalties.  This
module deliberately supports the original no-skip objective only: permitting
skip paths outside a match band requires a different state-closure convention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import math
import warnings

import numpy as np

from hellinger_etw import ETWBlock

try:  # Optional acceleration; the pure-Python solver remains available.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - depends on optional dependency
    njit = None  # type: ignore[assignment]
    _HAVE_NUMBA = False


NEG_INF = -np.inf
_EPS = 1.0e-12


class InfeasibleBandError(ValueError):
    """Raised when a valid band contains no complete no-skip alignment."""


_OP_NONE = np.int8(0)
_OP_MANY_F_TO_ONE_G = np.int8(1)
_OP_ONE_F_TO_MANY_G = np.int8(2)


@dataclass(frozen=True)
class RowBand:
    """A band with one half-open allowed column interval per row.

    Row ``i`` allows exactly the pair cells

    ``lo[i] <= j < hi[i]``.

    This representation is natural for a corridor around a monotone path.  It
    avoids constructing an ``n x m`` Boolean mask and gives constant-time lookup
    of a stored state.
    """

    lo: np.ndarray
    hi: np.ndarray
    m: int

    def __post_init__(self) -> None:
        lo = np.asarray(self.lo, dtype=np.int64)
        hi = np.asarray(self.hi, dtype=np.int64)
        m = int(self.m)
        if lo.ndim != 1 or hi.ndim != 1 or lo.shape != hi.shape:
            raise ValueError("lo and hi must be one-dimensional arrays of equal length.")
        if m <= 0:
            raise ValueError("m must be positive.")
        if np.any(lo < 0) or np.any(hi < lo) or np.any(hi > m):
            raise ValueError("Row-band bounds must satisfy 0 <= lo <= hi <= m.")
        # Store validated copies so later mutation of caller-owned arrays cannot
        # invalidate the representation.
        object.__setattr__(self, "lo", lo.copy())
        object.__setattr__(self, "hi", hi.copy())
        object.__setattr__(self, "m", m)

    @property
    def n(self) -> int:
        return int(self.lo.size)

    @property
    def widths(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def allowed_cells(self) -> int:
        return int(np.sum(self.widths, dtype=np.int64))

    @property
    def density(self) -> float:
        return float(self.allowed_cells) / float(self.n * self.m)

    def contains(self, i: int, j: int) -> bool:
        ii = int(i)
        jj = int(j)
        return 0 <= ii < self.n and int(self.lo[ii]) <= jj < int(self.hi[ii])

    def to_mask(self) -> np.ndarray:
        mask = np.zeros((self.n, self.m), dtype=np.bool_)
        for i in range(self.n):
            mask[i, int(self.lo[i]) : int(self.hi[i])] = True
        return mask

    @classmethod
    def full(cls, n: int, m: int) -> "RowBand":
        if n <= 0 or m <= 0:
            raise ValueError("n and m must be positive.")
        return cls(np.zeros(n, dtype=np.int64), np.full(n, m, dtype=np.int64), m)

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> "RowBand":
        """Convert a Boolean mask whose allowed cells are contiguous in each row.

        Empty rows are retained as ``lo[i] == hi[i] == 0``.  A mask with a gap
        inside a nonempty row is rejected rather than silently filled in.
        """

        arr = np.asarray(mask, dtype=np.bool_)
        if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("mask must be a nonempty two-dimensional array.")
        n, m = arr.shape
        lo = np.zeros(n, dtype=np.int64)
        hi = np.zeros(n, dtype=np.int64)
        for i in range(n):
            cols = np.flatnonzero(arr[i])
            if cols.size == 0:
                continue
            a = int(cols[0])
            b = int(cols[-1]) + 1
            if not bool(np.all(arr[i, a:b])):
                raise ValueError(
                    f"Allowed cells in row {i} are not contiguous; use the dense "
                    "arbitrary-mask solver instead."
                )
            lo[i] = a
            hi[i] = b
        return cls(lo, hi, m)


@dataclass(frozen=True)
class SparseETWDiagnostics:
    """Computational diagnostics for a sparse banded solve."""

    n: int
    m: int
    allowed_cells: int
    state_cells: int
    similarity_evaluations: int
    row_similarity_calls: int
    envelope_slots: int
    used_numba: bool

    @property
    def band_density(self) -> float:
        return float(self.allowed_cells) / float(self.n * self.m)


@dataclass(frozen=True)
class SparseETWResult:
    """Result returned by :func:`etw_align_sparse_banded`."""

    score: float
    blocks: List[ETWBlock]
    pairs: List[Tuple[int, int]]
    unmatched_f: List[int]
    unmatched_g: List[int]
    band: RowBand
    diagnostics: SparseETWDiagnostics
    score_table: Optional[np.ndarray] = None


def etw_align(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    **kwargs: Any,
) -> SparseETWResult:
    """Alias for :func:`etw_align_sparse_banded`."""

    return etw_align_sparse_banded(f_values, f_times, g_values, g_times, **kwargs)


def etw_align_sparse_banded(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    *,
    band: RowBand,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    row_similarity: Optional[Callable[[Any, Sequence[Any]], Sequence[float]]] = None,
    similarity_values: Optional[np.ndarray] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
    use_numba: bool = True,
    check_nonnegative: bool = True,
    return_score_table: bool = False,
) -> SparseETWResult:
    """Solve the no-skip ETW recurrence exactly inside ``band``.

    Exactly one similarity source must be supplied:

    ``similarity``
        Scalar callable ``C(f_i, g_j)``.  It is invoked exactly once for every
        allowed cell in the band.

    ``row_similarity``
        Batched callable ``row_similarity(f_i, g_slice)`` returning one value
        for each allowed ``g`` in row ``i``.  This is the preferred interface
        for numerical data.

    ``similarity_values``
        Flat row-major array of length ``band.allowed_cells``.

    ``similarity_matrix``
        Dense precomputed matrix, retained mainly for validation and backwards
        compatibility.  Only allowed entries are copied into sparse storage.

    Notes
    -----
    The first and final interval-pair cells must be in the band.  Empty rows are
    allowed by :class:`RowBand`, but they make a no-skip full alignment
    infeasible and are therefore rejected here.
    """

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both input series must contain at least one value.")
    if band.n != n or band.m != m:
        raise ValueError(
            f"band has shape {(band.n, band.m)}, but the series have shape {(n, m)}."
        )
    if band.allowed_cells <= 0:
        raise ValueError("The band contains no allowed cells.")
    if np.any(band.widths == 0):
        empty = int(np.flatnonzero(band.widths == 0)[0])
        raise ValueError(
            f"Row {empty} of the band is empty, so a no-skip full alignment is impossible."
        )
    if not band.contains(0, 0):
        raise ValueError("The band must contain the first pair cell (0, 0).")
    if not band.contains(n - 1, m - 1):
        raise ValueError(
            f"The band must contain the final pair cell ({n - 1}, {m - 1})."
        )

    ds, _ = _interval_lengths(f_times, n, end_f, "f_times")
    dt, _ = _interval_lengths(g_times, m, end_g, "g_times")
    row_ptr = _row_ptr(band)
    col_ptr = _column_ptr(band)

    C, similarity_evaluations, row_calls = _allowed_similarity_values(
        f_values,
        g_values,
        band,
        row_ptr,
        similarity=similarity,
        row_similarity=row_similarity,
        similarity_values=similarity_values,
        similarity_matrix=similarity_matrix,
    )
    if not np.all(np.isfinite(C)):
        raise ValueError("All allowed similarities must be finite real numbers.")
    if check_nonnegative:
        min_c = float(np.min(C))
        if min_c < -1.0e-12:
            raise ValueError(
                "Similarities must be nonnegative for this ETW recurrence. "
                f"Minimum allowed value was {min_c}."
            )
        C = np.maximum(C, 0.0)

    if use_numba and _HAVE_NUMBA:
        score, op, pred = _sparse_dp_numba(
            ds,
            dt,
            C,
            band.lo,
            band.hi,
            row_ptr,
            col_ptr,
        )
        used_numba = True
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python sparse solver.",
                RuntimeWarning,
                stacklevel=2,
            )
        score, op, pred = _sparse_dp_python(
            ds,
            dt,
            C,
            band.lo,
            band.hi,
            row_ptr,
        )
        used_numba = False

    final_idx = _pair_index(n - 1, m - 1, band.lo, band.hi, row_ptr)
    final_score = float(score[final_idx])
    if not math.isfinite(final_score):
        raise InfeasibleBandError(
            "No finite band-constrained alignment was found.  The band may be "
            "disconnected or too narrow."
        )

    blocks, pairs = _traceback_sparse(
        n,
        m,
        score,
        op,
        pred,
        band.lo,
        band.hi,
        row_ptr,
    )

    score_table = None
    if return_score_table:
        score_table = _materialize_score_table(n, m, score, band.lo, band.hi, row_ptr)

    diagnostics = SparseETWDiagnostics(
        n=n,
        m=m,
        allowed_cells=band.allowed_cells,
        state_cells=int(score.size),
        similarity_evaluations=similarity_evaluations,
        row_similarity_calls=row_calls,
        envelope_slots=band.allowed_cells + int(np.max(band.widths)),
        used_numba=used_numba,
    )
    return SparseETWResult(
        score=final_score,
        blocks=blocks,
        pairs=pairs,
        unmatched_f=[],
        unmatched_g=[],
        band=band,
        diagnostics=diagnostics,
        score_table=score_table,
    )


def make_path_row_band(
    n: int,
    m: int,
    path: Sequence[Tuple[int, int]],
    *,
    radius: int,
) -> RowBand:
    """Create a row-interval corridor around matched index pairs.

    The Chebyshev-radius rectangles around the supplied pairs are merged within
    each row.  The resulting band has one contiguous interval per row, which is
    the representation needed by the sparse solver.
    """

    if n <= 0 or m <= 0:
        raise ValueError("n and m must be positive.")
    if radius < 0:
        raise ValueError("radius must be nonnegative.")
    lo = np.full(n, m, dtype=np.int64)
    hi = np.zeros(n, dtype=np.int64)
    r = int(radius)
    for i_raw, j_raw in path:
        i = int(i_raw)
        j = int(j_raw)
        if not (0 <= i < n and 0 <= j < m):
            continue
        ra = max(0, i - r)
        rb = min(n, i + r + 1)
        ca = max(0, j - r)
        cb = min(m, j + r + 1)
        for row in range(ra, rb):
            if ca < lo[row]:
                lo[row] = ca
            if cb > hi[row]:
                hi[row] = cb
    empty = hi == 0
    lo[empty] = 0
    return RowBand(lo, hi, m)


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
            f"{name} must contain either {n} start times or {n + 1} breakpoints; "
            f"got {arr.size}."
        )
    if not np.all(np.isfinite(breaks)):
        raise ValueError(f"{name} must contain only finite times.")
    diffs = np.diff(breaks)
    if np.any(diffs <= 0.0):
        raise ValueError(
            f"{name} must be strictly increasing after appending the endpoint."
        )
    return diffs.astype(np.float64), breaks


def _row_ptr(band: RowBand) -> np.ndarray:
    row_ptr = np.empty(band.n + 1, dtype=np.int64)
    row_ptr[0] = 0
    np.cumsum(band.widths, dtype=np.int64, out=row_ptr[1:])
    return row_ptr


def _column_ptr(band: RowBand) -> np.ndarray:
    counts = np.zeros(band.m, dtype=np.int64)
    for i in range(band.n):
        counts[int(band.lo[i]) : int(band.hi[i])] += 1
    ptr = np.empty(band.m + 1, dtype=np.int64)
    ptr[0] = 0
    np.cumsum(counts, dtype=np.int64, out=ptr[1:])
    return ptr


def _allowed_similarity_values(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    band: RowBand,
    row_ptr: np.ndarray,
    *,
    similarity: Optional[Callable[[Any, Any], float]],
    row_similarity: Optional[Callable[[Any, Sequence[Any]], Sequence[float]]],
    similarity_values: Optional[np.ndarray],
    similarity_matrix: Optional[np.ndarray],
) -> Tuple[np.ndarray, int, int]:
    supplied = sum(
        source is not None
        for source in (similarity, row_similarity, similarity_values, similarity_matrix)
    )
    if supplied != 1:
        raise ValueError(
            "Provide exactly one of similarity, row_similarity, similarity_values, "
            "or similarity_matrix."
        )

    K = int(row_ptr[-1])
    if similarity_values is not None:
        arr = np.asarray(similarity_values, dtype=np.float64)
        if arr.ndim != 1 or arr.size != K:
            raise ValueError(
                f"similarity_values must be a one-dimensional length-{K} array."
            )
        return arr.copy(), 0, 0

    if similarity_matrix is not None:
        matrix = np.asarray(similarity_matrix, dtype=np.float64)
        if matrix.shape != (band.n, band.m):
            raise ValueError(
                f"similarity_matrix must have shape {(band.n, band.m)}, got {matrix.shape}."
            )
        out = np.empty(K, dtype=np.float64)
        for i in range(band.n):
            a = int(row_ptr[i])
            b = int(row_ptr[i + 1])
            out[a:b] = matrix[i, int(band.lo[i]) : int(band.hi[i])]
        return out, 0, 0

    out = np.empty(K, dtype=np.float64)
    if row_similarity is not None:
        calls = 0
        for i in range(band.n):
            lo = int(band.lo[i])
            hi = int(band.hi[i])
            try:
                targets = g_values[lo:hi]  # type: ignore[index]
            except (TypeError, AttributeError):
                targets = [g_values[j] for j in range(lo, hi)]
            vals = np.asarray(row_similarity(f_values[i], targets), dtype=np.float64)
            width = hi - lo
            if vals.ndim != 1 or vals.size != width:
                raise ValueError(
                    f"row_similarity returned shape {vals.shape} for row {i}; "
                    f"expected a length-{width} vector."
                )
            out[int(row_ptr[i]) : int(row_ptr[i + 1])] = vals
            calls += 1
        return out, K, calls

    assert similarity is not None
    pos = 0
    for i in range(band.n):
        for j in range(int(band.lo[i]), int(band.hi[i])):
            out[pos] = float(similarity(f_values[i], g_values[j]))
            pos += 1
    return out, K, 0


def _pair_index(
    r: int,
    q: int,
    lo: np.ndarray,
    hi: np.ndarray,
    row_ptr: np.ndarray,
) -> int:
    if r < 0 or r >= lo.size or q < int(lo[r]) or q >= int(hi[r]):
        return -1
    return int(row_ptr[r] + q - int(lo[r]))


def _state_score_python(
    i: int,
    j: int,
    score: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    row_ptr: np.ndarray,
) -> float:
    if i == 0 and j == 0:
        return 0.0
    if i <= 0 or j <= 0:
        return NEG_INF
    idx = _pair_index(i - 1, j - 1, lo, hi, row_ptr)
    return NEG_INF if idx < 0 else float(score[idx])


class _Envelope:
    """Monotone upper envelope for ``v + a*sqrt(x-p)``."""

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
            x0 = _crossing_start_python(
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


def _crossing_start_python(
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


def _sparse_dp_python(
    ds: np.ndarray,
    dt: np.ndarray,
    C: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    row_ptr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(lo.size)
    m = int(dt.size)
    K = int(row_ptr[-1])
    score = np.full(K, NEG_INF, dtype=np.float64)
    op = np.zeros(K, dtype=np.int8)
    pred = np.full(K, -1, dtype=np.int32)

    col_envs = [_Envelope(math.sqrt(float(dt[q]))) for q in range(m)]
    col_x = np.zeros(m, dtype=np.float64)
    last_allowed_row = np.full(m, -2, dtype=np.int64)

    for r in range(n):
        row_env = _Envelope(math.sqrt(float(ds[r])))
        row_x = 0.0
        for q in range(int(lo[r]), int(hi[r])):
            idx = int(row_ptr[r] + q - int(lo[r]))
            c = float(C[idx])
            c2 = c * c
            base_score = _state_score_python(r, q, score, lo, hi, row_ptr)

            if int(last_allowed_row[q]) != r - 1:
                col_envs[q].clear()
                col_x[q] = 0.0
            last_allowed_row[q] = r

            p_a = float(col_x[q])
            x_a = p_a + float(ds[r]) * c2
            col_envs[q].add(r, p_a, base_score, x_a)
            a_val, h_best = col_envs[q].query(x_a)
            col_x[q] = x_a

            p_b = row_x
            x_b = p_b + float(dt[q]) * c2
            row_env.add(q, p_b, base_score, x_b)
            b_val, q_best = row_env.query(x_b)
            row_x = x_b

            if h_best >= 0:
                score[idx] = a_val
                op[idx] = _OP_MANY_F_TO_ONE_G
                pred[idx] = np.int32(h_best)
            if b_val > float(score[idx]) and q_best >= 0:
                score[idx] = b_val
                op[idx] = _OP_ONE_F_TO_MANY_G
                pred[idx] = np.int32(q_best)

    return score, op, pred


if _HAVE_NUMBA:

    @njit(cache=True)
    def _state_score_numba(
        i: int,
        j: int,
        score: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        row_ptr: np.ndarray,
    ) -> float:
        if i == 0 and j == 0:
            return 0.0
        if i <= 0 or j <= 0:
            return NEG_INF
        r = i - 1
        q = j - 1
        if q < lo[r] or q >= hi[r]:
            return NEG_INF
        return score[row_ptr[r] + q - lo[r]]


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

        old_dx = current_x - old_p
        new_dx = current_x - new_p
        if old_dx < 0.0:
            old_dx = 0.0
        if new_dx < 0.0:
            new_dx = 0.0
        old_now = old_v + a * np.sqrt(old_dx)
        new_now = new_v + a * np.sqrt(new_dx)
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
        return v[pos] + a * np.sqrt(dx), cand[pos], head


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
        return v[head] + a * np.sqrt(dx), cand[head], head


    @njit(cache=True)
    def _sparse_dp_numba(
        ds: np.ndarray,
        dt: np.ndarray,
        C: np.ndarray,
        lo: np.ndarray,
        hi: np.ndarray,
        row_ptr: np.ndarray,
        col_ptr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = lo.size
        m = dt.size
        K = C.size
        score = np.empty(K, dtype=np.float64)
        op = np.zeros(K, dtype=np.int8)
        pred = np.empty(K, dtype=np.int32)
        for k in range(K):
            score[k] = NEG_INF
            pred[k] = -1

        col_cand = np.empty(K, dtype=np.int32)
        col_p = np.empty(K, dtype=np.float64)
        col_v = np.empty(K, dtype=np.float64)
        col_start = np.empty(K, dtype=np.float64)
        col_head = np.zeros(m, dtype=np.int64)
        col_tail = np.zeros(m, dtype=np.int64)
        col_x = np.zeros(m, dtype=np.float64)
        last_allowed_row = np.full(m, -2, dtype=np.int64)

        max_width = 0
        for r in range(n):
            width = hi[r] - lo[r]
            if width > max_width:
                max_width = width
        row_cand = np.empty(max_width, dtype=np.int32)
        row_p = np.empty(max_width, dtype=np.float64)
        row_v = np.empty(max_width, dtype=np.float64)
        row_start = np.empty(max_width, dtype=np.float64)

        for r in range(n):
            row_head = 0
            row_tail = 0
            row_x = 0.0
            for q in range(lo[r], hi[r]):
                idx = row_ptr[r] + q - lo[r]
                c = C[idx]
                c2 = c * c
                base_score = _state_score_numba(r, q, score, lo, hi, row_ptr)

                if last_allowed_row[q] != r - 1:
                    col_head[q] = 0
                    col_tail[q] = 0
                    col_x[q] = 0.0
                last_allowed_row[q] = r

                p_a = col_x[q]
                x_a = p_a + ds[r] * c2
                base = col_ptr[q]
                head = col_head[q]
                tail = col_tail[q]
                head, tail = _hull_add_segment_numba(
                    col_cand,
                    col_p,
                    col_v,
                    col_start,
                    base,
                    head,
                    tail,
                    r,
                    p_a,
                    base_score,
                    np.sqrt(dt[q]),
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
                    np.sqrt(dt[q]),
                    x_a,
                )
                col_head[q] = head
                col_tail[q] = tail
                col_x[q] = x_a

                p_b = row_x
                x_b = p_b + dt[q] * c2
                row_head, row_tail = _hull_add_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    q,
                    p_b,
                    base_score,
                    np.sqrt(ds[r]),
                    x_b,
                )
                b_val, q_best, row_head = _hull_query_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    np.sqrt(ds[r]),
                    x_b,
                )
                row_x = x_b

                if h_best >= 0:
                    score[idx] = a_val
                    op[idx] = _OP_MANY_F_TO_ONE_G
                    pred[idx] = h_best
                if q_best >= 0 and b_val > score[idx]:
                    score[idx] = b_val
                    op[idx] = _OP_ONE_F_TO_MANY_G
                    pred[idx] = q_best

        return score, op, pred

else:

    def _sparse_dp_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("Numba is not available.")


def _traceback_sparse(
    n: int,
    m: int,
    score: np.ndarray,
    op: np.ndarray,
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    row_ptr: np.ndarray,
) -> Tuple[List[ETWBlock], List[Tuple[int, int]]]:
    i = n
    j = m
    blocks_rev: List[ETWBlock] = []
    pairs_rev: List[Tuple[int, int]] = []

    while i > 0 or j > 0:
        idx = _pair_index(i - 1, j - 1, lo, hi, row_ptr)
        if idx < 0:
            raise RuntimeError(
                f"Traceback reached unstored state ({i}, {j}); the band is inconsistent."
            )
        code = int(op[idx])
        predecessor = int(pred[idx])
        current_score = float(score[idx])

        if code == int(_OP_MANY_F_TO_ONE_G):
            h = predecessor
            pi, pj = h, j - 1
            previous = _state_score_python(pi, pj, score, lo, hi, row_ptr)
            if not math.isfinite(previous):
                raise RuntimeError(
                    f"Invalid Type-F predecessor ({pi}, {pj}) at state ({i}, {j})."
                )
            blocks_rev.append(
                ETWBlock(
                    kind="many_f_to_one_g",
                    f_start=h,
                    f_stop=i,
                    g_start=j - 1,
                    g_stop=j,
                    contribution=current_score - previous,
                )
            )
            for r in range(i - 1, h - 1, -1):
                pairs_rev.append((r, j - 1))
        elif code == int(_OP_ONE_F_TO_MANY_G):
            q = predecessor
            pi, pj = i - 1, q
            previous = _state_score_python(pi, pj, score, lo, hi, row_ptr)
            if not math.isfinite(previous):
                raise RuntimeError(
                    f"Invalid Type-G predecessor ({pi}, {pj}) at state ({i}, {j})."
                )
            blocks_rev.append(
                ETWBlock(
                    kind="one_f_to_many_g",
                    f_start=i - 1,
                    f_stop=i,
                    g_start=q,
                    g_stop=j,
                    contribution=current_score - previous,
                )
            )
            for col in range(j - 1, q - 1, -1):
                pairs_rev.append((i - 1, col))
        else:
            raise RuntimeError(
                f"Traceback failed at state ({i}, {j}); no predecessor was stored."
            )

        i, j = pi, pj

    return list(reversed(blocks_rev)), list(reversed(pairs_rev))


def _materialize_score_table(
    n: int,
    m: int,
    score: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    row_ptr: np.ndarray,
) -> np.ndarray:
    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    V[0, 0] = 0.0
    for r in range(n):
        a = int(row_ptr[r])
        b = int(row_ptr[r + 1])
        V[r + 1, int(lo[r]) + 1 : int(hi[r]) + 1] = score[a:b]
    return V


__all__ = [
    "InfeasibleBandError",
    "RowBand",
    "SparseETWDiagnostics",
    "SparseETWResult",
    "etw_align",
    "etw_align_sparse_banded",
    "make_path_row_band",
]
