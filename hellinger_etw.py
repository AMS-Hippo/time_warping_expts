"""
Fast Elastic Time Warping with Hellinger block scores.

This module implements the dynamic program from the Hellinger Elastic
Time Warping recurrence, but evaluates each row/column maximization with a
monotone upper-envelope data structure rather than a linear scan over all
predecessors.  The resulting dense algorithm is O(n*m) time after the
similarity matrix has been computed.

Main entry point
----------------
    etw_align(f_values, f_times, g_values, g_times, similarity=...)

The input time arrays may be either
    * start times of length n / m, in which case end_f/end_g are appended, or
    * interval breakpoints of length n+1 / m+1.

The output contains the optimal score, a block traceback, expanded matched
pairs, and unmatched indices.
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

# Traceback op codes.
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
    """Result returned by :func:`etw_align`."""

    score: float
    blocks: List[ETWBlock]
    pairs: List[Tuple[int, int]]
    unmatched_f: List[int]
    unmatched_g: List[int]
    score_table: Optional[np.ndarray] = None


def etw_align(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    *,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
    skip_f_penalty: Optional[Sequence[float] | float] = None,
    skip_g_penalty: Optional[Sequence[float] | float] = None,
    use_numba: bool = True,
    check_nonnegative: bool = True,
    return_score_table: bool = False,
) -> ETWResult:
    """Align two timestamped series under the Hellinger ETW objective.

    Parameters
    ----------
    f_values, g_values:
        Series values.  They may be arbitrary Python objects if ``similarity``
        is provided, because the similarity matrix is computed before the
        numeric dynamic program runs.

    f_times, g_times:
        Either start times of the intervals, with lengths ``len(f_values)`` and
        ``len(g_values)``, or full breakpoint arrays, with lengths
        ``len(f_values)+1`` and ``len(g_values)+1``.  If start times are given,
        ``end_f`` and ``end_g`` are appended; when omitted, the default endpoint
        is 1.0, matching the normalization in the paper.

    similarity:
        Callable ``C(x, y) -> nonnegative float``.  Use this for arbitrary
        Python objects.  This is ignored when ``similarity_matrix`` is supplied.

    similarity_matrix:
        Optional precomputed matrix ``C[i, j]`` with shape ``(n, m)``.  This is
        the fastest interface, and is the natural hook for later vectorized or
        batched similarity functions.

    skip_f_penalty, skip_g_penalty:
        Nonnegative scalar or per-index penalties for leaving a point unmatched.
        ``None`` means ``np.inf``, i.e. skipping is disallowed by default.
        The DP maximizes score, so a skip contributes ``-penalty``.

        Important modeling note: for the original nonnegative Hellinger
        similarity objective, matching an extra interval with zero similarity is
        usually no worse than paying a positive skip penalty.  These transitions
        are therefore best viewed as the mechanical edit-operation extension.
        If gaps should be preferred over bad matches, use a signed/centered
        scoring model or add an explicit per-matched-vertex cost in a later
        variant.

    use_numba:
        Use the Numba implementation when available.  The similarity callable is
        still evaluated in ordinary Python unless ``similarity_matrix`` is
        supplied.

    check_nonnegative:
        If true, reject negative similarities.  The Hellinger block formula uses
        squared similarities and assumes a nonnegative similarity coefficient.

    return_score_table:
        Include the full DP table in the result.  This is useful for debugging
        or explanations, but can be memory-heavy.

    Returns
    -------
    ETWResult
        Contains the optimal score, traceback blocks, expanded matched pairs,
        and unmatched indices.
    """

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both input series must contain at least one value.")

    ds, _f_breaks = _interval_lengths(f_times, n, end_f, "f_times")
    dt, _g_breaks = _interval_lengths(g_times, m, end_g, "g_times")

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
        # Remove tiny negative roundoff if a vectorized C produced it.
        C = np.maximum(C, 0.0)

    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")

    pref_f, pref_g = _prefix_tables(C, ds, dt)
    sqrt_ds = np.sqrt(ds)
    sqrt_dt = np.sqrt(dt)

    if use_numba and _HAVE_NUMBA:
        V, ptr_op, ptr_i, ptr_j = _etw_dp_numba(
            pref_f, pref_g, sqrt_dt, sqrt_ds, skip_f, skip_g
        )
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python "
                "envelope implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        V, ptr_op, ptr_i, ptr_j = _etw_dp_python(
            pref_f, pref_g, sqrt_dt, sqrt_ds, skip_f, skip_g
        )

    score = float(V[n, m])
    if not np.isfinite(score):
        raise ValueError(
            "No finite alignment was found.  This usually means skipping is "
            "disallowed and all match paths were made invalid."
        )

    blocks, pairs, unmatched_f, unmatched_g = _traceback(V, ptr_op, ptr_i, ptr_j)
    return ETWResult(
        score=score,
        blocks=blocks,
        pairs=pairs,
        unmatched_f=unmatched_f,
        unmatched_g=unmatched_g,
        score_table=V if return_score_table else None,
    )


def compute_similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    similarity: Callable[[Any, Any], float],
) -> np.ndarray:
    """Compute ``C[i, j] = similarity(f_values[i], g_values[j])``.

    This intentionally uses a simple Python double loop so that ``similarity``
    can be any Python callable.  For numeric data, a vectorized or JIT-compiled
    similarity matrix builder can be substituted and passed to ``etw_align`` via
    the ``similarity_matrix`` argument.
    """

    n = len(f_values)
    m = len(g_values)
    C = np.empty((n, m), dtype=np.float64)
    for i, x in enumerate(f_values):
        for j, y in enumerate(g_values):
            C[i, j] = float(similarity(x, y))
    return C


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


def _block_score_from_a(prefix_right: float, prefix_left: float, scale_sqrt: float) -> float:
    # Numerical guard for tiny negative roundoff in prefix differences.
    diff = prefix_right - prefix_left
    if diff < 0.0 and diff > -1.0e-12:
        diff = 0.0
    return scale_sqrt * math.sqrt(diff) if diff >= 0.0 else NEG_INF


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

    def add(self, idx: int, p_new: float, v_new: float, current_x: float) -> None:
        if not math.isfinite(v_new):
            return

        # Compact occasionally; this keeps memory bounded without changing the
        # amortized O(number of insertions) behavior.
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

        # Empty active envelope.  Earlier x-values have already been queried,
        # so current_x is the earliest relevant breakpoint.
        if self.head > 0 and len(self.idx) == self.head:
            self.idx = []
            self.p = []
            self.v = []
            self.start = []
            self.head = 0
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
    """Earliest x >= current_x where new arc is at least old arc.

    Both arcs have the form v + a*sqrt(x - p), with old_p <= new_p in normal
    use.  Queries are monotone, so starts earlier than current_x are truncated
    to current_x by the caller.
    """

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


def _etw_dp_python(
    pref_f: np.ndarray,
    pref_g: np.ndarray,
    sqrt_dt: np.ndarray,
    sqrt_ds: np.ndarray,
    skip_f: np.ndarray,
    skip_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    m, n_plus = pref_f.shape
    n = n_plus - 1
    n2, m_plus = pref_g.shape
    if n2 != n:
        raise ValueError("Internal prefix table shape mismatch.")

    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
    ptr_i = np.full((n + 1, m + 1), -1, dtype=np.int64)
    ptr_j = np.full((n + 1, m + 1), -1, dtype=np.int64)
    V[0, 0] = 0.0

    # Boundary skips.
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
            jj = j - 1

            # A-transition: f[h:i] matched to g[j-1].
            x_a = float(pref_f[jj, i])
            col_envs[jj].add(
                i - 1,
                float(pref_f[jj, i - 1]),
                float(V[i - 1, j - 1]),
                x_a,
            )
            a_val, h_best = col_envs[jj].query(x_a)

            # B-transition: f[i-1] matched to g[l:j].
            x_b = float(pref_g[i - 1, j])
            row_env.add(
                j - 1,
                float(pref_g[i - 1, j - 1]),
                float(V[i - 1, j - 1]),
                x_b,
            )
            b_val, l_best = row_env.query(x_b)

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
    def _etw_dp_numba(
        pref_f: np.ndarray,
        pref_g: np.ndarray,
        sqrt_dt: np.ndarray,
        sqrt_ds: np.ndarray,
        skip_f: np.ndarray,
        skip_g: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

        # Column envelopes, one for each g interval.  Each active arc stores
        # idx=h, p=P_h, v=V[h, j-1], start=left breakpoint on the envelope.
        col_cand = np.empty((m, n + 1), dtype=np.int64)
        col_p = np.empty((m, n + 1), dtype=np.float64)
        col_v = np.empty((m, n + 1), dtype=np.float64)
        col_start = np.empty((m, n + 1), dtype=np.float64)
        col_head = np.zeros(m, dtype=np.int64)
        col_tail = np.zeros(m, dtype=np.int64)

        # Row envelope work arrays.  Reused for each row.
        row_cand = np.empty(m + 1, dtype=np.int64)
        row_p = np.empty(m + 1, dtype=np.float64)
        row_v = np.empty(m + 1, dtype=np.float64)
        row_start = np.empty(m + 1, dtype=np.float64)

        for i in range(1, n + 1):
            row_head = 0
            row_tail = 0
            for j in range(1, m + 1):
                jj = j - 1

                # A-transition: f[h:i] matched to g[j-1].
                x_a = pref_f[jj, i]
                h_new = i - 1
                head = col_head[jj]
                tail = col_tail[jj]
                head, tail = _hull_add_numba(
                    col_cand[jj],
                    col_p[jj],
                    col_v[jj],
                    col_start[jj],
                    head,
                    tail,
                    h_new,
                    pref_f[jj, h_new],
                    V[h_new, j - 1],
                    sqrt_dt[jj],
                    x_a,
                )
                a_val, h_best, head = _hull_query_numba(
                    col_cand[jj],
                    col_p[jj],
                    col_v[jj],
                    col_start[jj],
                    head,
                    tail,
                    sqrt_dt[jj],
                    x_a,
                )
                col_head[jj] = head
                col_tail[jj] = tail

                # B-transition: f[i-1] matched to g[l:j].
                x_b = pref_g[i - 1, j]
                l_new = j - 1
                row_head, row_tail = _hull_add_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    l_new,
                    pref_g[i - 1, l_new],
                    V[i - 1, l_new],
                    sqrt_ds[i - 1],
                    x_b,
                )
                b_val, l_best, row_head = _hull_query_numba(
                    row_cand,
                    row_p,
                    row_v,
                    row_start,
                    row_head,
                    row_tail,
                    sqrt_ds[i - 1],
                    x_b,
                )

                best = a_val
                best_op = _OP_NONE
                best_pi = -1
                best_pj = -1
                if h_best >= 0:
                    best_op = _OP_MANY_F_TO_ONE_G
                    best_pi = h_best
                    best_pj = j - 1

                if b_val > best:
                    best = b_val
                    best_op = _OP_NONE
                    best_pi = -1
                    best_pj = -1
                    if l_best >= 0:
                        best_op = _OP_ONE_F_TO_MANY_G
                        best_pi = i - 1
                        best_pj = l_best

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

    def _etw_dp_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
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


# A deliberately slow reference implementation for tests and small examples.
def _etw_score_bruteforce(
    similarity_matrix: np.ndarray,
    f_times: Sequence[float],
    g_times: Sequence[float],
    *,
    skip_f_penalty: Optional[Sequence[float] | float] = None,
    skip_g_penalty: Optional[Sequence[float] | float] = None,
    end_f: Optional[float] = None,
    end_g: Optional[float] = None,
) -> float:
    """Cubic reference score used for validation on small inputs."""

    C = np.asarray(similarity_matrix, dtype=np.float64)
    n, m = C.shape
    ds, _ = _interval_lengths(f_times, n, end_f, "f_times")
    dt, _ = _interval_lengths(g_times, m, end_g, "g_times")
    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")
    pref_f, pref_g = _prefix_tables(C, ds, dt)

    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    V[0, 0] = 0.0
    for i in range(1, n + 1):
        if math.isfinite(V[i - 1, 0]) and math.isfinite(float(skip_f[i - 1])):
            V[i, 0] = V[i - 1, 0] - float(skip_f[i - 1])
    for j in range(1, m + 1):
        if math.isfinite(V[0, j - 1]) and math.isfinite(float(skip_g[j - 1])):
            V[0, j] = V[0, j - 1] - float(skip_g[j - 1])

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = NEG_INF
            for h in range(0, i):
                if math.isfinite(V[h, j - 1]):
                    block = _block_score_from_a(pref_f[j - 1, i], pref_f[j - 1, h], math.sqrt(dt[j - 1]))
                    best = max(best, V[h, j - 1] + block)
            for ell in range(0, j):
                if math.isfinite(V[i - 1, ell]):
                    block = _block_score_from_a(pref_g[i - 1, j], pref_g[i - 1, ell], math.sqrt(ds[i - 1]))
                    best = max(best, V[i - 1, ell] + block)
            if math.isfinite(float(skip_f[i - 1])) and math.isfinite(V[i - 1, j]):
                best = max(best, V[i - 1, j] - float(skip_f[i - 1]))
            if math.isfinite(float(skip_g[j - 1])) and math.isfinite(V[i, j - 1]):
                best = max(best, V[i, j - 1] - float(skip_g[j - 1]))
            V[i, j] = best
    return float(V[n, m])


__all__ = [
    "ETWBlock",
    "ETWResult",
    "compute_similarity_matrix",
    "etw_align",
]
