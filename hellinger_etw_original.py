"""
Reference implementation of the original cubic Elastic Time Warping recurrence.

This module intentionally implements the page-10 Hellinger ETW dynamic program
by directly scanning every possible block length.  It is meant to be a simple,
independent correctness oracle for the faster envelope-based implementation.

Main entry point
----------------
    etw_align(f_values, f_times, g_values, g_times, similarity=...)

The public API mirrors ``hellinger_etw.py`` as closely as possible.  With the
skip penalties left at their default ``None`` values, this is exactly the
paper's recurrence.  If finite skip penalties are supplied, two extra edit-style
skip transitions are enabled; these transitions are not part of the paper, but
match the optional extension in the fast implementation.

Complexity
----------
After the similarity matrix is known, this implementation takes
    O(n*m*(n + m)) time
and
    O(n*m) memory,
matching the complexity stated for the original algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import math
import warnings

import numpy as np

try:  # Optional acceleration.  The algorithm is still the same cubic scan.
    from numba import njit

    _HAVE_NUMBA = True
except Exception:  # pragma: no cover - depends on optional dependency
    njit = None  # type: ignore[assignment]
    _HAVE_NUMBA = False


NEG_INF = -np.inf

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
    """Align two timestamped series using the original cubic ETW recurrence.

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
        Callable ``C(x, y) -> nonnegative float``.  Ignored when
        ``similarity_matrix`` is supplied.

    similarity_matrix:
        Optional precomputed matrix ``C[i, j]`` with shape ``(n, m)``.

    skip_f_penalty, skip_g_penalty:
        Nonnegative scalar or per-index penalties for leaving a point unmatched.
        ``None`` means ``np.inf``, i.e. skipping is disallowed by default.
        Finite penalties enable a simple edit-style extension of the paper's
        recurrence.

    use_numba:
        Use a Numba-compiled version of the same cubic loops when available.
        Set this to ``False`` for the clearest pure-Python execution path.

    check_nonnegative:
        If true, reject negative similarities.  The Hellinger block formula uses
        squared similarities and assumes a nonnegative similarity coefficient.

    return_score_table:
        Include the full DP table in the result.

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
        C = np.maximum(C, 0.0)

    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")

    C2 = C * C

    if use_numba and _HAVE_NUMBA:
        V, ptr_op, ptr_i, ptr_j = _etw_dp_original_numba(C2, ds, dt, skip_f, skip_g)
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python "
                "original cubic implementation.",
                RuntimeWarning,
                stacklevel=2,
            )
        V, ptr_op, ptr_i, ptr_j = _etw_dp_original_python(C2, ds, dt, skip_f, skip_g)

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


# A second public name is useful when importing both modules in one test file:
#     from hellinger_etw import etw_align as etw_fast
#     from hellinger_etw_original import etw_align_original as etw_slow
etw_align_original = etw_align


def compute_similarity_matrix(
    f_values: Sequence[Any],
    g_values: Sequence[Any],
    similarity: Callable[[Any, Any], float],
) -> np.ndarray:
    """Compute ``C[i, j] = similarity(f_values[i], g_values[j])``."""

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


def _safe_block_score(scale: float, accumulated_weighted_c2: float) -> float:
    # A tiny negative value can only be roundoff, but the direct accumulation
    # should normally be nonnegative.
    x = scale * accumulated_weighted_c2
    if x < 0.0 and x > -1.0e-12:
        x = 0.0
    if x < 0.0:
        return NEG_INF
    return math.sqrt(x)


def _etw_dp_original_python(
    C2: np.ndarray,
    ds: np.ndarray,
    dt: np.ndarray,
    skip_f: np.ndarray,
    skip_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Direct page-10 recurrence with explicit scans over every k and p."""

    n, m = C2.shape
    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
    ptr_i = np.full((n + 1, m + 1), -1, dtype=np.int64)
    ptr_j = np.full((n + 1, m + 1), -1, dtype=np.int64)
    V[0, 0] = 0.0

    # Boundary states are only reachable when skip penalties are finite.  With
    # default infinite penalties, the only finite boundary state is V[0, 0],
    # exactly as in the original paper.
    for i in range(1, n + 1):
        if math.isfinite(float(V[i - 1, 0])) and math.isfinite(float(skip_f[i - 1])):
            V[i, 0] = V[i - 1, 0] - float(skip_f[i - 1])
            ptr_op[i, 0] = _OP_SKIP_F
            ptr_i[i, 0] = i - 1
            ptr_j[i, 0] = 0

    for j in range(1, m + 1):
        if math.isfinite(float(V[0, j - 1])) and math.isfinite(float(skip_g[j - 1])):
            V[0, j] = V[0, j - 1] - float(skip_g[j - 1])
            ptr_op[0, j] = _OP_SKIP_G
            ptr_i[0, j] = 0
            ptr_j[0, j] = j - 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = NEG_INF
            best_op = _OP_NONE
            best_pi = -1
            best_pj = -1

            # F_k(i, j): match f[i-k], ..., f[i-1] to g[j-1].
            # This loop updates the sum in the same incremental spirit as the
            # paper's cubic algorithm.
            running = 0.0
            for k in range(1, i + 1):
                r = i - k
                running += float(ds[r]) * float(C2[r, j - 1])
                pi = i - k
                pj = j - 1
                prev = float(V[pi, pj])
                if math.isfinite(prev):
                    block = _safe_block_score(float(dt[j - 1]), running)
                    cand = prev + block
                    if cand > best:
                        best = cand
                        best_op = _OP_MANY_F_TO_ONE_G
                        best_pi = pi
                        best_pj = pj

            # G_p(i, j): match f[i-1] to g[j-p], ..., g[j-1].
            running = 0.0
            for p in range(1, j + 1):
                q = j - p
                running += float(dt[q]) * float(C2[i - 1, q])
                pi = i - 1
                pj = j - p
                prev = float(V[pi, pj])
                if math.isfinite(prev):
                    block = _safe_block_score(float(ds[i - 1]), running)
                    cand = prev + block
                    if cand > best:
                        best = cand
                        best_op = _OP_ONE_F_TO_MANY_G
                        best_pi = pi
                        best_pj = pj

            # Optional edit-style extension: leave f[i-1] unmatched.
            if math.isfinite(float(skip_f[i - 1])) and math.isfinite(float(V[i - 1, j])):
                cand = float(V[i - 1, j]) - float(skip_f[i - 1])
                if cand > best:
                    best = cand
                    best_op = _OP_SKIP_F
                    best_pi = i - 1
                    best_pj = j

            # Optional edit-style extension: leave g[j-1] unmatched.
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
    def _etw_dp_original_numba(
        C2: np.ndarray,
        ds: np.ndarray,
        dt: np.ndarray,
        skip_f: np.ndarray,
        skip_g: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Numba-compiled version of the same direct cubic recurrence."""

        n = C2.shape[0]
        m = C2.shape[1]
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

        for i in range(1, n + 1):
            for j in range(1, m + 1):
                best = NEG_INF
                best_op = _OP_NONE
                best_pi = -1
                best_pj = -1

                # F_k(i, j): match f[i-k:i] to g[j-1].
                running = 0.0
                for k in range(1, i + 1):
                    r = i - k
                    running += ds[r] * C2[r, j - 1]
                    pi = i - k
                    pj = j - 1
                    prev = V[pi, pj]
                    if np.isfinite(prev):
                        x = dt[j - 1] * running
                        if x < 0.0 and x > -1.0e-12:
                            x = 0.0
                        if x >= 0.0:
                            cand = prev + np.sqrt(x)
                            if cand > best:
                                best = cand
                                best_op = _OP_MANY_F_TO_ONE_G
                                best_pi = pi
                                best_pj = pj

                # G_p(i, j): match f[i-1] to g[j-p:j].
                running = 0.0
                for p in range(1, j + 1):
                    q = j - p
                    running += dt[q] * C2[i - 1, q]
                    pi = i - 1
                    pj = j - p
                    prev = V[pi, pj]
                    if np.isfinite(prev):
                        x = ds[i - 1] * running
                        if x < 0.0 and x > -1.0e-12:
                            x = 0.0
                        if x >= 0.0:
                            cand = prev + np.sqrt(x)
                            if cand > best:
                                best = cand
                                best_op = _OP_ONE_F_TO_MANY_G
                                best_pi = pi
                                best_pj = pj

                if np.isfinite(skip_f[i - 1]) and np.isfinite(V[i - 1, j]):
                    cand = V[i - 1, j] - skip_f[i - 1]
                    if cand > best:
                        best = cand
                        best_op = _OP_SKIP_F
                        best_pi = i - 1
                        best_pj = j

                if np.isfinite(skip_g[j - 1]) and np.isfinite(V[i, j - 1]):
                    cand = V[i, j - 1] - skip_g[j - 1]
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

else:

    def _etw_dp_original_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
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
    "compute_similarity_matrix",
    "etw_align",
    "etw_align_original",
]
