"""Independent small-instance references used only by the test suite."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Tuple

import numpy as np


NEG_INF = -np.inf
F = 1
G = 2


@dataclass(frozen=True)
class ReferenceResult:
    score: float
    score_table: np.ndarray
    op: np.ndarray
    pred: np.ndarray


def brute_force_banded_score(
    C: np.ndarray,
    ds: np.ndarray,
    dt: np.ndarray,
    mask: np.ndarray,
) -> ReferenceResult:
    """Cubic exact recurrence for an arbitrary pair-cell mask.

    This implementation deliberately does not use prefix sums or envelope code.
    A candidate block is accumulated one cell at a time and the scan stops at
    the first forbidden cell, so every block is contained in the mask.
    """

    C = np.asarray(C, dtype=np.float64)
    ds = np.asarray(ds, dtype=np.float64)
    dt = np.asarray(dt, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.bool_)
    n, m = C.shape
    assert mask.shape == (n, m)
    assert ds.shape == (n,)
    assert dt.shape == (m,)

    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    op = np.zeros((n + 1, m + 1), dtype=np.int8)
    pred = np.full((n + 1, m + 1), -1, dtype=np.int64)
    V[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if not mask[i - 1, j - 1]:
                continue

            best = NEG_INF
            best_op = 0
            best_pred = -1

            accum = 0.0
            for h in range(i - 1, -1, -1):
                if not mask[h, j - 1]:
                    break
                accum += ds[h] * C[h, j - 1] ** 2
                if np.isfinite(V[h, j - 1]):
                    cand = V[h, j - 1] + math.sqrt(dt[j - 1] * accum)
                    if cand > best:
                        best = cand
                        best_op = F
                        best_pred = h

            accum = 0.0
            for q in range(j - 1, -1, -1):
                if not mask[i - 1, q]:
                    break
                accum += dt[q] * C[i - 1, q] ** 2
                if np.isfinite(V[i - 1, q]):
                    cand = V[i - 1, q] + math.sqrt(ds[i - 1] * accum)
                    # Match the production tie convention: retain F on ties.
                    if cand > best:
                        best = cand
                        best_op = G
                        best_pred = q

            V[i, j] = best
            op[i, j] = best_op
            pred[i, j] = best_pred

    return ReferenceResult(float(V[n, m]), V, op, pred)


def random_monotone_path(n: int, m: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """Return pair cells on a random up/right path from (0,0) to (n-1,m-1)."""

    i = 0
    j = 0
    path = [(0, 0)]
    while i < n - 1 or j < m - 1:
        if i == n - 1:
            j += 1
        elif j == m - 1:
            i += 1
        elif rng.random() < 0.5:
            i += 1
        else:
            j += 1
        path.append((i, j))
    return path


def mask_from_path_with_noise(
    n: int,
    m: int,
    rng: np.random.Generator,
    *,
    extra_probability: float = 0.25,
) -> np.ndarray:
    mask = rng.random((n, m)) < extra_probability
    for i, j in random_monotone_path(n, m, rng):
        mask[i, j] = True
    return mask
