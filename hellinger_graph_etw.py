"""
Timestamped tree/DAG alignment with Hellinger-style node weights.

This module is the graph analogue of ``hellinger_etw.py``.  It implements two
closely related dynamic programs:

1. ``tree_etw_align`` aligns two rooted, timestamped ordered forests.  Children
   are ordered by timestamp, so each node-pair subproblem aligns two child
   sequences.  This is a whole-forest objective: every matched node recursively
   aligns its children, and skipped children are skipped as whole subtrees.

2. ``dag_etw_align`` aligns one directed trajectory through each timestamped
   DAG.  It is a product-DAG dynamic program, analogous to pairwise alignment on
   partial-order graphs.  It is exact for that path/trajectory objective, not
   for unrestricted graph edit distance.

Both entry points accept either an arbitrary Python similarity callable or a
precomputed similarity matrix.  They maximize

    sum sqrt(weight_f[u] * weight_g[v]) * C(f_u, g_v) - skip penalties,

where the default node weights are 1.0 and the default skip penalties are
infinite, i.e. skipping is disallowed unless requested.

The timestamps are used to validate and order the graph.  Every edge must point
from an earlier time to a later time when ``check_time_order=True``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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

# Traceback op codes shared by the small child-list aligners and DAG DP.
_OP_NONE = np.int8(0)
_OP_MATCH = np.int8(1)
_OP_SKIP_F = np.int8(2)
_OP_SKIP_G = np.int8(3)


@dataclass(frozen=True)
class GraphETWBlock:
    """One operation in a graph alignment traceback.

    ``kind`` is one of:
        * ``"match"``
        * ``"skip_f"``
        * ``"skip_g"``
        * ``"skip_f_subtree"``
        * ``"skip_g_subtree"``

    ``f_nodes`` and ``g_nodes`` contain original input node indices, not the
    internal timestamp-sorted indices.  For a match, both tuples have length 1.
    For a subtree skip, the skipped tuple contains every node in that subtree.
    """

    kind: str
    f_nodes: Tuple[int, ...]
    g_nodes: Tuple[int, ...]
    contribution: float


@dataclass(frozen=True)
class GraphETWResult:
    """Result returned by :func:`tree_etw_align` or :func:`dag_etw_align`."""

    score: float
    blocks: List[GraphETWBlock]
    pairs: List[Tuple[int, int]]
    unmatched_f: List[int]
    unmatched_g: List[int]
    score_table: Optional[np.ndarray] = None


@dataclass(frozen=True)
class _PreparedTree:
    n: int
    order: np.ndarray                 # sorted index -> original index
    inverse: np.ndarray               # original index -> sorted index
    times: np.ndarray                 # sorted timestamps
    children_ptr: np.ndarray
    children_idx: np.ndarray
    roots: np.ndarray                 # sorted node indices


@dataclass(frozen=True)
class _PreparedDAG:
    n: int
    order: np.ndarray                 # sorted index -> original index
    inverse: np.ndarray               # original index -> sorted index
    times: np.ndarray                 # sorted timestamps
    pred_ptr: np.ndarray              # augmented indices: 0 is virtual source
    pred_idx: np.ndarray
    sinks_aug: np.ndarray             # augmented sink indices in 1..n


# ---------------------------------------------------------------------------
# Public API: ordered tree/forest alignment
# ---------------------------------------------------------------------------


def tree_etw_align(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    f_edges: Sequence[Tuple[int, int]],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    g_edges: Sequence[Tuple[int, int]],
    *,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    node_weight_f: Optional[Sequence[float] | float] = None,
    node_weight_g: Optional[Sequence[float] | float] = None,
    skip_f_penalty: Optional[Sequence[float] | float] = None,
    skip_g_penalty: Optional[Sequence[float] | float] = None,
    roots_f: Optional[Sequence[int]] = None,
    roots_g: Optional[Sequence[int]] = None,
    use_numba: bool = True,
    check_nonnegative: bool = True,
    check_time_order: bool = True,
    return_score_table: bool = False,
) -> GraphETWResult:
    """Align two timestamped ordered trees or forests.

    Parameters
    ----------
    f_values, g_values:
        Node values.  They may be arbitrary Python objects when ``similarity``
        is supplied, because the similarity matrix is built before the numeric
        dynamic program runs.

    f_times, g_times:
        One timestamp per node.  Children are ordered by ``(timestamp, index)``.
        With ``check_time_order=True``, every edge must go from a smaller
        timestamp to a larger timestamp.

    f_edges, g_edges:
        Directed parent-to-child edges using original 0-based node indices.  A
        forest is allowed.  Every non-root node must have exactly one parent.

    similarity:
        Callable ``C(x, y) -> nonnegative float``.  Ignored if
        ``similarity_matrix`` is supplied.

    similarity_matrix:
        Optional matrix ``C[u, v]`` in original input-node order.

    node_weight_f, node_weight_g:
        Nonnegative scalar or per-node weights.  The match contribution is
        ``sqrt(weight_f[u] * weight_g[v]) * C(u, v)``.  Defaults to 1.0.

    skip_f_penalty, skip_g_penalty:
        Nonnegative scalar or per-node penalties.  In the tree/forest objective,
        skipping a child skips its whole subtree and costs the sum of its node
        penalties.  ``None`` means ``np.inf`` and disallows skipping.

    roots_f, roots_g:
        Optional explicit roots in original node indices.  If omitted, roots are
        inferred as the zero-indegree nodes.

    Returns
    -------
    GraphETWResult
        Contains the optimal forest score, matched node pairs, skipped nodes,
        traceback blocks, and optionally the node-pair score table.

    Notes
    -----
    This is an ordered-tree objective.  Timestamps provide the sibling order.
    For unordered trees one would replace each child-list DP by a bipartite
    matching subproblem.
    """

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both graphs must contain at least one node.")

    tree_f = _prepare_tree(
        n, f_times, f_edges, roots_f, "f", check_time_order=check_time_order
    )
    tree_g = _prepare_tree(
        m, g_times, g_edges, roots_g, "g", check_time_order=check_time_order
    )

    C = _similarity_matrix(f_values, g_values, similarity, similarity_matrix)
    if C.shape != (n, m):
        raise ValueError(f"similarity_matrix must have shape {(n, m)}, got {C.shape}.")
    if not np.all(np.isfinite(C)):
        raise ValueError("All similarities must be finite real numbers.")
    if check_nonnegative:
        min_c = float(np.min(C))
        if min_c < -1.0e-12:
            raise ValueError(
                "Similarities must be nonnegative for this Hellinger-style score. "
                f"Minimum observed value was {min_c}."
            )
        C = np.maximum(C, 0.0)

    # Reorder everything into timestamp/topological order.
    C_sorted = C[np.ix_(tree_f.order, tree_g.order)]
    wf = _weight_array(node_weight_f, n, "node_weight_f")[tree_f.order]
    wg = _weight_array(node_weight_g, m, "node_weight_g")[tree_g.order]
    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")[tree_f.order]
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")[tree_g.order]

    match_scores = C_sorted * np.sqrt(wf[:, None] * wg[None, :])
    subtree_skip_f = _subtree_skip_costs_python(
        tree_f.children_ptr, tree_f.children_idx, skip_f
    )
    subtree_skip_g = _subtree_skip_costs_python(
        tree_g.children_ptr, tree_g.children_idx, skip_g
    )

    if use_numba and _HAVE_NUMBA:
        S = _tree_score_table_numba(
            match_scores,
            tree_f.children_ptr,
            tree_f.children_idx,
            tree_g.children_ptr,
            tree_g.children_idx,
            subtree_skip_f,
            subtree_skip_g,
        )
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python tree DP.",
                RuntimeWarning,
                stacklevel=2,
            )
        S = _tree_score_table_python(
            match_scores,
            tree_f.children_ptr,
            tree_f.children_idx,
            tree_g.children_ptr,
            tree_g.children_idx,
            subtree_skip_f,
            subtree_skip_g,
        )

    root_score, _ = _node_list_alignment_dp(
        tree_f.roots, tree_g.roots, S, subtree_skip_f, subtree_skip_g, keep_ptr=False
    )
    if not np.isfinite(root_score):
        raise ValueError(
            "No finite tree/forest alignment was found. This usually means "
            "skipping is disallowed and the ordered child structures cannot be aligned."
        )

    blocks: List[GraphETWBlock] = []
    pairs: List[Tuple[int, int]] = []
    unmatched_f: List[int] = []
    unmatched_g: List[int] = []
    _trace_node_list_alignment(
        tree_f.roots,
        tree_g.roots,
        S,
        match_scores,
        subtree_skip_f,
        subtree_skip_g,
        tree_f,
        tree_g,
        blocks,
        pairs,
        unmatched_f,
        unmatched_g,
    )

    score_table: Optional[np.ndarray]
    if return_score_table:
        score_table = np.empty_like(S)
        # Reorder score table back to original input order.
        for si, oi in enumerate(tree_f.order):
            for sj, oj in enumerate(tree_g.order):
                score_table[oi, oj] = S[si, sj]
    else:
        score_table = None

    return GraphETWResult(
        score=float(root_score),
        blocks=blocks,
        pairs=pairs,
        unmatched_f=unmatched_f,
        unmatched_g=unmatched_g,
        score_table=score_table,
    )


# ---------------------------------------------------------------------------
# Public API: product-DAG trajectory alignment
# ---------------------------------------------------------------------------


def dag_etw_align(
    f_values: Sequence[Any],
    f_times: Sequence[float],
    f_edges: Sequence[Tuple[int, int]],
    g_values: Sequence[Any],
    g_times: Sequence[float],
    g_edges: Sequence[Tuple[int, int]],
    *,
    similarity: Optional[Callable[[Any, Any], float]] = None,
    similarity_matrix: Optional[np.ndarray] = None,
    node_weight_f: Optional[Sequence[float] | float] = None,
    node_weight_g: Optional[Sequence[float] | float] = None,
    skip_f_penalty: Optional[Sequence[float] | float] = None,
    skip_g_penalty: Optional[Sequence[float] | float] = None,
    end_mode: str = "sinks",
    use_numba: bool = True,
    check_nonnegative: bool = True,
    check_time_order: bool = True,
    return_score_table: bool = False,
) -> GraphETWResult:
    """Align one ordered trajectory through each timestamped DAG.

    The state space is the product DAG.  A traceback is a sequence of operations
    that advances in ``f`` only, in ``g`` only, or in both graphs:

    * ``match``: advance to nodes ``u`` and ``v`` and add
      ``sqrt(weight_f[u] * weight_g[v]) * C(u, v)``;
    * ``skip_f``: advance to node ``u`` only and subtract its skip penalty;
    * ``skip_g``: advance to node ``v`` only and subtract its skip penalty.

    ``end_mode='sinks'`` returns the best alignment ending at a sink in each
    DAG.  ``end_mode='any'`` returns the best state anywhere in the product DAG,
    which is a local/best-subtrajectory variant.

    This is cheap for bounded-indegree timestamped DAGs: roughly
    ``O(E_f * E_g + E_f * |V_g| + |V_f| * E_g)`` after similarities are known.
    It is not a general graph-edit-distance solver.
    """

    n = len(f_values)
    m = len(g_values)
    if n <= 0 or m <= 0:
        raise ValueError("Both graphs must contain at least one node.")
    if end_mode not in {"sinks", "any"}:
        raise ValueError("end_mode must be either 'sinks' or 'any'.")

    dag_f = _prepare_dag(n, f_times, f_edges, "f", check_time_order=check_time_order)
    dag_g = _prepare_dag(m, g_times, g_edges, "g", check_time_order=check_time_order)

    C = _similarity_matrix(f_values, g_values, similarity, similarity_matrix)
    if C.shape != (n, m):
        raise ValueError(f"similarity_matrix must have shape {(n, m)}, got {C.shape}.")
    if not np.all(np.isfinite(C)):
        raise ValueError("All similarities must be finite real numbers.")
    if check_nonnegative:
        min_c = float(np.min(C))
        if min_c < -1.0e-12:
            raise ValueError(
                "Similarities must be nonnegative for this Hellinger-style score. "
                f"Minimum observed value was {min_c}."
            )
        C = np.maximum(C, 0.0)

    C_sorted = C[np.ix_(dag_f.order, dag_g.order)]
    wf = _weight_array(node_weight_f, n, "node_weight_f")[dag_f.order]
    wg = _weight_array(node_weight_g, m, "node_weight_g")[dag_g.order]
    skip_f = _penalty_array(skip_f_penalty, n, "skip_f_penalty")[dag_f.order]
    skip_g = _penalty_array(skip_g_penalty, m, "skip_g_penalty")[dag_g.order]
    match_scores = C_sorted * np.sqrt(wf[:, None] * wg[None, :])

    if use_numba and _HAVE_NUMBA:
        V, ptr_op, ptr_i, ptr_j = _dag_dp_numba(
            match_scores,
            dag_f.pred_ptr,
            dag_f.pred_idx,
            dag_g.pred_ptr,
            dag_g.pred_idx,
            skip_f,
            skip_g,
        )
    else:
        if use_numba and not _HAVE_NUMBA:
            warnings.warn(
                "Numba is not installed; falling back to the pure-Python DAG DP.",
                RuntimeWarning,
                stacklevel=2,
            )
        V, ptr_op, ptr_i, ptr_j = _dag_dp_python(
            match_scores,
            dag_f.pred_ptr,
            dag_f.pred_idx,
            dag_g.pred_ptr,
            dag_g.pred_idx,
            skip_f,
            skip_g,
        )

    if end_mode == "sinks":
        best = NEG_INF
        best_i = -1
        best_j = -1
        for i in dag_f.sinks_aug:
            for j in dag_g.sinks_aug:
                val = float(V[int(i), int(j)])
                if val > best:
                    best = val
                    best_i = int(i)
                    best_j = int(j)
    else:
        best = NEG_INF
        best_i = -1
        best_j = -1
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                if i == 0 and j == 0:
                    continue
                val = float(V[i, j])
                if val > best:
                    best = val
                    best_i = i
                    best_j = j

    if best_i < 0 or not np.isfinite(best):
        raise ValueError(
            "No finite DAG trajectory alignment was found. This usually means "
            "skipping is disallowed and no compatible source-to-sink trajectories exist."
        )

    blocks, pairs, unmatched_f, unmatched_g = _trace_dag_alignment(
        V, ptr_op, ptr_i, ptr_j, best_i, best_j, dag_f, dag_g
    )

    score_table = V if return_score_table else None
    return GraphETWResult(
        score=float(best),
        blocks=blocks,
        pairs=pairs,
        unmatched_f=unmatched_f,
        unmatched_g=unmatched_g,
        score_table=score_table,
    )


# ---------------------------------------------------------------------------
# Similarity, weights, and penalties
# ---------------------------------------------------------------------------


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


def _weight_array(
    weights: Optional[Sequence[float] | float], n: int, name: str
) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim == 0:
        out = np.full(n, float(arr), dtype=np.float64)
    elif arr.ndim == 1 and arr.size == n:
        out = arr.astype(np.float64, copy=True)
    else:
        raise ValueError(f"{name} must be a scalar or a length-{n} array.")
    if np.any(~np.isfinite(out)) or np.any(out < 0.0):
        raise ValueError(f"{name} must contain finite nonnegative weights.")
    return out


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


# ---------------------------------------------------------------------------
# Graph preparation
# ---------------------------------------------------------------------------


def _sorted_order_from_times(times: Sequence[float], n: int, name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(times, dtype=np.float64)
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(f"{name}_times must be a one-dimensional length-{n} array.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}_times must contain only finite values.")
    order = np.lexsort((np.arange(n, dtype=np.int64), arr)).astype(np.int64)
    inverse = np.empty(n, dtype=np.int64)
    for si, oi in enumerate(order):
        inverse[int(oi)] = si
    return order, inverse, arr[order].astype(np.float64)


def _validate_edge(edge: Tuple[int, int], n: int, name: str) -> Tuple[int, int]:
    if len(edge) != 2:
        raise ValueError(f"Each {name}_edge must be a pair (parent, child).")
    p = int(edge[0])
    c = int(edge[1])
    if p < 0 or p >= n or c < 0 or c >= n:
        raise ValueError(f"{name}_edge ({p}, {c}) contains an out-of-range node.")
    if p == c:
        raise ValueError(f"{name}_edge ({p}, {c}) is a self-loop.")
    return p, c


def _prepare_tree(
    n: int,
    times: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    roots: Optional[Sequence[int]],
    name: str,
    *,
    check_time_order: bool,
) -> _PreparedTree:
    order, inverse, times_sorted = _sorted_order_from_times(times, n, name)
    times_orig = np.asarray(times, dtype=np.float64)

    children: List[List[int]] = [[] for _ in range(n)]
    indeg = np.zeros(n, dtype=np.int64)
    seen_edges: set[Tuple[int, int]] = set()

    for edge in edges:
        p_orig, c_orig = _validate_edge(edge, n, name)
        if (p_orig, c_orig) in seen_edges:
            raise ValueError(f"Duplicate {name}_edge ({p_orig}, {c_orig}).")
        seen_edges.add((p_orig, c_orig))
        if check_time_order and not (times_orig[p_orig] < times_orig[c_orig]):
            raise ValueError(
                f"{name}_edge ({p_orig}, {c_orig}) violates timestamp order: "
                f"{times_orig[p_orig]} is not < {times_orig[c_orig]}."
            )
        p = int(inverse[p_orig])
        c = int(inverse[c_orig])
        children[p].append(c)
        indeg[c] += 1
        if indeg[c] > 1:
            raise ValueError(
                f"{name} is not a tree/forest: node {c_orig} has more than one parent."
            )

    if roots is None:
        root_list = [i for i in range(n) if indeg[i] == 0]
    else:
        root_set = set()
        for r_orig in roots:
            r = int(r_orig)
            if r < 0 or r >= n:
                raise ValueError(f"{name}_root {r} is out of range.")
            sr = int(inverse[r])
            if indeg[sr] != 0:
                raise ValueError(f"{name}_root {r} has a parent and cannot be a root.")
            root_set.add(sr)
        root_list = sorted(root_set, key=lambda x: (times_sorted[x], int(order[x])))
        inferred = {i for i in range(n) if indeg[i] == 0}
        if set(root_list) != inferred:
            missing = [int(order[i]) for i in sorted(inferred - set(root_list))]
            if missing:
                raise ValueError(
                    f"{name}_roots omit zero-indegree nodes {missing}. "
                    "Pass all forest roots or omit roots to infer them."
                )

    # A forest with n nodes and r roots must have n-r edges when every non-root
    # has exactly one parent.  This also catches disconnected cycles when time
    # checking is disabled.
    if len(edges) != n - len(root_list):
        raise ValueError(
            f"{name} is not a valid forest: expected {n - len(root_list)} edges "
            f"for {n} nodes and {len(root_list)} roots, got {len(edges)}."
        )

    for u in range(n):
        children[u].sort(key=lambda x: (times_sorted[x], int(order[x])))
        # In timestamp order children should appear later.  This is guaranteed
        # if check_time_order is true, but keep a cheap internal check.
        for c in children[u]:
            if c <= u and check_time_order:
                raise ValueError(f"Internal ordering failure for {name}; edge is not forward.")

    children_ptr = np.zeros(n + 1, dtype=np.int64)
    flat: List[int] = []
    for u in range(n):
        children_ptr[u] = len(flat)
        flat.extend(children[u])
    children_ptr[n] = len(flat)
    children_idx = np.asarray(flat, dtype=np.int64)
    roots_arr = np.asarray(root_list, dtype=np.int64)
    return _PreparedTree(n, order, inverse, times_sorted, children_ptr, children_idx, roots_arr)


def _prepare_dag(
    n: int,
    times: Sequence[float],
    edges: Sequence[Tuple[int, int]],
    name: str,
    *,
    check_time_order: bool,
) -> _PreparedDAG:
    order, inverse, times_sorted = _sorted_order_from_times(times, n, name)
    times_orig = np.asarray(times, dtype=np.float64)

    preds: List[List[int]] = [[] for _ in range(n)]
    outdeg = np.zeros(n, dtype=np.int64)
    seen_edges: set[Tuple[int, int]] = set()

    for edge in edges:
        p_orig, c_orig = _validate_edge(edge, n, name)
        if (p_orig, c_orig) in seen_edges:
            raise ValueError(f"Duplicate {name}_edge ({p_orig}, {c_orig}).")
        seen_edges.add((p_orig, c_orig))
        if check_time_order and not (times_orig[p_orig] < times_orig[c_orig]):
            raise ValueError(
                f"{name}_edge ({p_orig}, {c_orig}) violates timestamp order: "
                f"{times_orig[p_orig]} is not < {times_orig[c_orig]}."
            )
        p = int(inverse[p_orig])
        c = int(inverse[c_orig])
        if check_time_order and p >= c:
            raise ValueError(f"Internal ordering failure for {name}; edge is not forward.")
        preds[c].append(p + 1)       # augmented real node index
        outdeg[p] += 1

    for u in range(n):
        preds[u].sort(key=lambda aug: (times_sorted[aug - 1], int(order[aug - 1])))

    pred_ptr = np.zeros(n + 2, dtype=np.int64)  # states 0..n inclusive
    flat: List[int] = []
    pred_ptr[0] = 0
    for u in range(n):
        pred_ptr[u + 1] = len(flat)
        if preds[u]:
            flat.extend(preds[u])
        else:
            flat.append(0)  # virtual source -> root
    pred_ptr[n + 1] = len(flat)
    pred_idx = np.asarray(flat, dtype=np.int64)
    sinks_aug = np.asarray([u + 1 for u in range(n) if outdeg[u] == 0], dtype=np.int64)
    if sinks_aug.size == 0:
        raise ValueError(f"{name} has no sinks; check that the input is acyclic.")
    return _PreparedDAG(n, order, inverse, times_sorted, pred_ptr, pred_idx, sinks_aug)


# ---------------------------------------------------------------------------
# Ordered tree score DP
# ---------------------------------------------------------------------------


def _subtree_skip_costs_python(
    children_ptr: np.ndarray, children_idx: np.ndarray, skip: np.ndarray
) -> np.ndarray:
    n = skip.size
    out = np.array(skip, dtype=np.float64, copy=True)
    for u in range(n - 1, -1, -1):
        total = float(skip[u])
        for kk in range(int(children_ptr[u]), int(children_ptr[u + 1])):
            total += float(out[int(children_idx[kk])])
        out[u] = total
    return out


def _tree_score_table_python(
    match_scores: np.ndarray,
    child_ptr_f: np.ndarray,
    child_idx_f: np.ndarray,
    child_ptr_g: np.ndarray,
    child_idx_g: np.ndarray,
    subtree_skip_f: np.ndarray,
    subtree_skip_g: np.ndarray,
) -> np.ndarray:
    n, m = match_scores.shape
    S = np.full((n, m), NEG_INF, dtype=np.float64)
    for u in range(n - 1, -1, -1):
        children_u = child_idx_f[int(child_ptr_f[u]) : int(child_ptr_f[u + 1])]
        for v in range(m - 1, -1, -1):
            children_v = child_idx_g[int(child_ptr_g[v]) : int(child_ptr_g[v + 1])]
            child_score, _ = _node_list_alignment_dp(
                children_u, children_v, S, subtree_skip_f, subtree_skip_g, keep_ptr=False
            )
            S[u, v] = float(match_scores[u, v]) + child_score
    return S


if _HAVE_NUMBA:

    @njit(cache=True)
    def _tree_score_table_numba(
        match_scores: np.ndarray,
        child_ptr_f: np.ndarray,
        child_idx_f: np.ndarray,
        child_ptr_g: np.ndarray,
        child_idx_g: np.ndarray,
        subtree_skip_f: np.ndarray,
        subtree_skip_g: np.ndarray,
    ) -> np.ndarray:
        n = match_scores.shape[0]
        m = match_scores.shape[1]
        S = np.empty((n, m), dtype=np.float64)
        for u0 in range(n):
            for v0 in range(m):
                S[u0, v0] = NEG_INF

        for u in range(n - 1, -1, -1):
            fu0 = child_ptr_f[u]
            fu1 = child_ptr_f[u + 1]
            du = fu1 - fu0
            for v in range(m - 1, -1, -1):
                gv0 = child_ptr_g[v]
                gv1 = child_ptr_g[v + 1]
                dv = gv1 - gv0

                prev = np.empty(dv + 1, dtype=np.float64)
                curr = np.empty(dv + 1, dtype=np.float64)
                prev[0] = 0.0
                for b in range(1, dv + 1):
                    cg = child_idx_g[gv0 + b - 1]
                    prev[b] = prev[b - 1] - subtree_skip_g[cg]

                for a in range(1, du + 1):
                    cf = child_idx_f[fu0 + a - 1]
                    curr[0] = prev[0] - subtree_skip_f[cf]
                    for b in range(1, dv + 1):
                        cg = child_idx_g[gv0 + b - 1]
                        best = prev[b - 1] + S[cf, cg]
                        cand = prev[b] - subtree_skip_f[cf]
                        if cand > best:
                            best = cand
                        cand = curr[b - 1] - subtree_skip_g[cg]
                        if cand > best:
                            best = cand
                        curr[b] = best
                    tmp = prev
                    prev = curr
                    curr = tmp

                S[u, v] = match_scores[u, v] + prev[dv]
        return S

else:

    def _tree_score_table_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("Numba is not available.")


def _node_list_alignment_dp(
    nodes_f: Sequence[int] | np.ndarray,
    nodes_g: Sequence[int] | np.ndarray,
    S: np.ndarray,
    subtree_skip_f: np.ndarray,
    subtree_skip_g: np.ndarray,
    *,
    keep_ptr: bool,
) -> Tuple[float, Optional[np.ndarray]]:
    """Needleman-Wunsch alignment of two ordered node lists using subtree scores."""

    a_nodes = [int(x) for x in nodes_f]
    b_nodes = [int(x) for x in nodes_g]
    a = len(a_nodes)
    b = len(b_nodes)
    D = np.full((a + 1, b + 1), NEG_INF, dtype=np.float64)
    ptr = np.zeros((a + 1, b + 1), dtype=np.int8) if keep_ptr else None
    D[0, 0] = 0.0
    for i in range(1, a + 1):
        cf = a_nodes[i - 1]
        D[i, 0] = D[i - 1, 0] - float(subtree_skip_f[cf])
        if ptr is not None:
            ptr[i, 0] = _OP_SKIP_F
    for j in range(1, b + 1):
        cg = b_nodes[j - 1]
        D[0, j] = D[0, j - 1] - float(subtree_skip_g[cg])
        if ptr is not None:
            ptr[0, j] = _OP_SKIP_G

    for i in range(1, a + 1):
        cf = a_nodes[i - 1]
        for j in range(1, b + 1):
            cg = b_nodes[j - 1]
            best = D[i - 1, j - 1] + float(S[cf, cg])
            op = _OP_MATCH
            cand = D[i - 1, j] - float(subtree_skip_f[cf])
            if cand > best:
                best = cand
                op = _OP_SKIP_F
            cand = D[i, j - 1] - float(subtree_skip_g[cg])
            if cand > best:
                best = cand
                op = _OP_SKIP_G
            D[i, j] = best
            if ptr is not None:
                ptr[i, j] = op
    return float(D[a, b]), ptr


def _trace_node_list_alignment(
    nodes_f: Sequence[int] | np.ndarray,
    nodes_g: Sequence[int] | np.ndarray,
    S: np.ndarray,
    match_scores: np.ndarray,
    subtree_skip_f: np.ndarray,
    subtree_skip_g: np.ndarray,
    tree_f: _PreparedTree,
    tree_g: _PreparedTree,
    blocks: List[GraphETWBlock],
    pairs: List[Tuple[int, int]],
    unmatched_f: List[int],
    unmatched_g: List[int],
) -> None:
    a_nodes = [int(x) for x in nodes_f]
    b_nodes = [int(x) for x in nodes_g]
    _, ptr = _node_list_alignment_dp(
        a_nodes, b_nodes, S, subtree_skip_f, subtree_skip_g, keep_ptr=True
    )
    if ptr is None:
        raise RuntimeError("Internal error: missing traceback table.")

    ops_rev: List[Tuple[int, Optional[int], Optional[int]]] = []
    i = len(a_nodes)
    j = len(b_nodes)
    while i > 0 or j > 0:
        op = int(ptr[i, j])
        if op == int(_OP_MATCH):
            ops_rev.append((op, a_nodes[i - 1], b_nodes[j - 1]))
            i -= 1
            j -= 1
        elif op == int(_OP_SKIP_F):
            ops_rev.append((op, a_nodes[i - 1], None))
            i -= 1
        elif op == int(_OP_SKIP_G):
            ops_rev.append((op, None, b_nodes[j - 1]))
            j -= 1
        else:
            raise RuntimeError(f"Traceback failed in node-list alignment at ({i}, {j}).")

    for op, cf, cg in reversed(ops_rev):
        if op == int(_OP_MATCH):
            assert cf is not None and cg is not None
            _trace_tree_pair(
                cf,
                cg,
                S,
                match_scores,
                subtree_skip_f,
                subtree_skip_g,
                tree_f,
                tree_g,
                blocks,
                pairs,
                unmatched_f,
                unmatched_g,
            )
        elif op == int(_OP_SKIP_F):
            assert cf is not None
            nodes = _subtree_original_nodes(cf, tree_f)
            blocks.append(
                GraphETWBlock(
                    kind="skip_f_subtree",
                    f_nodes=tuple(nodes),
                    g_nodes=(),
                    contribution=-float(subtree_skip_f[cf]),
                )
            )
            unmatched_f.extend(nodes)
        elif op == int(_OP_SKIP_G):
            assert cg is not None
            nodes = _subtree_original_nodes(cg, tree_g)
            blocks.append(
                GraphETWBlock(
                    kind="skip_g_subtree",
                    f_nodes=(),
                    g_nodes=tuple(nodes),
                    contribution=-float(subtree_skip_g[cg]),
                )
            )
            unmatched_g.extend(nodes)


def _trace_tree_pair(
    u: int,
    v: int,
    S: np.ndarray,
    match_scores: np.ndarray,
    subtree_skip_f: np.ndarray,
    subtree_skip_g: np.ndarray,
    tree_f: _PreparedTree,
    tree_g: _PreparedTree,
    blocks: List[GraphETWBlock],
    pairs: List[Tuple[int, int]],
    unmatched_f: List[int],
    unmatched_g: List[int],
) -> None:
    orig_u = int(tree_f.order[u])
    orig_v = int(tree_g.order[v])
    blocks.append(
        GraphETWBlock(
            kind="match",
            f_nodes=(orig_u,),
            g_nodes=(orig_v,),
            contribution=float(match_scores[u, v]),
        )
    )
    pairs.append((orig_u, orig_v))

    children_u = tree_f.children_idx[int(tree_f.children_ptr[u]) : int(tree_f.children_ptr[u + 1])]
    children_v = tree_g.children_idx[int(tree_g.children_ptr[v]) : int(tree_g.children_ptr[v + 1])]
    _trace_node_list_alignment(
        children_u,
        children_v,
        S,
        match_scores,
        subtree_skip_f,
        subtree_skip_g,
        tree_f,
        tree_g,
        blocks,
        pairs,
        unmatched_f,
        unmatched_g,
    )


def _subtree_original_nodes(u: int, tree: _PreparedTree) -> List[int]:
    out: List[int] = []
    stack = [int(u)]
    while stack:
        x = stack.pop()
        out.append(int(tree.order[x]))
        children = tree.children_idx[int(tree.children_ptr[x]) : int(tree.children_ptr[x + 1])]
        for c in reversed(children):
            stack.append(int(c))
    return out


# ---------------------------------------------------------------------------
# Product-DAG trajectory DP
# ---------------------------------------------------------------------------


def _dag_dp_python(
    match_scores: np.ndarray,
    pred_ptr_f: np.ndarray,
    pred_idx_f: np.ndarray,
    pred_ptr_g: np.ndarray,
    pred_idx_g: np.ndarray,
    skip_f: np.ndarray,
    skip_g: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n, m = match_scores.shape
    V = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
    ptr_i = np.full((n + 1, m + 1), -1, dtype=np.int64)
    ptr_j = np.full((n + 1, m + 1), -1, dtype=np.int64)
    V[0, 0] = 0.0

    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            best = NEG_INF
            best_op = _OP_NONE
            best_i = -1
            best_j = -1

            if i > 0 and j > 0:
                for kk_f in range(int(pred_ptr_f[i]), int(pred_ptr_f[i + 1])):
                    pi = int(pred_idx_f[kk_f])
                    for kk_g in range(int(pred_ptr_g[j]), int(pred_ptr_g[j + 1])):
                        pj = int(pred_idx_g[kk_g])
                        cand = float(V[pi, pj]) + float(match_scores[i - 1, j - 1])
                        if cand > best:
                            best = cand
                            best_op = _OP_MATCH
                            best_i = pi
                            best_j = pj

            if i > 0:
                for kk_f in range(int(pred_ptr_f[i]), int(pred_ptr_f[i + 1])):
                    pi = int(pred_idx_f[kk_f])
                    cand = float(V[pi, j]) - float(skip_f[i - 1])
                    if cand > best:
                        best = cand
                        best_op = _OP_SKIP_F
                        best_i = pi
                        best_j = j

            if j > 0:
                for kk_g in range(int(pred_ptr_g[j]), int(pred_ptr_g[j + 1])):
                    pj = int(pred_idx_g[kk_g])
                    cand = float(V[i, pj]) - float(skip_g[j - 1])
                    if cand > best:
                        best = cand
                        best_op = _OP_SKIP_G
                        best_i = i
                        best_j = pj

            V[i, j] = best
            ptr_op[i, j] = best_op
            ptr_i[i, j] = best_i
            ptr_j[i, j] = best_j
    return V, ptr_op, ptr_i, ptr_j


if _HAVE_NUMBA:

    @njit(cache=True)
    def _dag_dp_numba(
        match_scores: np.ndarray,
        pred_ptr_f: np.ndarray,
        pred_idx_f: np.ndarray,
        pred_ptr_g: np.ndarray,
        pred_idx_g: np.ndarray,
        skip_f: np.ndarray,
        skip_g: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = match_scores.shape[0]
        m = match_scores.shape[1]
        V = np.empty((n + 1, m + 1), dtype=np.float64)
        ptr_op = np.zeros((n + 1, m + 1), dtype=np.int8)
        ptr_i = np.empty((n + 1, m + 1), dtype=np.int64)
        ptr_j = np.empty((n + 1, m + 1), dtype=np.int64)
        for i0 in range(n + 1):
            for j0 in range(m + 1):
                V[i0, j0] = NEG_INF
                ptr_i[i0, j0] = -1
                ptr_j[i0, j0] = -1
        V[0, 0] = 0.0

        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 and j == 0:
                    continue
                best = NEG_INF
                best_op = _OP_NONE
                best_i = -1
                best_j = -1

                if i > 0 and j > 0:
                    for kk_f in range(pred_ptr_f[i], pred_ptr_f[i + 1]):
                        pi = pred_idx_f[kk_f]
                        for kk_g in range(pred_ptr_g[j], pred_ptr_g[j + 1]):
                            pj = pred_idx_g[kk_g]
                            cand = V[pi, pj] + match_scores[i - 1, j - 1]
                            if cand > best:
                                best = cand
                                best_op = _OP_MATCH
                                best_i = pi
                                best_j = pj

                if i > 0:
                    for kk_f in range(pred_ptr_f[i], pred_ptr_f[i + 1]):
                        pi = pred_idx_f[kk_f]
                        cand = V[pi, j] - skip_f[i - 1]
                        if cand > best:
                            best = cand
                            best_op = _OP_SKIP_F
                            best_i = pi
                            best_j = j

                if j > 0:
                    for kk_g in range(pred_ptr_g[j], pred_ptr_g[j + 1]):
                        pj = pred_idx_g[kk_g]
                        cand = V[i, pj] - skip_g[j - 1]
                        if cand > best:
                            best = cand
                            best_op = _OP_SKIP_G
                            best_i = i
                            best_j = pj

                V[i, j] = best
                ptr_op[i, j] = best_op
                ptr_i[i, j] = best_i
                ptr_j[i, j] = best_j
        return V, ptr_op, ptr_i, ptr_j

else:

    def _dag_dp_numba(*args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise RuntimeError("Numba is not available.")


def _trace_dag_alignment(
    V: np.ndarray,
    ptr_op: np.ndarray,
    ptr_i: np.ndarray,
    ptr_j: np.ndarray,
    end_i: int,
    end_j: int,
    dag_f: _PreparedDAG,
    dag_g: _PreparedDAG,
) -> Tuple[List[GraphETWBlock], List[Tuple[int, int]], List[int], List[int]]:
    i = int(end_i)
    j = int(end_j)
    blocks_rev: List[GraphETWBlock] = []
    pairs_rev: List[Tuple[int, int]] = []
    unmatched_f_rev: List[int] = []
    unmatched_g_rev: List[int] = []

    while i > 0 or j > 0:
        op = int(ptr_op[i, j])
        pi = int(ptr_i[i, j])
        pj = int(ptr_j[i, j])
        if op == int(_OP_NONE) or pi < 0 or pj < 0:
            raise RuntimeError(f"DAG traceback failed at state ({i}, {j}).")
        contribution = float(V[i, j] - V[pi, pj])

        if op == int(_OP_MATCH):
            of = int(dag_f.order[i - 1])
            og = int(dag_g.order[j - 1])
            blocks_rev.append(
                GraphETWBlock(
                    kind="match",
                    f_nodes=(of,),
                    g_nodes=(og,),
                    contribution=contribution,
                )
            )
            pairs_rev.append((of, og))
        elif op == int(_OP_SKIP_F):
            of = int(dag_f.order[i - 1])
            blocks_rev.append(
                GraphETWBlock(
                    kind="skip_f",
                    f_nodes=(of,),
                    g_nodes=(),
                    contribution=contribution,
                )
            )
            unmatched_f_rev.append(of)
        elif op == int(_OP_SKIP_G):
            og = int(dag_g.order[j - 1])
            blocks_rev.append(
                GraphETWBlock(
                    kind="skip_g",
                    f_nodes=(),
                    g_nodes=(og,),
                    contribution=contribution,
                )
            )
            unmatched_g_rev.append(og)
        else:
            raise RuntimeError(f"Unknown DAG traceback op {op} at state ({i}, {j}).")

        i, j = pi, pj

    return (
        list(reversed(blocks_rev)),
        list(reversed(pairs_rev)),
        list(reversed(unmatched_f_rev)),
        list(reversed(unmatched_g_rev)),
    )


__all__ = [
    "GraphETWBlock",
    "GraphETWResult",
    "compute_similarity_matrix",
    "tree_etw_align",
    "dag_etw_align",
]
