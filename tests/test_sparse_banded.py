from __future__ import annotations

import math

import numpy as np
import pytest

from hellinger_etw_sparse import RowBand, etw_align_sparse_banded, make_path_row_band
from tests.reference_banded import brute_force_banded_score, random_monotone_path


def _breaks_from_lengths(lengths: np.ndarray) -> np.ndarray:
    x = np.asarray(lengths, dtype=float)
    x = x / x.sum()
    return np.concatenate(([0.0], np.cumsum(x)))


def _validate_block_partition(result, n: int, m: int, mask: np.ndarray) -> None:
    i = 0
    j = 0
    for block in result.blocks:
        assert block.f_start == i
        assert block.g_start == j
        if block.kind == "many_f_to_one_g":
            assert block.f_stop > block.f_start
            assert block.g_stop == block.g_start + 1
        elif block.kind == "one_f_to_many_g":
            assert block.f_stop == block.f_start + 1
            assert block.g_stop > block.g_start
        else:  # no-skip module
            raise AssertionError(block.kind)
        for r in range(block.f_start, block.f_stop):
            for q in range(block.g_start, block.g_stop):
                assert mask[r, q]
        i = block.f_stop
        j = block.g_stop
    assert (i, j) == (n, m)
    assert sum(block.contribution for block in result.blocks) == pytest.approx(
        result.score, abs=3e-12
    )


@pytest.mark.parametrize("seed", range(30))
def test_sparse_solver_matches_independent_contiguous_band(seed: int) -> None:
    rng = np.random.default_rng(3000 + seed)
    n = int(rng.integers(1, 10))
    m = int(rng.integers(1, 10))
    path = random_monotone_path(n, m, rng)
    radius = int(rng.integers(0, 3))
    band = make_path_row_band(n, m, path, radius=radius)
    # Path coverage guarantees nonempty rows and endpoints.
    mask = band.to_mask()
    ds = rng.uniform(0.1, 1.0, size=n)
    dt = rng.uniform(0.1, 1.0, size=m)
    ds /= ds.sum()
    dt /= dt.sum()
    C = rng.random((n, m))
    reference = brute_force_banded_score(C, ds, dt, mask)

    result = etw_align_sparse_banded(
        list(range(n)),
        _breaks_from_lengths(ds),
        list(range(m)),
        _breaks_from_lengths(dt),
        band=band,
        similarity_matrix=C,
        use_numba=False,
        return_score_table=True,
    )
    assert result.score == pytest.approx(reference.score, abs=3e-12)
    assert np.allclose(
        result.score_table[np.isfinite(reference.score_table)],
        reference.score_table[np.isfinite(reference.score_table)],
        atol=3e-12,
        rtol=0.0,
    )
    _validate_block_partition(result, n, m, mask)


def test_sparse_numba_matches_python_and_reference() -> None:
    rng = np.random.default_rng(401)
    n, m = 17, 13
    path = random_monotone_path(n, m, rng)
    band = make_path_row_band(n, m, path, radius=2)
    ds = rng.uniform(0.1, 1.0, size=n)
    dt = rng.uniform(0.1, 1.0, size=m)
    ds /= ds.sum()
    dt /= dt.sum()
    C = rng.random((n, m))
    reference = brute_force_banded_score(C, ds, dt, band.to_mask())

    py = etw_align_sparse_banded(
        list(range(n)),
        _breaks_from_lengths(ds),
        list(range(m)),
        _breaks_from_lengths(dt),
        band=band,
        similarity_matrix=C,
        use_numba=False,
    )
    nb = etw_align_sparse_banded(
        list(range(n)),
        _breaks_from_lengths(ds),
        list(range(m)),
        _breaks_from_lengths(dt),
        band=band,
        similarity_matrix=C,
        use_numba=True,
    )
    assert py.score == pytest.approx(reference.score, abs=3e-12)
    assert nb.score == pytest.approx(reference.score, abs=3e-12)


def test_scalar_similarity_is_called_once_per_allowed_cell() -> None:
    n, m = 12, 15
    path = [(i, min(m - 1, round(i * (m - 1) / (n - 1)))) for i in range(n)]
    band = make_path_row_band(n, m, path, radius=2)
    calls = 0

    def similarity(x: int, y: int) -> float:
        nonlocal calls
        calls += 1
        return math.exp(-abs(x / n - y / m))

    result = etw_align_sparse_banded(
        list(range(n)),
        np.linspace(0.0, 1.0, n + 1),
        list(range(m)),
        np.linspace(0.0, 1.0, m + 1),
        band=band,
        similarity=similarity,
        use_numba=False,
    )
    assert calls == band.allowed_cells
    assert result.diagnostics.similarity_evaluations == band.allowed_cells
    assert result.diagnostics.state_cells == band.allowed_cells
    assert result.diagnostics.envelope_slots <= band.allowed_cells + m


def test_row_similarity_batches_by_nonempty_row() -> None:
    n, m = 10, 11
    band = RowBand.full(n, m)
    calls = 0

    def row_similarity(x: int, ys) -> np.ndarray:
        nonlocal calls
        calls += 1
        return np.exp(-np.abs(float(x) - np.asarray(ys, dtype=float)))

    result = etw_align_sparse_banded(
        list(range(n)),
        np.linspace(0.0, 1.0, n + 1),
        list(range(m)),
        np.linspace(0.0, 1.0, m + 1),
        band=band,
        row_similarity=row_similarity,
        use_numba=False,
    )
    assert calls == n
    assert result.diagnostics.row_similarity_calls == n
    assert result.diagnostics.similarity_evaluations == n * m


def test_row_band_rejects_gapped_mask() -> None:
    mask = np.array([[True, False, True], [True, True, True]], dtype=bool)
    with pytest.raises(ValueError, match="not contiguous"):
        RowBand.from_mask(mask)


def test_infeasible_empty_row_is_rejected() -> None:
    band = RowBand(np.array([0, 0, 1]), np.array([1, 0, 3]), 3)
    with pytest.raises(ValueError, match="empty"):
        etw_align_sparse_banded(
            [0, 1, 2],
            np.linspace(0.0, 1.0, 4),
            [0, 1, 2],
            np.linspace(0.0, 1.0, 4),
            band=band,
            similarity=lambda x, y: 1.0,
            use_numba=False,
        )

@pytest.mark.parametrize("value", [0.0, 1.0])
def test_sparse_solver_handles_flat_similarities_and_repeated_query_points(value: float) -> None:
    n, m = 7, 6
    band = RowBand.full(n, m)
    C = np.full((n, m), value, dtype=float)
    ds = np.array([1, 2, 1, 3, 2, 4, 1], dtype=float)
    dt = np.array([2, 1, 3, 1, 4, 2], dtype=float)
    ds /= ds.sum()
    dt /= dt.sum()
    reference = brute_force_banded_score(C, ds, dt, band.to_mask())
    for use_numba in (False, True):
        result = etw_align_sparse_banded(
            list(range(n)),
            _breaks_from_lengths(ds),
            list(range(m)),
            _breaks_from_lengths(dt),
            band=band,
            similarity_matrix=C,
            use_numba=use_numba,
        )
        assert result.score == pytest.approx(reference.score, abs=3e-12)
