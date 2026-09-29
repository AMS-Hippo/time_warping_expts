from __future__ import annotations

import math

import numpy as np
import pytest

from hellinger_etw import etw_align as etw_align_dense
from hellinger_etw_multiscale import (
    build_dyadic_hierarchy,
    etw_align_multiscale,
    lift_pairs_to_row_band,
)
from simulate_etw_data import rbf_similarity_matrix, simulate_pair


def _scalar_rbf(sigma: float):
    def similarity(x, y) -> float:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(y, dtype=float)
        return float(np.exp(-0.5 * np.sum((xa - ya) ** 2) / (sigma * sigma)))

    return similarity


def _row_rbf(sigma: float):
    def row_similarity(x, ys) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        ya = np.asarray(ys, dtype=float)
        diff = ya - xa
        return np.exp(-0.5 * np.sum(diff * diff, axis=-1) / (sigma * sigma))

    return row_similarity


def test_dyadic_hierarchy_preserves_durations_and_weighted_means() -> None:
    f_values = np.array([0.0, 2.0, 10.0, 14.0, 20.0])
    f_breaks = np.array([0.0, 0.1, 0.4, 0.5, 0.9, 1.0])
    g_values = np.array([1.0, 3.0, 5.0])
    g_breaks = np.array([0.0, 0.2, 0.7, 1.0])

    levels = build_dyadic_hierarchy(
        f_values,
        f_breaks,
        g_values,
        g_breaks,
        coarsest_size=2,
    )
    assert [level.shape for level in levels] == [(5, 3), (3, 2), (2, 2)]

    level1 = levels[1]
    assert np.allclose(level1.f.breaks, [0.0, 0.4, 0.9, 1.0])
    # First pair: weights 0.1 and 0.3.
    assert float(level1.f.values[0]) == pytest.approx(1.5)
    # Second pair: weights 0.1 and 0.4.
    assert float(level1.f.values[1]) == pytest.approx(13.2)
    assert np.array_equal(level1.f.child_start, [0, 2, 4])
    assert np.array_equal(level1.f.child_stop, [2, 4, 5])

    # g is coarsened once and then retained through an identity map.
    level2 = levels[2]
    assert np.array_equal(level2.g.child_start, [0, 1])
    assert np.array_equal(level2.g.child_stop, [1, 2])


def test_non_numeric_values_require_reducer_when_coarsening() -> None:
    f_values = ["a", "b", "c", "d"]
    g_values = ["a", "b", "c", "d"]
    breaks = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError, match="coarsen_values"):
        build_dyadic_hierarchy(
            f_values,
            breaks,
            g_values,
            breaks,
            coarsest_size=2,
        )

    levels = build_dyadic_hierarchy(
        f_values,
        breaks,
        g_values,
        breaks,
        coarsest_size=2,
        coarsen_values=lambda vals, weights: vals[0],
    )
    assert levels[-1].shape == (2, 2)


def test_lifted_band_contains_child_rectangles_and_endpoints() -> None:
    # Coarse path: (0,0), (1,0), (1,1), (2,1).
    coarse_pairs = [(0, 0), (1, 0), (1, 1), (2, 1)]
    f_start = np.array([0, 2, 4])
    f_stop = np.array([2, 4, 5])
    g_start = np.array([0, 2])
    g_stop = np.array([2, 4])
    band = lift_pairs_to_row_band(
        5,
        4,
        coarse_pairs,
        f_start,
        f_stop,
        g_start,
        g_stop,
        radius=0,
    )
    assert band.contains(0, 0)
    assert band.contains(4, 3)
    for I, J in coarse_pairs:
        for i in range(f_start[I], f_stop[I]):
            for j in range(g_start[J], g_stop[J]):
                assert band.contains(i, j)
    assert np.all(band.widths > 0)


def test_no_coarsening_is_exact_dense_solve() -> None:
    rng = np.random.default_rng(22)
    n, m = 7, 6
    f = rng.normal(size=(n, 2))
    g = rng.normal(size=(m, 2))
    tf = np.linspace(0.0, 1.0, n + 1)
    tg = np.linspace(0.0, 1.0, m + 1)
    sigma = 0.8
    C = rbf_similarity_matrix(f, g, sigma=sigma)
    exact = etw_align_dense(f, tf, g, tg, similarity_matrix=C, use_numba=False)
    result = etw_align_multiscale(
        f,
        tf,
        g,
        tg,
        row_similarity=_row_rbf(sigma),
        coarsest_size=max(n, m),
        use_numba=False,
    )
    assert result.score == pytest.approx(exact.score, abs=3e-12)
    assert result.exact
    assert len(result.diagnostics.levels) == 1


@pytest.mark.parametrize("seed", range(8))
def test_full_radius_multiscale_matches_dense_exact(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(7, 15))
    m = int(rng.integers(7, 15))
    f = rng.normal(size=(n, 2))
    g = rng.normal(size=(m, 2))
    tf = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 1.0, n))))
    tg = np.concatenate(([0.0], np.cumsum(rng.uniform(0.1, 1.0, m))))
    tf /= tf[-1]
    tg /= tg[-1]
    sigma = 1.1
    exact = etw_align_dense(
        f,
        tf,
        g,
        tg,
        similarity_matrix=rbf_similarity_matrix(f, g, sigma=sigma),
        use_numba=False,
    )
    result = etw_align_multiscale(
        f,
        tf,
        g,
        tg,
        row_similarity=_row_rbf(sigma),
        coarsest_size=3,
        initial_radius=max(n, m),
        adaptive=False,
        use_numba=False,
    )
    assert result.final_band.allowed_cells == n * m
    assert result.score == pytest.approx(exact.score, abs=5e-12)
    assert result.exact


def test_fixed_narrow_band_score_is_bounded_by_exact() -> None:
    pair = simulate_pair(48, 53, seed=123, noise=0.02, warp_strength=0.5)
    sigma = 0.25
    exact = etw_align_dense(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity_matrix=rbf_similarity_matrix(pair.f_values, pair.g_values, sigma=sigma),
        use_numba=True,
    )
    approx = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=_row_rbf(sigma),
        coarsest_size=8,
        initial_radius=1,
        adaptive=False,
        use_numba=True,
    )
    assert approx.score <= exact.score + 1e-10
    assert approx.diagnostics.total_similarity_evaluations < 48 * 53 * 2


def test_adaptive_attempt_scores_are_monotone_and_diagnostics_consistent() -> None:
    pair = simulate_pair(72, 67, seed=910, noise=0.04, warp_strength=0.8)
    sigma = 0.22
    result = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=_row_rbf(sigma),
        coarsest_size=9,
        initial_radius=1,
        adaptive=True,
        min_successful_attempts=2,
        max_attempts=5,
        use_numba=True,
    )
    for level in result.diagnostics.levels:
        scores = [a.score for a in level.attempts if a.score is not None]
        assert all(b + 1e-10 >= a for a, b in zip(scores, scores[1:]))
        assert level.accepted_attempt == len(level.attempts) - 1
        for attempt in level.attempts:
            assert attempt.similarity_evaluations == attempt.allowed_cells
    assert result.diagnostics.total_similarity_evaluations == sum(
        a.similarity_evaluations
        for level in result.diagnostics.levels
        for a in level.attempts
    )


def test_scalar_similarity_and_row_similarity_agree() -> None:
    pair = simulate_pair(31, 27, seed=77, noise=0.01, warp_strength=0.3)
    sigma = 0.3
    kwargs = dict(
        coarsest_size=6,
        initial_radius=2,
        adaptive=False,
        use_numba=False,
    )
    scalar = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity=_scalar_rbf(sigma),
        **kwargs,
    )
    row = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=_row_rbf(sigma),
        **kwargs,
    )
    assert scalar.score == pytest.approx(row.score, abs=5e-12)
    assert scalar.pairs == row.pairs


def test_adaptive_can_recover_full_grid_exactness() -> None:
    pair = simulate_pair(24, 21, seed=333, noise=0.08, warp_strength=1.2)
    sigma = 0.25
    exact = etw_align_dense(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity_matrix=rbf_similarity_matrix(pair.f_values, pair.g_values, sigma=sigma),
        use_numba=False,
    )
    result = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=_row_rbf(sigma),
        coarsest_size=4,
        initial_radius=0,
        adaptive=True,
        # Force widening to full grid by making stability impossible before it.
        min_successful_attempts=100,
        use_numba=False,
    )
    assert result.exact
    assert result.score == pytest.approx(exact.score, abs=5e-12)


def test_start_time_interface_with_explicit_endpoints() -> None:
    f = np.array([0.0, 0.2, 0.4, 0.9])
    g = np.array([0.0, 0.3, 0.7])
    f_starts = np.array([0.0, 0.2, 0.55, 0.8])
    g_starts = np.array([0.0, 0.4, 0.75])
    result = etw_align_multiscale(
        f,
        f_starts,
        g,
        g_starts,
        end_f=1.2,
        end_g=1.1,
        similarity=lambda x, y: math.exp(-abs(float(x) - float(y))),
        coarsest_size=2,
        initial_radius=2,
        adaptive=False,
        use_numba=False,
    )
    assert math.isfinite(result.score)
    assert result.pairs[0] == (0, 0)
    assert result.pairs[-1] == (len(f) - 1, len(g) - 1)


def test_similarity_callback_errors_are_not_misclassified_as_band_infeasibility() -> None:
    f = np.arange(8.0)
    g = np.arange(7.0)
    tf = np.linspace(0.0, 1.0, 9)
    tg = np.linspace(0.0, 1.0, 8)

    def bad_row_similarity(x, ys):
        return np.array([1.0])

    with pytest.raises(ValueError, match="row_similarity returned shape"):
        etw_align_multiscale(
            f,
            tf,
            g,
            tg,
            row_similarity=bad_row_similarity,
            coarsest_size=2,
            initial_radius=1,
            adaptive=True,
            max_attempts=3,
            use_numba=False,
        )


def test_fixed_radius_is_not_reported_as_adaptively_converged() -> None:
    f = np.linspace(0.0, 1.0, 16)
    g = np.linspace(0.0, 1.0, 16)
    times = np.linspace(0.0, 1.0, 17)
    result = etw_align_multiscale(
        f,
        times,
        g,
        times,
        similarity=lambda x, y: math.exp(-abs(float(x) - float(y))),
        coarsest_size=4,
        initial_radius=1,
        adaptive=False,
        use_numba=False,
    )
    assert not result.diagnostics.heuristic_converged
    assert any(level.stopping_reason == "fixed_radius" for level in result.diagnostics.levels)


def test_propagated_radius_avoids_intermediate_scale_plateau_regression() -> None:
    """A wider intermediate solve can matter even when two smaller scores tie."""

    n = 256
    pair = simulate_pair(
        n,
        n,
        seed=105602,
        noise=0.08,
        warp_strength=1.2,
        grid_size=2048,
    )
    sigma = 0.25
    exact = etw_align_dense(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        similarity_matrix=rbf_similarity_matrix(pair.f_values, pair.g_values, sigma=sigma),
        use_numba=True,
    )
    result = etw_align_multiscale(
        pair.f_values,
        pair.f_times,
        pair.g_values,
        pair.g_times,
        row_similarity=_row_rbf(sigma),
        coarsest_size=32,
        initial_radius=2,
        adaptive=True,
        propagate_radius=True,
        stability_rtol=1e-4,
        stability_atol=1e-4,
        max_attempts=8,
        use_numba=True,
    )
    assert result.score == pytest.approx(exact.score, abs=5e-12)


def test_minimum_one_successful_attempt_can_stop_without_a_comparison() -> None:
    f = np.linspace(0.0, 1.0, 32)
    times = np.linspace(0.0, 1.0, 33)
    result = etw_align_multiscale(
        f,
        times,
        f,
        times,
        similarity=lambda x, y: math.exp(-abs(float(x) - float(y))),
        coarsest_size=8,
        initial_radius=4,
        adaptive=True,
        min_successful_attempts=1,
        use_numba=False,
    )
    for level in result.diagnostics.levels:
        if not level.coarsest:
            assert len(level.attempts) == 1
            assert level.stopping_reason == "score_stable"
