from __future__ import annotations

import numpy as np

from examples.sampling_rate_comparison import (
    density_quantile_breaks,
    run_swapped_density_comparison,
    true_warp,
)


def test_density_breaks_and_true_warp_are_monotone() -> None:
    for beta in (-2.0, 0.0, 2.0):
        breaks = density_quantile_breaks(80, beta)
        assert breaks.shape == (81,)
        assert breaks[0] == 0.0
        assert breaks[-1] == 1.0
        assert np.all(np.diff(breaks) > 0.0)
    grid = np.linspace(0.0, 1.0, 1001)
    warp = true_warp(grid)
    assert warp[0] == 0.0
    assert warp[-1] == 1.0
    assert np.all(np.diff(warp) > 0.0)


def test_selected_sampling_stress_example_is_reproducible() -> None:
    comparison = run_swapped_density_comparison(
        n=80, beta=2.0, sigma=0.25, use_numba=True
    )
    assert comparison.hellinger_between_rmse < 0.06
    assert comparison.dtw_between_rmse > 0.30
    assert max(x.hellinger_rmse for x in comparison.experiments) < 0.06
    assert max(x.dtw_rmse for x in comparison.experiments) > 0.30


def test_equal_sampling_is_easy_for_both_methods() -> None:
    comparison = run_swapped_density_comparison(
        n=80, beta=0.0, sigma=0.25, use_numba=False
    )
    assert max(x.hellinger_rmse for x in comparison.experiments) < 0.02
    assert max(x.dtw_rmse for x in comparison.experiments) < 0.02
