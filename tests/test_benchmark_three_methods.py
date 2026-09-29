from __future__ import annotations

import numpy as np
import pytest

from examples.benchmark_three_methods import (
    BenchmarkConfig,
    DEFAULT_REGIMES,
    estimate_dense_peak_gb,
    make_instance,
    power_of_two_sizes,
    validate_config,
)


def test_power_of_two_sizes() -> None:
    assert power_of_two_sizes(32, 256) == (32, 64, 128, 256)
    with pytest.raises(ValueError):
        power_of_two_sizes(48, 256)
    with pytest.raises(ValueError):
        power_of_two_sizes(256, 32)


def test_benchmark_instances_are_deterministic_and_well_formed() -> None:
    for regime in DEFAULT_REGIMES:
        first = make_instance(64, regime, seed=356)
        second = make_instance(64, regime, seed=356)
        assert first.f_values.shape == (64, 2)
        assert first.g_values.shape == (64, 2)
        assert first.f_times.shape == (65,)
        assert first.g_times.shape == (65,)
        assert np.all(np.diff(first.f_times) > 0)
        assert np.all(np.diff(first.g_times) > 0)
        np.testing.assert_allclose(first.f_values, second.f_values)
        np.testing.assert_allclose(first.g_values, second.g_values)
        np.testing.assert_allclose(first.f_times, second.f_times)
        np.testing.assert_allclose(first.g_times, second.g_times)


def test_benchmark_regimes_are_distinct() -> None:
    easy = make_instance(128, DEFAULT_REGIMES[0], seed=356)
    hard = make_instance(128, DEFAULT_REGIMES[1], seed=356)
    assert not np.allclose(easy.f_values, hard.f_values)
    assert not np.allclose(easy.g_values, hard.g_values)


def test_config_validation_and_memory_estimate() -> None:
    config = BenchmarkConfig(
        n_max=256,
        n_min=32,
        repeats=1,
        cubic_max_n=64,
        quadratic_max_n=256,
        coarsest_size=16,
    )
    validate_config(config)
    assert estimate_dense_peak_gb(2048) > estimate_dense_peak_gb(1024) > 0

    with pytest.raises(ValueError):
        validate_config(
            BenchmarkConfig(
                n_max=300,
                n_min=32,
                cubic_max_n=64,
                quadratic_max_n=256,
                coarsest_size=16,
            )
        )
