from __future__ import annotations

import numpy as np
import pytest

import hellinger_etw
import hellinger_etw_banded
import hellinger_etw_original


def _breaks(n: int, rng: np.random.Generator) -> np.ndarray:
    lengths = rng.uniform(0.1, 1.0, size=n)
    lengths /= lengths.sum()
    return np.concatenate(([0.0], np.cumsum(lengths)))


@pytest.mark.parametrize("seed", range(12))
def test_cubic_quadratic_and_full_band_scores_agree(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 8))
    m = int(rng.integers(1, 8))
    C = rng.uniform(0.0, 1.0, size=(n, m))
    f_times = _breaks(n, rng)
    g_times = _breaks(m, rng)
    f = list(range(n))
    g = list(range(m))

    cubic = hellinger_etw_original.etw_align(
        f, f_times, g, g_times, similarity_matrix=C, use_numba=False
    )
    quadratic = hellinger_etw.etw_align(
        f, f_times, g, g_times, similarity_matrix=C, use_numba=False
    )
    banded = hellinger_etw_banded.etw_align_banded(
        f,
        f_times,
        g,
        g_times,
        similarity_matrix=C,
        band_mask=np.ones((n, m), dtype=bool),
        use_numba=False,
    )

    assert quadratic.score == pytest.approx(cubic.score, abs=2e-12)
    assert banded.score == pytest.approx(cubic.score, abs=2e-12)


@pytest.mark.parametrize("seed", range(8))
def test_finite_skip_full_band_agrees(seed: int) -> None:
    rng = np.random.default_rng(1000 + seed)
    n = int(rng.integers(1, 7))
    m = int(rng.integers(1, 7))
    # Signed similarities are disallowed, but small nonnegative values plus
    # small skip penalties still exercise the edit transitions.
    C = rng.uniform(0.0, 0.3, size=(n, m))
    f_times = _breaks(n, rng)
    g_times = _breaks(m, rng)
    skip_f = rng.uniform(0.0, 0.08, size=n)
    skip_g = rng.uniform(0.0, 0.08, size=m)
    f = list(range(n))
    g = list(range(m))

    cubic = hellinger_etw_original.etw_align(
        f,
        f_times,
        g,
        g_times,
        similarity_matrix=C,
        skip_f_penalty=skip_f,
        skip_g_penalty=skip_g,
        use_numba=False,
    )
    quadratic = hellinger_etw.etw_align(
        f,
        f_times,
        g,
        g_times,
        similarity_matrix=C,
        skip_f_penalty=skip_f,
        skip_g_penalty=skip_g,
        use_numba=False,
    )
    banded = hellinger_etw_banded.etw_align_banded(
        f,
        f_times,
        g,
        g_times,
        similarity_matrix=C,
        band_mask=np.ones((n, m), dtype=bool),
        skip_f_penalty=skip_f,
        skip_g_penalty=skip_g,
        use_numba=False,
    )

    assert quadratic.score == pytest.approx(cubic.score, abs=2e-12)
    assert banded.score == pytest.approx(cubic.score, abs=2e-12)


def test_numba_and_python_dense_quadratic_agree() -> None:
    rng = np.random.default_rng(77)
    n, m = 9, 8
    C = rng.random((n, m))
    f_times = _breaks(n, rng)
    g_times = _breaks(m, rng)
    f = list(range(n))
    g = list(range(m))

    py = hellinger_etw.etw_align(
        f, f_times, g, g_times, similarity_matrix=C, use_numba=False
    )
    nb = hellinger_etw.etw_align(
        f, f_times, g, g_times, similarity_matrix=C, use_numba=True
    )
    assert nb.score == pytest.approx(py.score, abs=2e-12)
