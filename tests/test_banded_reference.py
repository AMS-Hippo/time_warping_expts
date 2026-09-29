from __future__ import annotations

import numpy as np
import pytest

import hellinger_etw_banded
from tests.reference_banded import brute_force_banded_score, mask_from_path_with_noise


def _breaks_from_lengths(lengths: np.ndarray) -> np.ndarray:
    x = np.asarray(lengths, dtype=float)
    x = x / x.sum()
    return np.concatenate(([0.0], np.cumsum(x)))


@pytest.mark.parametrize("seed", range(30))
def test_arbitrary_mask_matches_independent_recurrence(seed: int) -> None:
    rng = np.random.default_rng(2000 + seed)
    n = int(rng.integers(1, 8))
    m = int(rng.integers(1, 8))
    ds = rng.uniform(0.1, 1.0, size=n)
    dt = rng.uniform(0.1, 1.0, size=m)
    ds /= ds.sum()
    dt /= dt.sum()
    C = rng.random((n, m))
    mask = mask_from_path_with_noise(n, m, rng, extra_probability=0.3)
    reference = brute_force_banded_score(C, ds, dt, mask)

    result = hellinger_etw_banded.etw_align_banded(
        list(range(n)),
        _breaks_from_lengths(ds),
        list(range(m)),
        _breaks_from_lengths(dt),
        similarity_matrix=C,
        band_mask=mask,
        use_numba=False,
    )
    assert result.score == pytest.approx(reference.score, abs=3e-12)
    assert all(mask[i, j] for i, j in result.pairs)


def test_numba_arbitrary_mask_matches_reference() -> None:
    rng = np.random.default_rng(991)
    n, m = 11, 9
    ds = rng.uniform(0.1, 1.0, size=n)
    dt = rng.uniform(0.1, 1.0, size=m)
    ds /= ds.sum()
    dt /= dt.sum()
    C = rng.random((n, m))
    mask = mask_from_path_with_noise(n, m, rng, extra_probability=0.2)
    reference = brute_force_banded_score(C, ds, dt, mask)

    result = hellinger_etw_banded.etw_align_banded(
        list(range(n)),
        _breaks_from_lengths(ds),
        list(range(m)),
        _breaks_from_lengths(dt),
        similarity_matrix=C,
        band_mask=mask,
        use_numba=True,
    )
    assert result.score == pytest.approx(reference.score, abs=3e-12)
