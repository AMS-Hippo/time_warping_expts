"""Small simulation helpers for Hellinger Elastic Time Warping demos.

The functions here intentionally have no dependency on the ETW implementation.
They generate timestamped 2D paths and RBF similarity matrices that can be passed
straight into ``hellinger_etw.etw_align`` or ``hellinger_etw_original.etw_align``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


ArrayLikeSeed = Optional[Union[int, np.random.Generator]]


@dataclass(frozen=True)
class SimulatedPair:
    """Two timestamped paths generated from the same latent curve."""

    f_values: np.ndarray
    f_times: np.ndarray
    g_values: np.ndarray
    g_times: np.ndarray
    latent_times: np.ndarray
    latent_values: np.ndarray


def _rng(seed: ArrayLikeSeed = None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def poisson_breakpoints(
    n: int,
    *,
    seed: ArrayLikeSeed = None,
    min_spacing: float = 0.0,
) -> np.ndarray:
    """Return ``n + 1`` breakpoints in ``[0, 1]`` with exponential spacings.

    Normalized exponential spacings are the interval lengths of a Poisson
    process conditioned on having ``n`` intervals.  ``min_spacing`` is optional
    regularization before normalization.
    """

    if n <= 0:
        raise ValueError("n must be positive.")
    rng = _rng(seed)
    spacings = rng.exponential(scale=1.0, size=n)
    if min_spacing < 0:
        raise ValueError("min_spacing must be nonnegative.")
    if min_spacing:
        spacings = spacings + float(min_spacing)
    out = np.empty(n + 1, dtype=np.float64)
    out[0] = 0.0
    out[1:] = np.cumsum(spacings)
    out /= out[-1]
    return out


def random_monotone_warp(
    t: np.ndarray,
    *,
    strength: float = 0.7,
    grid_size: int = 512,
    seed: ArrayLikeSeed = None,
) -> np.ndarray:
    """Apply a random increasing map ``[0, 1] -> [0, 1]`` to times ``t``."""

    if strength < 0:
        raise ValueError("strength must be nonnegative.")
    rng = _rng(seed)
    t = np.asarray(t, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, grid_size + 1)
    if strength == 0:
        return np.clip(t, 0.0, 1.0).copy()
    spacings = rng.lognormal(mean=-0.5 * strength * strength, sigma=strength, size=grid_size)
    warped = np.empty(grid_size + 1, dtype=np.float64)
    warped[0] = 0.0
    warped[1:] = np.cumsum(spacings)
    warped /= warped[-1]
    return np.interp(np.clip(t, 0.0, 1.0), grid, warped)


def latent_curve(
    *,
    kind: str = "integrated_random_walk",
    grid_size: int = 2048,
    dim: int = 2,
    seed: ArrayLikeSeed = None,
    velocity_scale: float = 1.0,
    acceleration_scale: float = 3.0,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a latent curve on a regular grid in ``[0, 1]``.

    ``kind`` may be ``"integrated_random_walk"``, ``"line"``, ``"circle"``,
    or ``"s_curve"``.  The integrated random walk is built by letting the
    velocity perform a random walk and integrating that velocity.
    """

    if grid_size < 2:
        raise ValueError("grid_size must be at least 2.")
    if dim < 1:
        raise ValueError("dim must be positive.")

    rng = _rng(seed)
    t = np.linspace(0.0, 1.0, grid_size)
    k = kind.lower().strip()

    if k == "integrated_random_walk":
        dt = 1.0 / (grid_size - 1)
        vel = np.empty((grid_size, dim), dtype=np.float64)
        vel[0] = rng.normal(scale=velocity_scale, size=dim)
        step_scale = acceleration_scale * np.sqrt(dt)
        for i in range(1, grid_size):
            vel[i] = vel[i - 1] + rng.normal(scale=step_scale, size=dim)
        x = np.empty((grid_size, dim), dtype=np.float64)
        x[0] = 0.0
        for i in range(1, grid_size):
            x[i] = x[i - 1] + vel[i - 1] * dt
    elif k == "line":
        x = np.zeros((grid_size, dim), dtype=np.float64)
        x[:, 0] = t
        if dim >= 2:
            x[:, 1] = 0.2 * t
    elif k == "circle":
        x = np.zeros((grid_size, dim), dtype=np.float64)
        x[:, 0] = np.cos(2.0 * np.pi * t)
        if dim >= 2:
            x[:, 1] = np.sin(2.0 * np.pi * t)
    elif k == "s_curve":
        x = np.zeros((grid_size, dim), dtype=np.float64)
        x[:, 0] = t
        if dim >= 2:
            x[:, 1] = np.sin(2.0 * np.pi * t)
    else:
        raise ValueError(
            "kind must be one of 'integrated_random_walk', 'line', 'circle', or 's_curve'."
        )

    if normalize:
        x = x - x.mean(axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(x, axis=1))
        if scale > 0:
            x = x / scale

    return t, x


def sample_curve(times: np.ndarray, latent_times: np.ndarray, latent_values: np.ndarray) -> np.ndarray:
    """Linearly sample ``latent_values`` at ``times``."""

    times = np.asarray(times, dtype=np.float64)
    latent_times = np.asarray(latent_times, dtype=np.float64)
    latent_values = np.asarray(latent_values, dtype=np.float64)
    return np.column_stack(
        [np.interp(times, latent_times, latent_values[:, d]) for d in range(latent_values.shape[1])]
    )


def simulate_pair(
    n: int,
    m: int,
    *,
    kind: str = "integrated_random_walk",
    seed: ArrayLikeSeed = None,
    dim: int = 2,
    noise: float = 0.03,
    warp_strength: float = 0.7,
    grid_size: int = 2048,
    min_spacing: float = 0.0,
) -> SimulatedPair:
    """Generate two timestamped paths from one latent curve.

    The two timestamp grids have Poisson/exponential spacings.  The ``g`` path
    is sampled after a random monotone time warp, then both paths receive
    independent Gaussian noise.
    """

    rng = _rng(seed)
    f_times = poisson_breakpoints(n, seed=rng, min_spacing=min_spacing)
    g_times = poisson_breakpoints(m, seed=rng, min_spacing=min_spacing)
    latent_times, latent_values = latent_curve(
        kind=kind, grid_size=grid_size, dim=dim, seed=rng
    )

    f_mid = 0.5 * (f_times[:-1] + f_times[1:])
    g_mid = 0.5 * (g_times[:-1] + g_times[1:])
    g_mid_warped = random_monotone_warp(g_mid, strength=warp_strength, seed=rng)

    f_values = sample_curve(f_mid, latent_times, latent_values)
    g_values = sample_curve(g_mid_warped, latent_times, latent_values)

    if noise:
        f_values = f_values + rng.normal(scale=noise, size=f_values.shape)
        g_values = g_values + rng.normal(scale=noise, size=g_values.shape)

    return SimulatedPair(
        f_values=f_values,
        f_times=f_times,
        g_values=g_values,
        g_times=g_times,
        latent_times=latent_times,
        latent_values=latent_values,
    )


def rbf_similarity_matrix(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sigma: Optional[float] = None,
) -> np.ndarray:
    """Return ``exp(-||x_i - y_j||^2 / (2 sigma^2))`` for two point clouds."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    diff = x[:, None, :] - y[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    if sigma is None:
        d = np.sqrt(d2)
        positive = d[d > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
    if sigma <= 0:
        raise ValueError("sigma must be positive.")
    return np.exp(-0.5 * d2 / (float(sigma) ** 2))


def rbf_similarity(*, sigma: float = 0.25):
    """Return a Python callable ``C(x, y)`` using the same RBF kernel."""

    if sigma <= 0:
        raise ValueError("sigma must be positive.")

    def C(x, y) -> float:
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        d2 = float(np.sum((x_arr - y_arr) ** 2))
        return float(np.exp(-0.5 * d2 / (sigma * sigma)))

    return C
