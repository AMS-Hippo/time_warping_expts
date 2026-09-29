# Hellinger Elastic Time Warping experiments

This repository contains:

1. `hellinger_etw_original.py`: a direct implementation of the cubic recurrence;
2. `hellinger_etw.py`: the exact dense quadratic upper-envelope algorithm;
3. `hellinger_etw_banded.py`: the earlier dense-mask banded implementation, retained as an arbitrary-mask reference and finite-skip fallback;
4. `hellinger_etw_sparse.py`: a genuinely sparse no-skip band solver that evaluates and stores only allowed band cells;
5. `hellinger_etw_multiscale.py`: a dyadic coarse-to-fine driver using the sparse band solver;
6. `hellinger_graph_etw.py`: experimental tree/DAG-indexed variants; and
7. notebooks demonstrating the original APIs on synthetic data.

The code is associated with the paper *Time warping with Hellinger elasticity*.
It remains research code, but the core one-dimensional solvers now have an
independent randomized test suite.

## Quick setup

For the presentation example and notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open `examples/run_multiscale_slide_example.ipynb`; change `N` to any power of
two at least 32 and run all cells. See `RUN_MULTISCALE_EXAMPLE.md` for Windows
commands and the file to send back.

## Run the tests

```bash
python -m pip install -r requirements-dev.txt
pytest
```

See `DEVELOPMENT.md` for test scope and structural smoke benchmarks.

## Sparse banded solver

The sparse solver uses one contiguous half-open interval of allowed columns per
row.  For a band with `K` allowed cells, it evaluates `K` similarities and stores
`K` DP states rather than constructing dense `n x m` arrays.

```python
import numpy as np

from hellinger_etw_sparse import RowBand, etw_align_sparse_banded

n = 1_000
row = np.arange(n)
band = RowBand(
    lo=np.maximum(0, row - 4),
    hi=np.minimum(n, row + 5),
    m=n,
)

times = np.linspace(0.0, 1.0, n + 1)
values = np.sin(2.0 * np.pi * times[:-1])

result = etw_align_sparse_banded(
    values,
    times,
    values,
    times,
    band=band,
    similarity=lambda x, y: np.exp(-abs(x - y)),
)

print(result.score)
print(result.diagnostics.allowed_cells)
```

For numerical data, a batched `row_similarity` callback is preferable to the
scalar callback.  The sparse solver is exact for the band-constrained problem;
it equals the unrestricted optimum only when the band contains an unrestricted
optimal matching.

## Multiscale solver

The multiscale method groups adjacent intervals, solves a small coarsest problem
exactly, lifts the coarse matching to a fine corridor, and refines with the
sparse solver.  Numerical values are coarsened using duration-weighted means.
For observations in a general metric space, pass a custom `coarsen_values`
function.

```python
import numpy as np

from hellinger_etw_multiscale import etw_align_multiscale


def row_rbf(sigma):
    def similarity(x, ys):
        y = np.asarray(ys, dtype=float)
        delta = y - np.asarray(x, dtype=float)
        return np.exp(-0.5 * np.sum(delta * delta, axis=-1) / sigma**2)

    return similarity

n = 8_000
times = np.linspace(0.0, 1.0, n + 1)
values = np.column_stack(
    [np.sin(2 * np.pi * times[:-1]), np.cos(2 * np.pi * times[:-1])]
)

result = etw_align_multiscale(
    values,
    times,
    values,
    times,
    row_similarity=row_rbf(0.2),
    coarsest_size=32,
    initial_radius=4,
    adaptive=True,
)

print(result.score)
print(result.diagnostics.total_similarity_evaluations)
print(result.exact)  # True only if the finest solve expanded to the full grid.
```

At each refinement level the score is exact *inside the retained corridor*.
Adaptive score stability and freedom from artificial band-boundary contact are
useful diagnostics, but are not proofs that the unrestricted optimum has been
found.  Accepted radii are propagated to the next finer level so the algorithm
does not immediately return to a corridor already found to be too narrow.  The result is certified exact only when `result.exact` is true.

## Multiscale visualization

The recorded coarse-to-fine example used for the presentation illustration is
reproducible from the repository:

```bash
python -m pip install -r requirements-visualization.txt
python examples/multiscale_visualization.py
jupyter lab examples/multiscale_visualization.ipynb
```

The script writes `MULTISCALE_ILLUSTRATION_RUN.png` and
`MULTISCALE_ILLUSTRATION_RUN.json`. For a larger user-selected power-of-two
example, use `examples/run_multiscale_slide_example.ipynb`. It writes a
shareable ZIP containing the figures, JSON summary, and the exact row-band and
path arrays used to reconstruct the slide.

The examples use a fixed radius of four so that the hierarchy and retained
corridors are easy to inspect. Every refinement is exact inside its displayed
corridor; a fixed-radius result is not, in general, a certificate of the
unrestricted optimum.

## Three-method timing benchmark

For the presentation timing comparison, open:

```text
examples/run_three_method_benchmark.ipynb
```

Set `N_MAX` to a power of two and run all cells. The notebook compares the
direct cubic, exact quadratic, and adaptive multiscale methods in both a
favourable smooth/low-noise regime and a realistic stronger-warp/high-noise
regime. It writes raw and summarized timings, score-gap and sparse-work
diagnostics, PNG/SVG figures, system metadata, and a shareable return ZIP. See
`RUN_THREE_METHOD_BENCHMARK.md` for setup, safe dense-size caps, and timing
conventions.

## Smoke benchmarks

```bash
python benchmarks/sparse_band_smoke.py
python benchmarks/multiscale_infill_smoke.py --adaptive
```

The first benchmark guards against accidental dense allocation in the sparse
kernel.  The second records wall time, evaluated pair cells, and—at moderate
sizes—the score gap to the unrestricted dense algorithm.
