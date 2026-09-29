# Development and validation

## Test suite

Run the complete suite from the repository root:

```bash
pytest
```

The tests currently cover:

- equality of the original cubic, dense quadratic, and full-band algorithms;
- finite-skip equality on the full grid;
- arbitrary-mask banded scores against an independent cubic recurrence;
- the sparse row-band solver against the same independent recurrence;
- Python/Numba equality;
- traceback partition and score checks;
- exact similarity-call accounting for sparse bands;
- dyadic hierarchy and duration-weighted coarsening;
- lifting coarse matchings to sparse fine-level bands;
- full-grid multiscale equality with the unrestricted dense solver;
- scalar versus batched-row similarity interfaces;
- adaptive score monotonicity across nested bands; and
- validation of malformed, infeasible, or unsupported inputs.

## Sparse band representation

`hellinger_etw_sparse.py` stores one half-open column interval per row:

```python
from hellinger_etw_sparse import RowBand

band = RowBand(lo, hi, m)
```

If the band contains `K = sum(hi - lo)` cells, the no-skip solver evaluates and
stores only those `K` cells.  Dense masks and finite skip penalties remain in
`hellinger_etw_banded.py` as a reference/fallback path.

## Multiscale hierarchy

`hellinger_etw_multiscale.py` uses the following sequence:

1. group adjacent intervals in pairs, preserving exact interval endpoints;
2. aggregate numerical values by duration-weighted means;
3. solve the coarsest level exactly on its full grid;
4. lift each matched coarse pair to the Cartesian product of its child ranges;
5. dilate the lifted path into a `RowBand`;
6. solve the original no-skip recurrence exactly inside that band; and
7. optionally double the radius until score and boundary diagnostics stabilize.

Adaptive stabilization is heuristic.  A distant alternative optimum can remain
outside two successive corridors even when their scores agree.  A full-grid
finest solve is the only certificate of unrestricted exactness.

## Manual scaling smoke tests

```bash
python benchmarks/sparse_band_smoke.py
python benchmarks/multiscale_infill_smoke.py --adaptive
```

The sparse benchmark uses a width-five diagonal corridor at sizes up to 32,000.
The multiscale benchmark reports evaluated pair cells and score gaps under a
synthetic infill experiment.  These are structural development checks, not
publication-quality performance studies.
