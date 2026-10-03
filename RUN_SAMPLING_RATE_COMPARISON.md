# Run the sampling-rate comparison notebook

## Linux or macOS

```bash
cd time_warping_expts-main
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab examples/sampling_rate_comparison.ipynb
```

## Windows PowerShell

```powershell
cd time_warping_expts-main
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab examples/sampling_rate_comparison.ipynb
```

The default example uses 80 intervals and two sampling experiments with opposite
exponential density profiles.  The continuous curves and true warp are identical
in both experiments.  Only the sample locations change.

The notebook writes a bundle under `user_runs/sampling_rate_comparison/` with
PNG/SVG figures, JSON/CSV summaries, and the recovered warp arrays.
