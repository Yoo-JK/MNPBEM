# M4 Final Report

**Total demos**: 72

## User requirements

1. **GPU acceleration as opt-in** - DONE (MNPBEM_GPU=1 explicit; default OFF)
2. **Python (CPU) > MATLAB on every demo** - 7/10 demos faster on Python CPU (median 6.06x speedup)
   - MATLAB timing is being collected sequentially in the background;
     this number grows as more demos complete.
3. **MATLAB == Python (1e-12)** - perf=71, OK=1, warn=0, BAD=0 (of 72 demos)

## Timing summary

- Python CPU vs MATLAB: 7/10 demos Python is faster
- Python GPU vs MATLAB: 10/10 demos Python+GPU is faster

## Plots

![Dashboard](plots/dashboard.png)

![Timing overview](plots/timing_overview.png)

![Accuracy](plots/accuracy_overview.png)

## Per-demo detail

See `summary_table.md` for full table; `plots/per_demo/` for per-demo spectrum + timing plots.

## Demos where MATLAB still wins (CPU)

| Demo | MATLAB (s) | PyCPU (s) | CPU/MATLAB |
|---|---|---|---|
| demodipstat11 | 350.40 | 900.23 | 0.39 |
| demodipstat10 | 192.00 | 401.41 | 0.48 |
| demodipstat4 | 67.50 | 69.29 | 0.97 |