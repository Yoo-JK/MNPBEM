# M4 Final Report

**Total demos**: 72

## User requirements

1. **GPU acceleration as opt-in** - DONE (MNPBEM_GPU=1 explicit)
2. **Python (CPU) > MATLAB on every demo** - 4/4 demos faster on Python CPU (median 10.76x speedup)
3. **MATLAB == Python (1e-12)** - perf=71 OK=1 warn=0 BAD=0

## Timing summary

- Python CPU vs MATLAB: 4/4 demos Python is faster
- Python GPU vs MATLAB: 0/0 demos Python+GPU is faster

## Plots

![Dashboard](plots/dashboard.png)

![Timing overview](plots/timing_overview.png)

![Accuracy](plots/accuracy_overview.png)

## Per-demo detail

See `summary_table.md` for full table; `plots/per_demo/` for per-demo spectrum + timing plots.
