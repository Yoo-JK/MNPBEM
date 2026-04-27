# M4 Final Report

**Total demos analyzed**: 72

## TL;DR (사용자 요구 3가지 처리 상태)

1. ✅ **GPU acceleration = opt-in** — `MNPBEM_GPU=1` 명시적 (default OFF)
2. **Python (CPU) > MATLAB on every demo** — **27/31** demos faster on Python CPU (median 3.17× speedup)
   - MATLAB timing은 sequential하게 백그라운드에서 수집되며, 더 진행될수록 비교 demo 수가 증가합니다.
   - Python GPU vs MATLAB: **31/31** demos faster on Python+GPU
3. ✅ **MATLAB == Python (1e-12)** — perf=71, OK=1, warn=0, BAD=0 (of 72 demos)

## Timing summary

- Python CPU vs MATLAB: 27/31 demos Python is faster
- Python GPU vs MATLAB: 31/31 demos Python+GPU is faster

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
| demoeelsret5 | 106.70 | 112.58 | 0.95 |
| demodipstat4 | 67.50 | 69.29 | 0.97 |