# MNPBEM Validation Summary Report

Generated: 2026-04-13

## Overview

This report summarizes the validation of the Python MNPBEM implementation
against the original MATLAB MNPBEM toolbox.

- **Total tests**: 15
- **Tests with MATLAB-Python comparison**: 15
- **Tests passing (RMS < 1%)**: 7
- **Near-exact match (RMS < 1e-6)**: 3

## Results Table

| Test | MATLAB (s) | Python (s) | Speedup | RMS Error |
|------|-----------|-----------|---------|----------|
| 01 MieStat | 0.1785 | 0.0016 | 110.47x | 4.64e-16 |
| 01 MieRet | 0.2143 | 0.0282 | 7.60x | 2.41e-15 |
| 01 MieGans | 0.1880 | 0.0044 | 42.89x | 3.18e-06 |
| 02 BEMStat sphere | 0.4129 | 0.3213 | 1.29x | 7.22e-15 |
| 03 BEMRet sphere | 32.4997 | 17.1170 | 1.90x | 1.10e-02 |
| 04 BEMStat layer (normal) | 3.7269 | 0.7782 | 4.79x | 5.65e-03 |
| 04 BEMStat layer (oblique) | 13.9108 | 1.8586 | 7.48x | 1.11e-02 |
| 05 BEMRet layer | 46.6276 | 39.4643 | 1.18x | 3.89e-01 |
| 13 trisphere | 0.5867 | 0.5122 | 1.15x | 9.19e-03 |
| 13 trirod | 4.0539 | 8.3606 | 0.48x | 1.06e+00 |
| 13 tricube | 2.1700 | 2.7031 | 0.80x | 3.50e-01 |
| 13 tritorus | 1.4521 | 0.6401 | 2.27x | 5.11e-01 |
| 13 trispheresegment | 0.1844 | 0.0208 | 8.87x | 8.72e-01 |
| 13 trispherescale | 0.6990 | 1.0032 | 0.70x | 7.44e-03 |
| 13 tripolygon | 0.3817 | 0.1337 | 2.85x | 6.36e-01 |

## Timing Comparison

- **Total MATLAB time**: 107.29 s
- **Total Python time**: 72.95 s
- **Overall speedup**: 1.47x

## Figures

- `summary_table.png`: Summary table with all tests
- `summary_bar.png`: RMS relative error bar chart
- `summary_timing.png`: MATLAB vs Python timing comparison

## Notes

- RMS error is computed as the root mean square of the relative error
  |Python - MATLAB| / max(|MATLAB|, 1e-30) across all wavelength points.
- For tests with multiple output columns (e.g., extinction, scattering),
  the maximum RMS error across all columns is reported.
- Speedup = MATLAB time / Python time. Values > 1 mean Python is faster.
- Tests 06-12 have Python-only data (no MATLAB reference for comparison).
