# MNPBEM Validation Summary
Total cases: 24
Total MATLAB time: 889.24 s
Total Python time: 762.92 s
Overall Python speedup: x1.17

## Results
| Test | MATLAB (s) | Python (s) | Speedup | Max RMS |
|------|-----------|------------|---------|---------|
| 01_mie/sphere | 0.868 | 0.081 | x10.74 | 3.18e-06 |
| 02_bemstat/rod | 4.665 | 6.424 | x0.73 | 2.21e-07 |
| 02_bemstat/sphere | 4.779 | 1.394 | x3.43 | 0.00e+00 |
| 03_bemret/rod | 40.949 | 38.326 | x1.07 | 3.75e-06 |
| 03_bemret/sphere | 6.422 | 6.698 | x0.96 | 4.28e-08 |
| 04_bemstat_layer/rod | 7.339 | 11.803 | x0.62 | 1.02e-01 |
| 04_bemstat_layer/sphere | 14.677 | 0.847 | x17.34 | 1.03e-02 |
| 05_bemret_layer/rod | 109.365 | 33.064 | x3.31 | 8.53e-02 |
| 05_bemret_layer/sphere | 43.663 | 16.685 | x2.62 | 6.12e-03 |
| 06_mirror/rod | 0.000 | 0.000 | x0.00 | N/A |
| 06_mirror/sphere | 320.264 | 496.606 | x0.64 | 2.13e-02 |
| 07_eigenmode/rod | 4.534 | 13.541 | x0.33 | 3.28e+13 |
| 07_eigenmode/sphere | 4.417 | 0.446 | x9.89 | 0.00e+00 |
| 08_iterative/rod | 168.828 | 60.458 | x2.79 | 3.43e-03 |
| 08_iterative/sphere | 60.819 | 15.327 | x3.97 | 0.00e+00 |
| 09_dipole/rod | 6.644 | 8.873 | x0.75 | 2.71e-05 |
| 09_dipole/sphere | 3.239 | 2.896 | x1.12 | 3.39e-07 |
| 10_dipole_layer/rod | 38.239 | 3.109 | x12.30 | 2.34e-01 |
| 10_dipole_layer/sphere | 21.122 | 3.899 | x5.42 | 9.48e-02 |
| 11_eels/rod | 9.248 | 12.566 | x0.74 | 1.03e-01 |
| 11_eels/sphere | 3.687 | 3.376 | x1.09 | 0.00e+00 |
| 12_nearfield/rod | 5.688 | 3.035 | x1.87 | 2.20e+01 |
| 12_nearfield/sphere | 3.394 | 1.832 | x1.85 | 4.07e+00 |
| 13_shapes | 6.389 | 21.631 | x0.30 | 1.24e-05 |

## Notes
- RMS is computed per matching `*_python.csv` / `*_matlab.csv` pair as `sqrt(mean((|py-ml|/max(|ml|,1e-30))^2))`, 최대값이 "Max RMS"로 보고됨.
- Timing은 각 폴더의 `python_timing.csv` / `matlab_timing.csv` 합계.
- `Speedup = MATLAB time / Python time`. >1 이면 Python이 빠름.
- 06_mirror/rod: trirod quarter-mesh 생성 한계로 skip (README 참고).
- 10_dipole_layer: Python BEMRetLayer 성능 이슈로 ret 파트 스크립트 내 skip 플래그.
