# M4 Summary Table

| Demo | Type | MATLAB (s) | PyCPU (s) | PyGPU (s) | CPU/MATLAB | GPU/MATLAB | max_rel_err | class |
|---|---|---|---|---|---|---|---|---|
| demodipret1 | 1d_series | - | 6.54 | 3.57 |  - |  - | 5.58e-15 | perf |
| demodipret10 | 1d_series | 704.90 | 580.03 | 323.94 | :green_circle: 1.22 | :green_circle: 2.18 | 1.29e-09 | perf |
| demodipret11 | 1d_series | 93.70 | 7.23 | 7.09 | :green_circle: 12.95 | :green_circle: 13.21 | 2.14e-14 | perf |
| demodipret12 | 1d_series | 141.80 | 8.32 | 6.11 | :green_circle: 17.04 | :green_circle: 23.20 | 4.88e-14 | perf |
| demodipret2 | 1d_series | - | 37.14 | 45.21 |  - |  - | 2.88e-13 | perf |
| demodipret3 | 1d_series | 885.50 | 103.38 | 87.18 | :green_circle: 8.57 | :green_circle: 10.16 | 4.50e-14 | perf |
| demodipret4 | 1d_series | 627.80 | 176.53 | 175.63 | :green_circle: 3.56 | :green_circle: 3.57 | 2.40e-14 | perf |
| demodipret5 | face | 148.80 | 7.36 | 6.24 | :green_circle: 20.22 | :green_circle: 23.85 | 1.91e-13 | perf |
| demodipret6 | 1d_series | 232.50 | - | - |  - |  - | 1.32e-14 | perf |
| demodipret7 | face | 107.60 | - | - |  - |  - | 5.18e-13 | perf |
| demodipret8 | 1d_series | - | 46.59 | 52.08 |  - |  - | 3.76e-10 | perf |
| demodipret9 | 1d_series | 108.80 | - | - |  - |  - | 3.46e-09 | perf |
| demodipstat1 | 1d_series | - | - | - |  - |  - | 3.40e-15 | perf |
| demodipstat10 | 1d_series | 192.00 | 401.41 | 99.43 | :red_circle: 0.48 | :green_circle: 1.93 | 8.55e-15 | perf |
| demodipstat11 | 1d_series | 350.40 | 900.23 | 179.74 | :red_circle: 0.39 | :green_circle: 1.95 | 2.15e-13 | perf |
| demodipstat2 | 1d_series | 73.70 | - | - |  - |  - | 3.34e-15 | perf |
| demodipstat3 | face | 8.40 | - | - |  - |  - | 8.89e-15 | perf |
| demodipstat4 | 1d_series | 67.50 | 69.29 | 37.52 | :red_circle: 0.97 | :green_circle: 1.80 | 7.63e-13 | perf |
| demodipstat5 | 1d_series | 260.50 | 9.54 | 7.07 | :green_circle: 27.32 | :green_circle: 36.86 | 3.08e-15 | perf |
| demodipstat6 | face | 199.40 | - | - |  - |  - | 5.34e-15 | perf |
| demodipstat7 | 1d_series | 265.80 | 123.89 | 32.98 | :green_circle: 2.15 | :green_circle: 8.06 | 2.28e-14 | perf |
| demodipstat8 | face | 71.50 | 16.30 | 7.48 | :green_circle: 4.39 | :green_circle: 9.56 | 1.92e-14 | perf |
| demodipstat9 | 1d_series | 44.60 | 22.63 | 6.22 | :green_circle: 1.97 | :green_circle: 7.18 | 5.28e-15 | perf |
| demoeelsret1 | 1d_series | - | 28.87 | 14.03 |  - |  - | 1.66e-14 | perf |
| demoeelsret2 | 1d_series | 445.60 | 124.21 | 40.54 | :green_circle: 3.59 | :green_circle: 10.99 | 6.36e-14 | perf |
| demoeelsret3 | face | 20.00 | 12.11 | 4.88 | :green_circle: 1.65 | :green_circle: 4.10 | 3.17e-13 | perf |
| demoeelsret4 | 1d_series | 214.50 | 90.60 | 28.26 | :green_circle: 2.37 | :green_circle: 7.59 | 2.89e-14 | perf |
| demoeelsret5 | 2d_spatial | 106.70 | 112.58 | 35.08 | :red_circle: 0.95 | :green_circle: 3.04 | 3.61e-14 | perf |
| demoeelsret6 | face | 34.40 | 21.95 | 8.37 | :green_circle: 1.57 | :green_circle: 4.11 | 2.05e-13 | perf |
| demoeelsret7 | 1d_series | 1200.13 | 469.17 | 327.53 | :green_circle: 2.56 | :green_circle: 3.66 | 9.97e-08 | perf |
| demoeelsret8 | 2d_spatial | 1200.14 | 714.38 | 220.25 | :green_circle: 1.68 | :green_circle: 5.45 | 1.86e-07 | perf |
| demoeelsstat1 | 1d_series | - | 5.28 | 1.46 |  - |  - | 1.22e-14 | perf |
| demoeelsstat2 | 2d_impact | 1200.16 | - | - |  - |  - | 2.18e-08 | perf |
| demoeelsstat3 | 2d_impact | 1200.14 | 184.78 | 19.40 | :green_circle: 6.49 | :green_circle: 61.86 | 1.53e-14 | perf |
| demospecret1 | 1d_series | - | 20.44 | 3.91 |  - |  - | 6.12e-15 | perf |
| demospecret10 | face | 363.00 | 114.39 | 18.16 | :green_circle: 3.17 | :green_circle: 19.99 | 1.21e-04 | OK |
| demospecret11 | 1d_series | 1200.13 | 900.18 | 146.34 | :green_circle: 1.33 | :green_circle: 8.20 | 4.03e-07 | perf |
| demospecret12 | 1d_series | 666.30 | 113.13 | 12.86 | :green_circle: 5.89 | :green_circle: 51.81 | 1.18e-16 | perf |
| demospecret13 | 1d_series | 1200.13 | 862.04 | 396.20 | :green_circle: 1.39 | :green_circle: 3.03 | 2.89e-08 | perf |
| demospecret14 | 1d_series | 1200.10 | 900.57 | 733.50 | :green_circle: 1.33 | :green_circle: 1.64 | 2.68e-07 | perf |
| demospecret15 | 1d_series | - | 26.02 | 6.08 |  - |  - | 5.34e-14 | perf |
| demospecret16 | 1d_series | 1200.13 | 327.63 | 81.99 | :green_circle: 3.66 | :green_circle: 14.64 | 2.79e-14 | perf |
| demospecret17 | 1d_series | 186.50 | 33.18 | 10.49 | :green_circle: 5.62 | :green_circle: 17.77 | 1.00e-14 | perf |
| demospecret18 | 1d_series | 1200.13 | 186.24 | 117.92 | :green_circle: 6.44 | :green_circle: 10.18 | 8.88e-13 | perf |
| demospecret2 | 1d_series | - | 41.22 | 10.29 |  - |  - | 3.20e-14 | perf |
| demospecret3 | face | 1200.14 | 47.27 | 7.31 | :green_circle: 25.39 | :green_circle: 164.11 | 6.92e-14 | perf |
| demospecret4 | 1d_series | 1200.14 | 159.52 | 57.35 | :green_circle: 7.52 | :green_circle: 20.93 | 7.50e-14 | perf |
| demospecret5 | 1d_series | - | 165.13 | 55.08 |  - |  - | 5.68e-13 | perf |
| demospecret6 | 1d_series | 1200.14 | - | - |  - |  - | 1.62e-06 | perf |
| demospecret7 | 1d_series | - | 159.20 | 58.48 |  - |  - | 4.97e-08 | perf |
| demospecret8 | 1d_series | 138.00 | 136.26 | 27.79 | :green_circle: 1.01 | :green_circle: 4.97 | 2.88e-08 | perf |
| demospecret9 | 1d_series | 927.10 | 128.89 | 27.31 | :green_circle: 7.19 | :green_circle: 33.94 | 4.77e-08 | perf |
| demospecstat1 | 1d_series | - | - | - |  - |  - | 1.03e-14 | perf |
| demospecstat10 | 1d_series | 1200.14 | 13.50 | 2.92 | :green_circle: 88.92 | :green_circle: 411.47 | 7.23e-13 | perf |
| demospecstat11 | face | 1200.14 | - | - |  - |  - | 3.37e-15 | perf |
| demospecstat12 | 1d_series | 1200.16 | 22.38 | 2.77 | :green_circle: 53.63 | :green_circle: 432.60 | 6.30e-15 | perf |
| demospecstat13 | face | 1200.14 | 25.51 | 3.88 | :green_circle: 47.05 | :green_circle: 309.31 | 3.86e-15 | perf |
| demospecstat14 | 1d_series | 1200.12 | - | - |  - |  - | 1.09e-14 | perf |
| demospecstat15 | 1d_series | 1200.13 | 157.80 | 10.78 | :green_circle: 7.61 | :green_circle: 111.36 | 9.78e-15 | perf |
| demospecstat16 | 1d_series | 1137.30 | 100.20 | 7.26 | :green_circle: 11.35 | :green_circle: 156.57 | 1.95e-15 | perf |
| demospecstat17 | 1d_series | 1200.14 | 345.78 | 359.77 | :green_circle: 3.47 | :green_circle: 3.34 | 7.90e-14 | perf |
| demospecstat18 | 1d_series | 1200.07 | 65.24 | 30.89 | :green_circle: 18.39 | :green_circle: 38.85 | 3.61e-14 | perf |
| demospecstat19 | 1d_series | 1200.13 | 21.42 | 3.05 | :green_circle: 56.03 | :green_circle: 393.53 | 3.50e-14 | perf |
| demospecstat2 | 1d_series | - | 5.31 | 1.77 |  - |  - | 1.06e-14 | perf |
| demospecstat20 | 1d_series | - | 99.78 | 44.71 |  - |  - | 5.39e-14 | perf |
| demospecstat3 | 1d_series | - | - | - |  - |  - | 1.63e-14 | perf |
| demospecstat4 | face | - | - | - |  - |  - | 8.01e-15 | perf |
| demospecstat5 | 1d_series | - | 19.85 | 2.80 |  - |  - | 7.39e-15 | perf |
| demospecstat6 | face | - | 12.35 | 2.17 |  - |  - | 3.73e-15 | perf |
| demospecstat7 | 1d_series | - | 17.52 | 3.58 |  - |  - | 9.68e-15 | perf |
| demospecstat8 | 1d_series | - | 31.82 | 8.02 |  - |  - | 3.72e-13 | perf |
| demospecstat9 | face | - | 12.86 | 4.40 |  - |  - | 5.05e-15 | perf |