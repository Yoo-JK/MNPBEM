# PyMNPBEM — Performance Optimization Strategy

Generated: 2026-04-23  
Status: Reference document for future optimization work

## Context

In the performance benchmark of the MATLAB → Python port against MATLAB,
**mathematical accuracy was achieved** (most sphere cases at machine precision;
recently trirod RMS improved 1.06 → 2.2e-07, a 5M× improvement), but **the
execution time varies per case from 0.11× to 90×**. In particular, in the
regime of small matrices × many wavelength iterations, Python is 5-10× slower
than MATLAB. This document covers the profile-based root cause and a staged
improvement roadmap.

## Benchmark summary (as of 2026-04-23)

Python performance ratio relative to MATLAB (1.0 = parity, >1 Python faster):

### Cases where Python is faster (9 cases)
| Test | Speedup | Cause |
|---|---:|---|
| 01 MieStat | 89.7× | Analytical vectorization |
| 01 MieGans | 40.5× | Same |
| 04 BEMStat layer oblique | 9.21× | Sommerfeld tabulation vectorization |
| 13 trispheresegment | 8.17× | Small-mesh vector gain |
| 01 MieRet | 7.1× | Vectorization |
| 09 DipoleStat | 3.4× | Stat dipole vectorization |
| 09 Distance | 2.5× | Distance-scan vectorization |
| 12 stat meshfield | 2.4× | - |
| 13 tricube | 1.9× | - |

### Cases where Python is slower (needs attention)
| Test | Speedup | Root cause (profile) |
|---|---:|---|
| 02 BEMStat sphere | **0.11×** | `scipy.linalg.lu_factor` × 41 wavelengths × wrapper overhead 170ms/call |
| 12 ret meshfield | **0.11×** | Per-point field eval accumulation |
| 13 trirod | 0.35× | Large mesh + ret coupling |
| 13 tritorus | 0.47× | Same |
| 09 DipoleRet | 0.55× | Retarded across the board |
| 05 BEMRet layer | 0.65× | Sommerfeld integral heavy |
| 13 trispherescale | 0.69× | - |
| 03 BEMRet sphere | 0.76× | Retarded LU |

### Python-only values (no MATLAB reference — not measurable due to a test-script bug)
| Test | Python (s) | Note |
|---|---:|---|
| 06 mirror stat full / mirror | 75 / 2.5 | Mirror symmetry 30× speedup |
| 06 mirror ret full / mirror | 420 / 30 | Mirror symmetry 14× speedup |
| 07 eigenmode (BEMStatEig) | 2.7 | - |
| 08 iterative stat dir/iter | 2.6 / 2.0 | Iter 23% faster |
| 08 iterative ret dir/iter | 20.3 / 15.6 | Iter 23% faster |
| 08 iterative retlayer dir/iter | **392 / 354** | retlayer very heavy |
| 09 DipoleStat/Ret/Distance | 2.9 / 32.6 / 1.6 | - |
| **10 DipoleRetLayer** | **3471 (≈58 min)** | **slowest case** ⚠️ |
| 10 DipoleStatLayer | 2.3 | - |
| 11 EELS stat/ret spectrum | 9.8 / 22.7 | - |
| 11 EELS map | 28.3 | - |
| 12 stat/ret nearfield | 1.9 / 18.1 | ret 9× slower than stat |

**Note:** `10 DipoleRetLayer` = **58 min** — if the user's 47nm dimer retarded +
substrate + dipole simulation falls into this category, it takes 1 hour per
configuration. Parallelization / JIT compilation is strongly recommended.

## Root Cause — Profile analysis

**Important finding:** Python is also using MKL 2025 (`numpy.show_config`).  
→ This is **not** a BLAS/LAPACK backend difference.

### Bottleneck 1: scipy wrapper overhead per call

`02_bemstat_sphere` cProfile (41 wavelengths):
```
ncalls  cumtime  percall  function
   41     7.01    0.171s  bem_stat.py:solve
   41     6.99    0.170s  bem_stat.py:_init_matrices
   41     6.96    0.170s  scipy.linalg.lu_factor   ← 99% of time
```

Versus the standalone benchmark:
- `scipy.linalg.lu_factor` 284×284 real: 42 ms
- Actual run: 170 ms (4× inflation — presumably complex128 promotion + wrapper overhead)

**MATLAB calls MKL directly at the C level** → nearly 0 wrapper overhead.  
**Python scipy has interpreter + check_finite + object wrapping** → overhead accumulates on small operations.

### Bottleneck 2: Per-point evaluation (meshfield)

`12 ret meshfield`: multiple field points × per-point Green function eval. Python-loop interpreter overhead accumulates point by point.

### Bottleneck 3: F matrix assembly (some cases)

The $O(N^2)$ Green function kernel in `greenfun/compgreen_stat.py` — hundreds to thousands of times slower if written as a Python nested loop. It currently uses NumPy broadcasting (fast), but the complex retarded kernel could become slow.

## Improvement roadmap (in priority order)

### Tier 1 — Zero-effort fix (~10-20% gain)

**1. Optimize scipy call flags**

Add to every `scipy.linalg.{lu_factor, lu_solve, solve}` call:
```python
lu_factor(A, check_finite=False, overwrite_a=True)
lu_solve(lu_piv, b, check_finite=False, overwrite_b=True)
solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True)
```

**Target files:**
- `mnpbem/bem/bem_stat.py`
- `mnpbem/bem/bem_ret.py`
- `mnpbem/bem/bem_stat_eig.py`
- `mnpbem/bem/bem_ret_*.py` family

**Caution:** `check_finite=False` skips NaN/Inf validation → if a NaN arises in the Green function assembly it propagates silently. Re-running the existing validation to confirm no regression is mandatory.

**Expected effect:** 02 BEMStat sphere 3.68s → 3.0s (~20%)

### Tier 2 — Refactor (a few days' work, 2-3× gain)

**2. Batch wavelength processing**

Currently a Python wrapper is called for each wavelength → stack the several wavelength RHS into a matrix:
```python
# current
for wl in wavelengths:
    sig = bem.solve(exc(p, wl))

# improved (only L changes, so the factor is still per-wl, but the solve part can be batched)
# Since Λ(ω) is diagonal, Sherman-Morrison-Woodbury is hard to apply.
# Instead, stack exc.phip and reduce the call count with a block-solve.
```

**3. Remove unnecessary array copies**

Remove `np.asarray`, `np.ascontiguousarray`, `np.copy`, etc. from the hot loop. Especially inside `_init_matrices`.

**4. Iterative solver option (large mesh)**

```python
from scipy.sparse.linalg import gmres, LinearOperator

def solve_iterative(F, Lambda, rhs):
    M = LinearOperator((n, n), matvec=lambda x: (Lambda + F) @ x)
    # precondition on F only (wavelength-independent)
    P_lu = lu_factor(F, check_finite=False)
    P_inv = LinearOperator((n, n), matvec=lambda x: lu_solve(P_lu, x))
    sig, _ = gmres(M, rhs, M=P_inv, tol=1e-8)
    return sig
```

**Expected effect:** 03/05 BEMRet 1.5-3× (strong on large meshes)

### Tier 3 — JIT compile (1-2 weeks, 5-50× gain on hot loops)

**5. Numba JIT for Green function kernel**

Per-element eval in retarded Green functions such as `mnpbem/greenfun/compgreen_ret.py`:

```python
from numba import jit, prange

@jit(nopython=True, parallel=True, fastmath=True, cache=True)
def green_ret_kernel(positions, normals, areas, k):
    n = positions.shape[0]
    G = np.zeros((n, n), dtype=np.complex128)
    F = np.zeros((n, n), dtype=np.complex128)
    for i in prange(n):
        for j in range(n):
            if i == j:
                continue
            r_vec = positions[i] - positions[j]
            d = np.linalg.norm(r_vec)
            phase = np.exp(1j * k * d)
            G[i, j] = phase / d * areas[j]
            F[i, j] = -phase * (1 - 1j*k*d) * np.dot(normals[i], r_vec) / d**3 * areas[j]
    return G, F
```

**Application order:**
1. `compgreen_stat.py` — simpler kernel first
2. `compgreen_ret.py` — retarded
3. `compgreen_stat_layer.py` / `compgreen_ret_layer.py` — layer variants
4. `meshfield*.py` — per-point field evaluation

**Expected effect:**
- Green function assembly 50-100× faster
- Meshfield evaluation 10-50× faster
- Overall BEM 30-50% improvement

**Caution:** Numba has first-call compile overhead (1-3 s on the first call). Persist it with Cache=True.

### Tier 4 — Advanced (months, 10-1000× gain, requires research)

**6. Hierarchical matrix (H-matrix) for Green function**

On large meshes (N > 1000), $O(N^2) \to O(N \log N)$. Library: `h2tools`, `pyhml2d`.

**7. Fast Multipole Method (FMM)**

Same complexity reduction. Library: `pyfmmlib`, `FMMLIB3D`.

**8. GPU (CuPy)**

```python
import cupy as cp
F_gpu = cp.asarray(F)
for wl in wavelengths:
    sig = cp.linalg.solve(Lambda_gpu + F_gpu, phi_gpu)
```
Requires an NVIDIA GPU. 5-50× on large matrices.

**9. C extension bypassing scipy**

The MATLAB way (direct MKL call). Hardest but 0 wrapper overhead.

## Immediately measurable values (when re-running the benchmark)

Metrics to check after improvement:

| Test | Current (Apr 23) | Expected after Tier 1 | Expected after Tier 3 |
|---|---|---|---|
| 02 BEMStat sphere | 3.68s | 3.0s | 1.5s |
| 03 BEMRet sphere | 42.5s | 35s | 15-20s |
| 05 BEMRet layer | 71.6s | 60s | 25-30s |
| 12 ret meshfield | 18.1s | 15s | 3-5s |
| 13 trirod | 11.6s | 9s | 4-5s |

## Implementation priority (by cost-effectiveness)

**Recommended order:**
1. **Tier 1** (scipy flags) — 0.5 day, global 10-20% gain
2. **Tier 3 step 5a** (compgreen_stat.py Numba) — 2-3 days, BEM stat 30-50% gain
3. **Tier 3 step 5b** (compgreen_ret.py Numba) — 3-5 days, BEM ret 30-50% gain
4. **Tier 3 step 5d** (meshfield Numba) — 2 days, meshfield 10-50× gain
5. **Tier 2.4** (iterative solver) — optional, specialized for large meshes

The rest (H-matrix, FMM, GPU) are for the **research stage** or extreme
conditions such as **user meshes with 5000+ faces**.

## Profile reproduction method

```python
import cProfile, pstats
import sys; sys.path.insert(0, '.')

# test code (e.g. BEMStat sphere)
def run_test():
    from mnpbem.materials import EpsConst, EpsTable
    from mnpbem.geometry import trisphere, ComParticle
    from mnpbem.bem import BEMStat
    from mnpbem.simulation import PlaneWaveStat
    import numpy as np
    eps_tab = [EpsConst(1.0), EpsTable('gold.dat')]
    p = ComParticle(eps_tab, [trisphere(144, 20.0)], [[2, 1]])
    bem = BEMStat(p)
    exc = PlaneWaveStat([1, 0, 0])
    enei = np.linspace(400, 800, 41)
    for e in enei:
        sig, bem = bem.solve(exc(p, e))
        _ = exc.extinction(sig)

pr = cProfile.Profile()
pr.enable()
run_test()
pr.disable()
pstats.Stats(pr).sort_stats('cumtime').print_stats(20)
```

For visualization, snakeviz:
```bash
pip install snakeviz
python -c "import cProfile; cProfile.run('run_test()', 'profile.out')"
snakeviz profile.out
```

## Regression verification

Re-running the validation after every optimization is mandatory:

```bash
cd validation
for dir in 01_mie 02_bemstat_sphere 03_bemret_sphere 04_bemstat_layer \
           05_bemret_layer 09_dipole 12_nearfield 13_shapes; do
    [ -f "$dir/run_python.py" ] && python "$dir/run_python.py" > "$dir/opt_run.log" 2>&1
done
python validation/summary/generate_summary.py
```

Confirm that the RMS error **stays at machine-precision level** before and after optimization (the existing 1e-14 ~ 1e-7 must remain unchanged).

## Related files

**Modification targets:**
- `mnpbem/bem/bem_stat.py`, `bem_ret.py`, `bem_stat_eig.py`
- `mnpbem/greenfun/compgreen_*.py` (Tier 3)
- `mnpbem/simulation/meshfield*.py` (Tier 3)

**validation references:**
- `validation/summary/summary_report.md` — current performance baseline
- `validation/13_shapes/data/*_matlab.csv` — validation reference data
- `validation/summary/data/summary_data.csv` — RMS + timing figures

## Historical context

**Before 2026-04-13:** Python porting prioritized accuracy, performance second.  
**2026-04-13 ~ 04-22:** Mesh2d FP drift, MATLAB-parity bug fixes. Trirod RMS 1.06 → 2.2e-07 (5M× accuracy improvement).  
**2026-04-23 (current):** Accuracy is production-ready for both sphere and rod. **Now in the performance optimization stage.**

## Notes

- NumPy config: MKL 2025 (not a backend problem)
- Python 3.11
- scipy 1.x
- Test CPU: (the environment at benchmark time needs to be recorded)
- Main bottleneck: **interpreter overhead × iteration count**
