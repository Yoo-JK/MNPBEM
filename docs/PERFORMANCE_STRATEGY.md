# MNPBEM Python Port — Performance Optimization Strategy

Generated: 2026-04-23  
Status: Reference document for future optimization work

## Context

MATLAB → Python port의 MATLAB 대비 performance benchmark 결과, **수학적 정확성은 달성**(sphere 대부분 머신 정밀도, 최근 trirod RMS 1.06 → 2.2e-07로 5M× 개선)했으나 **실행 시간은 케이스별 0.11× ~ 90× 편차**. 특히 작은 matrix × 많은 wavelength 반복 구간에서 Python이 MATLAB 대비 5-10× 느림. 이 문서는 profile 기반 root cause 와 단계별 개선 로드맵.

## Benchmark 요약 (2026-04-23 기준)

MATLAB 대비 Python 성능 비율 (1.0 = 동등, >1 Python 빠름):

### Python 빠른 경우 (9 cases)
| 테스트 | Speedup | 원인 |
|---|---:|---|
| 01 MieStat | 89.7× | Analytical 벡터화 |
| 01 MieGans | 40.5× | 동일 |
| 04 BEMStat layer oblique | 9.21× | Sommerfeld tabulation 벡터화 |
| 13 trispheresegment | 8.17× | Small mesh 벡터 이득 |
| 01 MieRet | 7.1× | 벡터화 |
| 09 DipoleStat | 3.4× | Stat dipole 벡터화 |
| 09 Distance | 2.5× | Distance scan 벡터화 |
| 12 stat meshfield | 2.4× | - |
| 13 tricube | 1.9× | - |

### Python 느린 경우 (요주의)
| 테스트 | Speedup | Root cause (profile) |
|---|---:|---|
| 02 BEMStat sphere | **0.11×** | `scipy.linalg.lu_factor` × 41 wavelengths × wrapper overhead 170ms/call |
| 12 ret meshfield | **0.11×** | Per-point field eval 누적 |
| 13 trirod | 0.35× | 큰 mesh + ret coupling |
| 13 tritorus | 0.47× | 동일 |
| 09 DipoleRet | 0.55× | Retarded 전반 |
| 05 BEMRet layer | 0.65× | Sommerfeld integral heavy |
| 13 trispherescale | 0.69× | - |
| 03 BEMRet sphere | 0.76× | Retarded LU |

### Python-only 값 (MATLAB reference 없음 — test script 버그로 측정 불가)
| 테스트 | Python (s) | 비고 |
|---|---:|---|
| 06 mirror stat full / mirror | 75 / 2.5 | Mirror symmetry 30× speedup |
| 06 mirror ret full / mirror | 420 / 30 | Mirror symmetry 14× speedup |
| 07 eigenmode (BEMStatEig) | 2.7 | - |
| 08 iterative stat dir/iter | 2.6 / 2.0 | Iter 23% faster |
| 08 iterative ret dir/iter | 20.3 / 15.6 | Iter 23% faster |
| 08 iterative retlayer dir/iter | **392 / 354** | retlayer 매우 heavy |
| 09 DipoleStat/Ret/Distance | 2.9 / 32.6 / 1.6 | - |
| **10 DipoleRetLayer** | **3471 (≈58분)** | **가장 느린 케이스** ⚠️ |
| 10 DipoleStatLayer | 2.3 | - |
| 11 EELS stat/ret spectrum | 9.8 / 22.7 | - |
| 11 EELS map | 28.3 | - |
| 12 stat/ret nearfield | 1.9 / 18.1 | ret 9× slower than stat |

**주목:** `10 DipoleRetLayer` = **58분** — 사용자님 47nm dimer retarded + substrate + dipole 시뮬이 이 카테고리라면 1 configuration 당 1시간 소요. 병렬화 / JIT compile 강력 권장.

## Root Cause — Profile 분석

**중요 발견:** Python도 MKL 2025 사용 중 (`numpy.show_config`).  
→ BLAS/LAPACK backend 차이 **아님**.

### 병목 1: scipy wrapper overhead per call

`02_bemstat_sphere` cProfile (41 wavelengths):
```
ncalls  cumtime  percall  function
   41     7.01    0.171s  bem_stat.py:solve
   41     6.99    0.170s  bem_stat.py:_init_matrices
   41     6.96    0.170s  scipy.linalg.lu_factor   ← 99% of time
```

단독 benchmark 대비:
- `scipy.linalg.lu_factor` 284×284 real: 42 ms
- 실제 실행: 170 ms (4× 부풀림 — complex128 promotion + wrapper overhead 추정)

**MATLAB은 C-level 직접 MKL 호출** → wrapper overhead 거의 0.  
**Python scipy은 interpreter + check_finite + object wrapping** → 작은 연산에서 overhead 누적.

### 병목 2: Per-point evaluation (meshfield)

`12 ret meshfield`: 여러 field point × per-point Green function eval. Python loop 인터프리터 overhead 점별 누적.

### 병목 3: F 행렬 assembly (일부 케이스)

`greenfun/compgreen_stat.py`의 $O(N^2)$ Green function kernel — Python nested loop일 경우 수백-수천배 느림. 현재는 NumPy broadcasting 씀 (fast) but 복잡한 retarded kernel은 느려질 가능성.

## 개선 로드맵 (우선순위순)

### Tier 1 — Zero-effort fix (~10-20% gain)

**1. scipy 호출 플래그 최적화**

모든 `scipy.linalg.{lu_factor, lu_solve, solve}` 호출에 추가:
```python
lu_factor(A, check_finite=False, overwrite_a=True)
lu_solve(lu_piv, b, check_finite=False, overwrite_b=True)
solve(A, b, check_finite=False, overwrite_a=True, overwrite_b=True)
```

**적용 대상 파일:**
- `mnpbem/bem/bem_stat.py`
- `mnpbem/bem/bem_ret.py`
- `mnpbem/bem/bem_stat_eig.py`
- `mnpbem/bem/bem_ret_*.py` 계열

**주의:** `check_finite=False`는 NaN/Inf 검증 skip → Green function assembly에 NaN 발생 시 silent 전파. 기존 validation 다시 돌려 regression 확인 필수.

**예상 효과:** 02 BEMStat sphere 3.68s → 3.0s (~20%)

### Tier 2 — Refactor (며칠 작업, 2-3× gain)

**2. Batch wavelength 처리**

현재 각 파장마다 Python wrapper 호출 → 여러 wavelength RHS를 matrix로 stack:
```python
# 현재
for wl in wavelengths:
    sig = bem.solve(exc(p, wl))

# 개선 (L만 바뀌므로 factor는 여전히 per-wl, but solve는 batch 가능한 부분)
# Λ(ω)가 diagonal이라 Sherman-Morrison-Woodbury는 적용 어려움.
# 대신 exc.phip을 stack하고 block-solve로 call count 줄이기.
```

**3. 불필요한 array copy 제거**

`np.asarray`, `np.ascontiguousarray`, `np.copy` 등을 hot loop에서 제거. 특히 `_init_matrices` 내부.

**4. Iterative solver option (큰 mesh)**

```python
from scipy.sparse.linalg import gmres, LinearOperator

def solve_iterative(F, Lambda, rhs):
    M = LinearOperator((n, n), matvec=lambda x: (Lambda + F) @ x)
    # F만 precondition (파장 독립)
    P_lu = lu_factor(F, check_finite=False)
    P_inv = LinearOperator((n, n), matvec=lambda x: lu_solve(P_lu, x))
    sig, _ = gmres(M, rhs, M=P_inv, tol=1e-8)
    return sig
```

**예상 효과:** 03/05 BEMRet 1.5-3× (큰 mesh에서 강함)

### Tier 3 — JIT compile (1-2주, 5-50× gain on hot loops)

**5. Numba JIT for Green function kernel**

`mnpbem/greenfun/compgreen_ret.py` 등 retarded Green function 에서 per-element eval:

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

**적용 순서:**
1. `compgreen_stat.py` — simpler kernel 먼저
2. `compgreen_ret.py` — retarded
3. `compgreen_stat_layer.py` / `compgreen_ret_layer.py` — layer variants
4. `meshfield*.py` — per-point field evaluation

**예상 효과:**
- Green function assembly 50-100× faster
- Meshfield evaluation 10-50× faster
- 전체 BEM 30-50% 개선

**주의:** Numba는 first-call compile overhead 있음 (첫 호출 1-3초). Cache=True로 영속화.

### Tier 4 — Advanced (월 단위, 10-1000× gain, 연구 필요)

**6. Hierarchical matrix (H-matrix) for Green function**

큰 mesh (N > 1000)에서 $O(N^2) \to O(N \log N)$. Library: `h2tools`, `pyhml2d`.

**7. Fast Multipole Method (FMM)**

Same complexity reduction. Library: `pyfmmlib`, `FMMLIB3D`.

**8. GPU (CuPy)**

```python
import cupy as cp
F_gpu = cp.asarray(F)
for wl in wavelengths:
    sig = cp.linalg.solve(Lambda_gpu + F_gpu, phi_gpu)
```
NVIDIA GPU 필요. 큰 matrix에서 5-50×.

**9. C extension bypass scipy**

MATLAB 방식 (MKL 직접 호출). 가장 어렵지만 wrapper overhead 0.

## 즉시 측정 가능한 값 (benchmark 재실행시)

개선 후 확인할 metrics:

| 테스트 | 현재 (Apr 23) | Tier 1 후 예상 | Tier 3 후 예상 |
|---|---|---|---|
| 02 BEMStat sphere | 3.68s | 3.0s | 1.5s |
| 03 BEMRet sphere | 42.5s | 35s | 15-20s |
| 05 BEMRet layer | 71.6s | 60s | 25-30s |
| 12 ret meshfield | 18.1s | 15s | 3-5s |
| 13 trirod | 11.6s | 9s | 4-5s |

## 구현 우선순위 (가성비 기준)

**권장 순서:**
1. **Tier 1** (scipy 플래그) — 0.5일, 전역 10-20% gain
2. **Tier 3 step 5a** (compgreen_stat.py Numba) — 2-3일, BEM stat 30-50% gain
3. **Tier 3 step 5b** (compgreen_ret.py Numba) — 3-5일, BEM ret 30-50% gain
4. **Tier 3 step 5d** (meshfield Numba) — 2일, meshfield 10-50× gain
5. **Tier 2.4** (iterative solver) — 선택적, 큰 mesh 특화

나머지 (H-matrix, FMM, GPU)는 **연구 단계** 또는 **사용자 mesh 5000+ faces** 같은 극한 조건용.

## Profile 재현 방법

```python
import cProfile, pstats
import sys; sys.path.insert(0, '/home/yoojk20/workspace/MNPBEM')

# 테스트 code (예: BEMStat sphere)
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

시각화는 snakeviz:
```bash
pip install snakeviz
python -c "import cProfile; cProfile.run('run_test()', 'profile.out')"
snakeviz profile.out
```

## Regression 검증

모든 optimization 후 validation 재실행 필수:

```bash
cd /home/yoojk20/workspace/MNPBEM/validation
for dir in 01_mie 02_bemstat_sphere 03_bemret_sphere 04_bemstat_layer \
           05_bemret_layer 09_dipole 12_nearfield 13_shapes; do
    [ -f "$dir/run_python.py" ] && python "$dir/run_python.py" > "$dir/opt_run.log" 2>&1
done
python validation/summary/generate_summary.py
```

RMS error가 optimization 전후 **머신 정밀도 수준 유지**되는지 확인 (기존 1e-14 ~ 1e-7 그대로여야 함).

## 관련 파일

**수정 대상:**
- `mnpbem/bem/bem_stat.py`, `bem_ret.py`, `bem_stat_eig.py`
- `mnpbem/greenfun/compgreen_*.py` (Tier 3)
- `mnpbem/simulation/meshfield*.py` (Tier 3)

**validation 참조:**
- `validation/summary/summary_report.md` — 현재 성능 baseline
- `validation/13_shapes/data/*_matlab.csv` — 검증 기준 데이터
- `validation/summary/data/summary_data.csv` — RMS + timing 수치

## 히스토리 맥락

**2026-04-13 이전:** Python porting 정확도 우선, 성능 2차.  
**2026-04-13 ~ 04-22:** Mesh2d FP drift, MATLAB 정합 버그 수정. Trirod RMS 1.06 → 2.2e-07 (5M× 정확도 개선).  
**2026-04-23 (현재):** 정확도는 sphere/rod 모두 production-ready. **이제 성능 최적화 단계.**

## 참고

- NumPy config: MKL 2025 (backend 문제 아님)
- Python 3.11
- scipy 1.x
- 테스트 CPU: (benchmark 당시 환경 기록 필요)
- 주요 bottleneck: **interpreter overhead × 반복 횟수**
